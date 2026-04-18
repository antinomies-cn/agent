import asyncio
import time
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.deps import resolve_runtime_dependency
from app.core.config import build_config_health_summary, get_qdrant_settings, get_rerank_gateway_settings
from app.core.embedding_config import resolve_embedding_config
from app.core.gateway_resilience import get_resilience_snapshot
from app.core.logger_setup import logger, mask_session_id, summarize_error_for_log

router = APIRouter()


class EmbeddingConfigResponse(BaseModel):
    model: str = Field(description="当前生效的 embedding 模型别名")
    dimensions: int = Field(description="当前生效的向量维度")
    dimension_source: str = Field(description="维度来源")
    collection: str = Field(description="当前 Qdrant 目标集合")
    qdrant_distance: str = Field(description="当前 Qdrant 距离度量")


class RerankConfigResponse(BaseModel):
    enabled: bool = Field(description="rerank 是否启用")
    direct_upstream: bool = Field(description="是否直连上游 rerank")
    model: str = Field(description="当前 rerank 路由模型别名")
    upstream_model: str = Field(description="当前直连上游使用的模型名")
    upstream_base: str = Field(description="当前直连上游的基础地址")
    timeout_seconds: float = Field(description="rerank 超时秒数")
    top_n: Optional[int] = Field(default=None, description="rerank 返回的最大结果数")
    startup_strict: bool = Field(description="rerank 启动探针是否严格失败")


class CircuitBreakerStatusItem(BaseModel):
    component: str = Field(description="熔断器组件名")
    enabled: bool = Field(description="是否启用熔断")
    state: str = Field(description="当前状态")
    failure_count: int = Field(description="当前失败次数")
    failure_threshold: int = Field(description="触发熔断的失败阈值")
    open_seconds: float = Field(description="熔断开启持续时间")
    retry_after_seconds: float = Field(description="距离可重试的剩余秒数")
def _ops_error(code: str, err: str) -> dict:
    messages = {
        "QDRANT_INIT_ERROR": "Qdrant 初始化失败，请检查连接、鉴权与集合配置。",
        "QDRANT_RECREATE_ERROR": "Qdrant 重建失败，请检查集合状态与权限配置。",
        "QDRANT_HEALTH_ERROR": "Qdrant 健康检查失败，请检查服务可达性。",
        "QDRANT_COLLECTIONS_ERROR": "Qdrant 集合读取失败，请检查连接与权限。",
        "QDRANT_STATUS_ERROR": "Qdrant 状态读取失败，请检查连接与权限。",
    }
    return {
        "code": 500,
        "error_code": code,
        "error": err,
        "explanation": messages.get(code, "请求处理失败"),
    }


@router.post(
    "/qdrant/init",
    summary="Qdrant初始化",
    description="初始化集合，必要时可先删除旧集合再重建。",
)
def init_qdrant(
    collection: Optional[str] = Query(default=None, description="可选：指定要初始化的collection，默认使用QDRANT_COLLECTION"),
    recreate: bool = Query(default=False, description="是否先删除旧集合再重建"),
):
    init_qdrant_collection = resolve_runtime_dependency("init_qdrant_collection", None)

    try:
        if init_qdrant_collection is None:
            raise RuntimeError("qdrant init dependency unavailable")
        result = init_qdrant_collection(collection_name=collection, force_recreate=recreate)
        return {"code": 200, "data": result}
    except Exception as e:
        return _ops_error("QDRANT_INIT_ERROR", str(e)[:120])


@router.post(
    "/qdrant/recreate",
    summary="Qdrant重建",
    description="删除并重建指定集合，默认使用 QDRANT_COLLECTION。",
)
def recreate_qdrant(
    collection: Optional[str] = Query(default=None, description="可选：指定要重建的collection，默认使用QDRANT_COLLECTION"),
):
    recreate_qdrant_collection = resolve_runtime_dependency("recreate_qdrant_collection", None)

    target = (collection or get_qdrant_settings().collection).strip()
    try:
        if recreate_qdrant_collection is None:
            raise RuntimeError("qdrant recreate dependency unavailable")
        result = recreate_qdrant_collection(collection_name=target)
        return {"code": 200, "data": result}
    except Exception as e:
        return _ops_error("QDRANT_RECREATE_ERROR", str(e)[:120])


@router.get("/qdrant/health", summary="Qdrant健康检查", description="检查Qdrant连通性与可用性。")
def qdrant_health_check():
    qdrant_health = resolve_runtime_dependency("qdrant_health", None)

    try:
        if qdrant_health is None:
            raise RuntimeError("qdrant health dependency unavailable")
        result = qdrant_health()
        return {"code": 200, "data": result}
    except Exception as e:
        return _ops_error("QDRANT_HEALTH_ERROR", str(e)[:120])


@router.get("/qdrant/collections", summary="Qdrant集合列表", description="列出当前Qdrant中的集合。")
def qdrant_collections():
    qdrant_list_collections = resolve_runtime_dependency("qdrant_list_collections", None)

    try:
        if qdrant_list_collections is None:
            raise RuntimeError("qdrant collections dependency unavailable")
        result = qdrant_list_collections()
        return {"code": 200, "data": result}
    except Exception as e:
        return _ops_error("QDRANT_COLLECTIONS_ERROR", str(e)[:120])


@router.get("/qdrant/status", summary="Qdrant状态", description="获取Qdrant仓库的轻量状态信息。")
def qdrant_status():
    qdrant_repo_status = resolve_runtime_dependency("qdrant_repo_status", None)

    try:
        if qdrant_repo_status is None:
            raise RuntimeError("qdrant status dependency unavailable")
        result = qdrant_repo_status()
        return {"code": 200, "data": result}
    except Exception as e:
        return _ops_error("QDRANT_STATUS_ERROR", str(e)[:120])


@router.get(
    "/config/health",
    summary="配置健康摘要",
    description="返回启动期配置健康摘要，便于排障（不包含敏感值）。",
)
def config_health():
    summary = build_config_health_summary()
    return {
        "code": 200,
        "data": {
            "ok": bool(summary.get("ok", False)),
            "warnings": list(summary.get("warnings", ())),
            "errors": list(summary.get("errors", ())),
            "highlights": dict(summary.get("highlights", {})),
        },
    }


@router.get(
    "/embedding/config",
    summary="Embedding配置",
    description="返回当前生效的 embedding 模型、维度与维度来源。",
)
def embedding_config():
    cfg = resolve_embedding_config()
    qdrant_settings = get_qdrant_settings()
    return EmbeddingConfigResponse(
        model=cfg.model,
        dimensions=cfg.dimensions,
        dimension_source=cfg.dimension_source,
        collection=qdrant_settings.collection,
        qdrant_distance=qdrant_settings.distance,
    )


@router.get(
    "/rerank/config",
    summary="Rerank配置",
    description="返回当前 rerank 是否直连上游、路由模型与基础地址。",
)
def rerank_config():
    cfg = get_rerank_gateway_settings()
    return RerankConfigResponse(
        enabled=cfg.enabled,
        direct_upstream=cfg.direct_upstream,
        model=cfg.model,
        upstream_model=cfg.upstream_model,
        upstream_base=cfg.display_base_url,
        timeout_seconds=cfg.timeout_seconds,
        top_n=cfg.top_n,
        startup_strict=cfg.startup_strict,
    )


@router.get(
    "/gateway/resilience/status",
    summary="网关熔断状态",
    description="返回当前 LLM / Qdrant / 外部 API 熔断器的只读状态，便于排障。",
)
def gateway_resilience_status():
    return [CircuitBreakerStatusItem(**item) for item in get_resilience_snapshot()]


@router.get("/health", summary="服务健康检查", description="检查服务与LLM连通性，返回运行状态。")
async def health_check():
    master = resolve_runtime_dependency("master", None)
    is_prod_runtime = resolve_runtime_dependency("_is_prod_runtime", lambda: False)
    common_errors = resolve_runtime_dependency("_COMMON_ERROR_EXPLANATIONS", {})

    if master is None:
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": "master service unavailable",
            "error_code": "HEALTH_CHECK_ERROR",
            "explanation": common_errors.get("HEALTH_CHECK_ERROR", "健康检查失败"),
        }

    try:
        llm_ok = await asyncio.to_thread(master.check_llm_health)
        llm_status = "ok" if llm_ok else "error"
        health_info = {
            "status": "healthy",
            "timestamp": time.time(),
            "llm_status": llm_status,
            "env": "production" if is_prod_runtime() else "development",
        }
        logger.info(f"健康检查 | 状态: {health_info}")
        return health_info
    except Exception as e:
        logger.error(f"健康检查异常 | 错误: {str(e)[:100]}", exc_info=True)
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e)[:100],
            "error_code": "HEALTH_CHECK_ERROR",
            "explanation": common_errors.get("HEALTH_CHECK_ERROR", "健康检查失败"),
        }


@router.get("/health/live", summary="服务存活检查", description="轻量存活检查，不依赖LLM。")
def health_live():
    is_prod_runtime = resolve_runtime_dependency("_is_prod_runtime", lambda: False)

    return {
        "status": "healthy",
        "timestamp": time.time(),
        "env": "production" if is_prod_runtime() else "development",
    }


@router.get("/memory/status", summary="会话记忆状态", description="按 session_id 查询会话记忆状态。")
def memory_status(session_id: str):
    master = resolve_runtime_dependency("master", None)
    common_errors = resolve_runtime_dependency("_COMMON_ERROR_EXPLANATIONS", {})

    if not session_id.strip():
        raise HTTPException(
            status_code=400,
            detail={
                "message": "session_id不能为空",
                "error_code": "INVALID_SESSION_ID",
                "explanation": common_errors.get("INVALID_SESSION_ID", "session_id 参数非法"),
            },
        )

    if master is None:
        return {
            "code": 500,
            "error_code": "MEMORY_STATUS_ERROR",
            "session_id": session_id,
            "error": "master service unavailable",
            "explanation": common_errors.get("MEMORY_STATUS_ERROR", "会话记忆读取失败"),
        }

    try:
        status = master.get_memory_status(session_id)
        logger.info(
            "memory状态查询成功 | session_id: %s | count: %s",
            mask_session_id(status["session_id"]),
            status["message_count"],
        )
        return {"code": 200, "data": status}
    except Exception as e:
        err = str(e)[:120]
        logger.error(
            "memory状态查询失败 | session_id: %s | err: %s",
            mask_session_id(session_id),
            summarize_error_for_log(err),
            exc_info=True,
        )
        return {
            "code": 500,
            "error_code": "MEMORY_STATUS_ERROR",
            "session_id": session_id,
            "error": err,
            "explanation": common_errors.get("MEMORY_STATUS_ERROR", "会话记忆读取失败"),
        }
