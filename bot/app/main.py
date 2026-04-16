import os
import json
import sys
import time
import logging
import socket
import ipaddress
from enum import Enum
import requests
from urllib.parse import urlparse
from typing import Any, Dict, List, Literal, Optional
from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Qdrant
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

# 兼容 `python app/main.py` 直跑场景：把项目根目录加入模块搜索路径。
if __package__ in (None, ""):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from app.core.embedding_config import resolve_embedding_config
from app.core.config import get_rerank_gateway_settings
from app.core.logger_setup import logger, log_event
from app.core.litellm_adapters import build_litellm_embeddings_client
from app.services.master_service import Master
from app.tools.mytools import (
    astro_current_chart,
    astro_day_scope,
    astro_month_scope,
    astro_my_sign,
    astro_natal_chart,
    astro_transit_chart,
    astro_week_scope,
    search,
    test,
    vector_search,
    xingpan,
)
from app.services.qdrant_service import (
    init_qdrant_collection,
    qdrant_health,
    qdrant_list_collections,
    qdrant_repo_status,
    recreate_qdrant_collection,
)
from app.core.texts import USER_MESSAGES


def _is_prod_runtime() -> bool:
    return os.getenv("ENV", "dev").strip().lower() == "prod"

app = FastAPI(
    docs_url=None if _is_prod_runtime() else "/docs",
    redoc_url=None if _is_prod_runtime() else "/redoc",
    openapi_url=None if _is_prod_runtime() else "/openapi.json",
)
master = Master()

_PROD_ALLOWED_HTTP_PATHS = {"/chat"}

_COMMON_ERROR_EXPLANATIONS = {
    "ROUTE_NOT_EXPOSED": "生产环境仅暴露 /chat 接口，当前路由不可访问。",
    "DEBUG_TOOL_DISABLED": "调试接口仅在开发环境开放，请在 dev 环境调用。",
    "INVALID_SESSION_ID": "请提供非空 session_id，用于会话隔离与上下文追踪。",
    "CHAT_RUNTIME_ERROR": "对话处理失败，请稍后重试或检查模型与网络配置。",
    "QDRANT_INIT_ERROR": "Qdrant 初始化失败，请检查连接、鉴权与集合配置。",
    "QDRANT_RECREATE_ERROR": "Qdrant 重建失败，请检查集合状态与权限配置。",
    "QDRANT_HEALTH_ERROR": "Qdrant 健康检查失败，请检查服务可达性。",
    "QDRANT_COLLECTIONS_ERROR": "Qdrant 集合读取失败，请检查连接与权限。",
    "QDRANT_STATUS_ERROR": "Qdrant 状态读取失败，请检查连接与权限。",
    "MEMORY_STATUS_ERROR": "会话记忆读取失败，请检查 Redis/内存后端状态。",
    "HEALTH_CHECK_ERROR": "健康检查内部执行失败，请查看服务日志定位原因。",
    "TOOL_UNKNOWN_ERROR": "工具执行失败，请检查输入参数与依赖服务状态。",
    "REQUEST_VALIDATION_ERROR": "请求参数校验失败，请检查必填项、字段类型与取值范围。",
    "HTTP_400": "请求参数不合法，请检查输入内容。",
    "HTTP_401": "鉴权失败，请检查认证信息。",
    "HTTP_403": "当前请求无权限执行。",
    "HTTP_404": "目标资源不存在或当前环境不开放该接口。",
    "HTTP_409": "请求与当前资源状态冲突，请调整后重试。",
    "HTTP_422": "请求参数校验失败，请检查请求体结构与字段类型。",
    "HTTP_500": "服务内部错误，请稍后重试。",
    "INTERNAL_SERVER_ERROR": "服务内部发生未预期错误，请稍后重试。",
}


@app.middleware("http")
async def _restrict_routes_in_prod(request: Request, call_next):
    if _is_prod_runtime() and request.url.path not in _PROD_ALLOWED_HTTP_PATHS:
        return JSONResponse(
            status_code=404,
            content={
                "detail": "Not Found",
                "error_code": "ROUTE_NOT_EXPOSED",
                "explanation": _COMMON_ERROR_EXPLANATIONS["ROUTE_NOT_EXPOSED"],
            },
        )
    return await call_next(request)


def _resolve_explanation_by_error_code(error_code: str) -> str:
    code = (error_code or "").strip()
    if code in _COMMON_ERROR_EXPLANATIONS:
        return _COMMON_ERROR_EXPLANATIONS[code]
    return "请求处理失败，请查看错误信息并联系管理员。"


def _normalize_http_exception_payload(status_code: int, detail: Any) -> Dict[str, Any]:
    if isinstance(detail, dict):
        payload = dict(detail)
        error_code = payload.get("error_code") or f"HTTP_{status_code}"
        explanation = payload.get("explanation") or _resolve_explanation_by_error_code(str(error_code))
        if "message" not in payload and "detail" not in payload:
            payload["message"] = "请求处理失败"
        return {
            "detail": payload,
            "error_code": str(error_code),
            "explanation": explanation,
        }

    if isinstance(detail, list):
        return {
            "detail": "请求参数校验失败",
            "error_code": "REQUEST_VALIDATION_ERROR",
            "explanation": _resolve_explanation_by_error_code("REQUEST_VALIDATION_ERROR"),
            "errors": detail,
        }

    text = str(detail or "请求处理失败")
    code = f"HTTP_{status_code}"
    return {
        "detail": text,
        "error_code": code,
        "explanation": _resolve_explanation_by_error_code(code),
    }


@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException):
    payload = _normalize_http_exception_payload(exc.status_code, exc.detail)
    return JSONResponse(status_code=exc.status_code, content=payload)


@app.exception_handler(RequestValidationError)
async def _request_validation_exception_handler(request: Request, exc: RequestValidationError):
    payload = {
        "detail": "请求参数校验失败",
        "error_code": "REQUEST_VALIDATION_ERROR",
        "explanation": _resolve_explanation_by_error_code("REQUEST_VALIDATION_ERROR"),
        "errors": exc.errors(),
    }
    return JSONResponse(status_code=422, content=payload)


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("未捕获异常 | path: %s | err: %s", request.url.path, str(exc)[:200], exc_info=True)
    payload = {
        "detail": "Internal Server Error",
        "error_code": "INTERNAL_SERVER_ERROR",
        "explanation": _resolve_explanation_by_error_code("INTERNAL_SERVER_ERROR"),
    }
    return JSONResponse(status_code=500, content=payload)


class AddUrlsErrorCode(str, Enum):
    FETCH_ERROR = "FETCH_ERROR"
    BLOCKED_URL = "BLOCKED_URL"
    INVALID_URL = "INVALID_URL"
    UNSUPPORTED_SCHEME = "UNSUPPORTED_SCHEME"
    MISSING_HOST = "MISSING_HOST"
    INVALID_IP = "INVALID_IP"
    BLOCKED_PRIVATE_IP = "BLOCKED_PRIVATE_IP"
    BLOCKED_LOOPBACK = "BLOCKED_LOOPBACK"
    BLOCKED_LINK_LOCAL = "BLOCKED_LINK_LOCAL"
    BLOCKED_INTERNAL_HOST = "BLOCKED_INTERNAL_HOST"


_ADD_URLS_ERROR_EXPLANATIONS = {
    AddUrlsErrorCode.FETCH_ERROR.value: "目标地址访问失败或内容解析失败，请检查URL可访问性、SSL配置和页面结构。",
    AddUrlsErrorCode.BLOCKED_URL.value: "URL被安全策略拦截，请更换为可公开访问的HTTP/HTTPS地址。",
    AddUrlsErrorCode.INVALID_URL.value: "URL格式无法解析，请检查是否为完整地址。",
    AddUrlsErrorCode.UNSUPPORTED_SCHEME.value: "仅支持HTTP/HTTPS协议，请调整URL协议。",
    AddUrlsErrorCode.MISSING_HOST.value: "URL缺少主机名，请提供包含域名或IP的完整地址。",
    AddUrlsErrorCode.INVALID_IP.value: "IP地址格式无效，请检查IP是否正确。",
    AddUrlsErrorCode.BLOCKED_PRIVATE_IP.value: "检测到私网地址，为避免内网探测风险已拒绝访问。",
    AddUrlsErrorCode.BLOCKED_LOOPBACK.value: "检测到环回地址，为避免访问本机服务已拒绝。",
    AddUrlsErrorCode.BLOCKED_LINK_LOCAL.value: "检测到链路本地地址，为避免访问局域网络设备已拒绝。",
    AddUrlsErrorCode.BLOCKED_INTERNAL_HOST.value: "检测到疑似内网主机名（无公网域名特征），已按策略拦截。",
}


class AddUrlsRequest(BaseModel):
    """/add_urls 与 /add_urls/dry_run 请求体。

    语义约定:
    - 缺失: 使用字段默认值。
    - None: 视为未提供（如 url=None, separators=None）。
    - 空值: url="" 或 urls 中空白字符串会在归一化阶段被忽略。
    """

    url: Optional[str] = Field(default=None, description="单个URL；缺失/None/空字符串视为未提供")
    urls: List[str] = Field(default_factory=list, description="批量URL列表；缺失=空列表，列表中的空白URL会被忽略")
    chunk_strategy: Literal["balanced", "faq", "article", "custom"] = Field(
        default="balanced",
        description="切块策略：balanced|faq|article|custom",
    )
    chunk_size: Optional[int] = Field(default=None, ge=100, le=4000, description="可选；缺失/None时沿用策略默认值")
    chunk_overlap: Optional[int] = Field(default=None, ge=0, le=1000, description="可选；缺失/None时沿用策略默认值")
    separators: Optional[List[str]] = Field(default=None, description="仅custom策略建议传入；缺失/None/空列表时沿用策略默认分隔符")
    preview_limit: int = Field(default=3, ge=1, le=20, description="dry_run时返回示例chunk数量")


class FailedUrlItem(BaseModel):
    """单个失败URL信息。error 仅截取前120字符，避免日志/响应过长。"""

    url: str = Field(description="失败URL")
    code: AddUrlsErrorCode = Field(default=AddUrlsErrorCode.FETCH_ERROR, description="失败类型编码")
    error: str = Field(description="失败原因")
    explanation: str = Field(default="", description="错误解释与处理建议")


class ChunkConfigModel(BaseModel):
    """本次请求实际生效的切块配置，不返回 None。"""

    chunk_size: int = Field(description="实际chunk_size")
    chunk_overlap: int = Field(description="实际chunk_overlap")
    separators: List[str] = Field(description="实际分隔符")


class ChunkPreviewItem(BaseModel):
    """dry_run 预览项。content_preview 是截断文本，不保证完整。"""

    source_url: str = Field(description="来源URL")
    chunk_index: int = Field(description="chunk序号")
    content_length: int = Field(description="chunk长度")
    content_preview: str = Field(description="chunk预览")


class AddUrlsResponse(BaseModel):
    """/add_urls 响应体。

    语义约定:
    - failed_urls 永远是数组（可能为空），不会为 None。
    - chunk_config 永远为完整对象，不返回 None。
    """

    response: str
    collection: str
    mode: Literal["remote", "local"]
    source_urls: int
    chunks: int
    failed_urls: List[FailedUrlItem]
    chunk_strategy: Literal["balanced", "faq", "article", "custom"]
    chunk_config: ChunkConfigModel
    quality_report: Dict[str, Any]


class AddUrlsDryRunResponse(BaseModel):
    """/add_urls/dry_run 响应体。

    语义约定:
    - failed_urls 与 chunk_preview 永远是数组（可能为空），不会为 None。
    - chunk_config 永远为完整对象，不返回 None。
    """

    response: str
    source_urls: int
    chunks: int
    failed_urls: List[FailedUrlItem]
    chunk_strategy: Literal["balanced", "faq", "article", "custom"]
    chunk_config: ChunkConfigModel
    chunk_preview: List[ChunkPreviewItem]
    quality_report: Dict[str, Any]


class EmbeddingConfigResponse(BaseModel):
    """当前生效的 Embedding 配置，便于线上排障。"""

    model: str = Field(description="当前生效的 embedding 模型别名")
    dimensions: int = Field(description="当前生效的向量维度")
    dimension_source: Literal["env", "model_hint", "fallback", "default"] = Field(
        description="维度来源：env|model_hint|fallback|default"
    )
    collection: str = Field(description="当前 Qdrant 目标集合")
    qdrant_distance: str = Field(description="当前 Qdrant 距离度量")


class RerankConfigResponse(BaseModel):
    """当前生效的 rerank 配置，便于线上排障。"""

    enabled: bool = Field(description="rerank 是否启用")
    direct_upstream: bool = Field(description="是否直连上游 rerank，而不是走 LiteLLM")
    model: str = Field(description="当前 rerank 路由模型别名")
    upstream_model: str = Field(description="当前直连上游使用的模型名")
    upstream_base: str = Field(description="当前直连上游的基础地址")
    timeout_seconds: float = Field(description="rerank 超时秒数")
    top_n: Optional[int] = Field(default=None, description="rerank 返回的最大结果数，未设置则为 null")
    startup_strict: bool = Field(description="rerank 启动探针是否严格失败")


class ToolTestRequest(BaseModel):
    scope: Optional[str] = Field(default="all", description="all|astro|vector|search")


class ToolSearchRequest(BaseModel):
    query: str = Field(description="搜索关键词")


class ToolVectorSearchRequest(BaseModel):
    query: str = Field(description="向量检索关键词")


class ToolXingpanRequest(BaseModel):
    name: str = Field(description="姓名")
    birth_dt: str = Field(description="出生时间 YYYY-MM-DD HH:MM:SS")
    longitude: float = Field(description="经度")
    latitude: float = Field(description="纬度")


class ToolAstroChartRequest(BaseModel):
    birth_dt: str = Field(description="出生时间 YYYY-MM-DD HH:MM:SS")
    longitude: float = Field(description="经度")
    latitude: float = Field(description="纬度")


def _ensure_debug_tools_enabled():
    if _is_prod_runtime():
        raise HTTPException(
            status_code=404,
            detail={
                "message": "Not Found",
                "error_code": "DEBUG_TOOL_DISABLED",
                "explanation": _COMMON_ERROR_EXPLANATIONS["DEBUG_TOOL_DISABLED"],
            },
        )


def _tool_error_explanation(code: str, error: str) -> str:
    if not error:
        return ""
    if code == "CONFIG_MISSING":
        return "工具配置缺失，请检查相关环境变量。"
    if code == "TIMEOUT":
        return "工具调用超时，请稍后重试或上调超时参数。"
    if code.startswith("HTTP_"):
        return "上游服务返回HTTP错误，请检查上游可用性和鉴权。"
    return _COMMON_ERROR_EXPLANATIONS["TOOL_UNKNOWN_ERROR"]


def _wrap_tool_result(tool_name: str, raw_result: Any) -> Dict[str, Any]:
    parsed = None
    if isinstance(raw_result, str):
        try:
            parsed = json.loads(raw_result)
        except Exception:
            parsed = None

    if isinstance(parsed, dict) and {"ok", "code", "data", "error"}.issubset(parsed.keys()):
        wrapped = {"tool": tool_name, **parsed}
        wrapped["explanation"] = _tool_error_explanation(str(wrapped.get("code", "")), str(wrapped.get("error", "")))
        return wrapped

    wrapped = {
        "tool": tool_name,
        "ok": True,
        "code": "OK",
        "data": parsed if parsed is not None else raw_result,
        "error": "",
    }
    wrapped["explanation"] = ""
    return wrapped


def _resolve_add_urls_payload(
    payload: Optional[AddUrlsRequest] = Body(default=None),
    url: Optional[str] = Query(default=None),
    urls: Optional[List[str]] = Query(default=None),
    chunk_strategy: Optional[Literal["balanced", "faq", "article", "custom"]] = Query(default=None),
    chunk_size: Optional[int] = Query(default=None, ge=100, le=4000),
    chunk_overlap: Optional[int] = Query(default=None, ge=0, le=1000),
    separators: Optional[List[str]] = Query(default=None),
    preview_limit: Optional[int] = Query(default=None, ge=1, le=20),
) -> AddUrlsRequest:
    """统一接收JSON与Query参数，归一成 AddUrlsRequest。"""
    data = payload.model_dump() if payload is not None else {}

    existing_url = data.get("url")
    if url is not None and (existing_url is None or (isinstance(existing_url, str) and not existing_url.strip())):
        data["url"] = url
    if urls is not None and ("urls" not in data or not data.get("urls")):
        data["urls"] = urls
    if chunk_strategy is not None and data.get("chunk_strategy") is None:
        data["chunk_strategy"] = chunk_strategy
    if chunk_size is not None and data.get("chunk_size") is None:
        data["chunk_size"] = chunk_size
    if chunk_overlap is not None and data.get("chunk_overlap") is None:
        data["chunk_overlap"] = chunk_overlap
    if separators is not None and data.get("separators") is None:
        data["separators"] = separators
    if preview_limit is not None and data.get("preview_limit") is None:
        data["preview_limit"] = preview_limit

    return AddUrlsRequest(**data)


def _build_chunking_config(
    strategy: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    separators: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """根据策略构造切块配置，支持按需覆盖默认参数。"""
    strategy_defaults = {
        # 通用知识库：兼顾语义完整性与检索粒度
        "balanced": {
            "chunk_size": 900,
            "chunk_overlap": 120,
            "separators": ["\n\n", "\n", "。", "！", "？", "；", ". ", "! ", "? ", "，", ",", " ", ""],
        },
        # FAQ/问答：更细粒度，提升精确召回
        "faq": {
            "chunk_size": 480,
            "chunk_overlap": 80,
            "separators": ["\n\n", "\n", "。", "？", "！", "；", ". ", "? ", "! ", "，", ",", " ", ""],
        },
        # 长文/教程：更大块，保留上下文
        "article": {
            "chunk_size": 1400,
            "chunk_overlap": 180,
            "separators": ["\n\n", "\n", "###", "##", "#", "。", "；", ". ", ",", " ", ""],
        },
        # 自定义：允许调用方完全控制
        "custom": {
            "chunk_size": 800,
            "chunk_overlap": 100,
            "separators": ["\n\n", "\n", "。", ". ", " ", ""],
        },
    }

    base = strategy_defaults.get(strategy, strategy_defaults["balanced"]).copy()
    if chunk_size is not None:
        base["chunk_size"] = chunk_size
    if chunk_overlap is not None:
        base["chunk_overlap"] = chunk_overlap
    if separators:
        clean_separators = [s for s in separators if isinstance(s, str)]
        if clean_separators:
            base["separators"] = clean_separators

    # 防止无效参数导致切分器异常
    if base["chunk_overlap"] >= base["chunk_size"]:
        base["chunk_overlap"] = max(0, base["chunk_size"] // 5)
    return base


def _chunk_documents(documents: List[Any], source_url: str, cfg: Dict[str, Any], strategy: str) -> List[Any]:
    """执行切块并补充metadata，便于后续检索和排障。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        separators=cfg["separators"],
    )
    chunks = splitter.split_documents(documents)
    total = len(chunks)
    for idx, chunk in enumerate(chunks):
        chunk.metadata = {
            **(chunk.metadata or {}),
            "source_url": source_url,
            "chunk_strategy": strategy,
            "chunk_index": idx,
            "chunk_total": total,
        }
    return chunks


def _normalize_urls(payload: AddUrlsRequest) -> List[str]:
    """统一收敛 url/urls 参数，便于复用。"""
    clean_url_list = []
    if payload.url and payload.url.strip():
        clean_url_list.append(payload.url.strip())
    if payload.urls:
        clean_url_list.extend([u.strip() for u in payload.urls if isinstance(u, str) and u.strip()])
    return clean_url_list


def _is_truthy_env(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _get_add_urls_error_explanation(code: Optional[str]) -> str:
    clean_code = (code or "").strip()
    if clean_code in _ADD_URLS_ERROR_EXPLANATIONS:
        return _ADD_URLS_ERROR_EXPLANATIONS[clean_code]
    return "未知错误类型，请查看 error 字段并联系管理员排查。"


def _normalize_failed_urls(items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """统一补全 failed_urls 的 code/explanation，避免调用方分支判断复杂化。"""
    normalized: List[Dict[str, str]] = []
    for item in items or []:
        url = str((item or {}).get("url", "") or "")
        code = str((item or {}).get("code", AddUrlsErrorCode.FETCH_ERROR.value) or AddUrlsErrorCode.FETCH_ERROR.value)
        error = str((item or {}).get("error", "") or "")
        explanation = str((item or {}).get("explanation", "") or "")
        normalized.append(
            {
                "url": url,
                "code": code,
                "error": error,
                "explanation": explanation or _get_add_urls_error_explanation(code),
            }
        )
    return normalized


def _is_public_http_url(raw_url: str) -> tuple[bool, Optional[AddUrlsErrorCode], str]:
    """阻断私网/环回/链路本地地址，降低 SSRF 风险。"""
    try:
        parsed = urlparse(raw_url)
    except Exception:
        return False, AddUrlsErrorCode.INVALID_URL, "URL解析失败"

    scheme = (parsed.scheme or "").lower()
    if scheme not in {"http", "https"}:
        return False, AddUrlsErrorCode.UNSUPPORTED_SCHEME, "仅支持 http/https URL"

    hostname = (parsed.hostname or "").strip()
    if not hostname:
        return False, AddUrlsErrorCode.MISSING_HOST, "URL缺少主机名"

    normalized_host = hostname.lower().rstrip(".")
    if normalized_host in {"localhost", "localhost.localdomain"}:
        return False, AddUrlsErrorCode.BLOCKED_LOOPBACK, "禁止访问环回地址"

    ip_candidates = []
    host_is_ip_literal = False
    try:
        ip_candidates.append(ipaddress.ip_address(hostname))
        host_is_ip_literal = True
    except ValueError:
        if "." not in normalized_host:
            return False, AddUrlsErrorCode.BLOCKED_INTERNAL_HOST, "禁止访问疑似内网主机名"
        try:
            infos = socket.getaddrinfo(hostname, None)
            for info in infos:
                sockaddr = info[4]
                if not sockaddr:
                    continue
                candidate_ip = sockaddr[0]
                try:
                    ip_candidates.append(ipaddress.ip_address(candidate_ip))
                except ValueError:
                    continue
        except socket.gaierror:
            # DNS 暂不可用时，不把公网域名误判为私网；后续抓取流程再决定是否可访问。
            return True, None, ""
        except Exception:
            return True, None, ""

    if not ip_candidates and host_is_ip_literal:
        return False, AddUrlsErrorCode.INVALID_IP, "IP地址无效"

    for ip_obj in ip_candidates:
        if ip_obj.is_loopback:
            return False, AddUrlsErrorCode.BLOCKED_LOOPBACK, "禁止访问环回地址"
        if ip_obj.is_link_local:
            return False, AddUrlsErrorCode.BLOCKED_LINK_LOCAL, "禁止访问链路本地地址"
        if ip_obj.is_private:
            return False, AddUrlsErrorCode.BLOCKED_PRIVATE_IP, "禁止访问私网地址"

    return True, None, ""


def _partition_safe_urls(clean_url_list: List[str]) -> tuple[List[str], List[Dict[str, str]]]:
    """把URL分为可访问与阻断列表。"""
    allowed_urls: List[str] = []
    blocked_urls: List[Dict[str, str]] = []
    for one_url in clean_url_list:
        ok, code, reason = _is_public_http_url(one_url)
        if ok:
            allowed_urls.append(one_url)
        else:
            blocked_urls.append(
                {
                    "url": one_url,
                    "code": (code or AddUrlsErrorCode.BLOCKED_URL).value,
                    "error": reason[:120],
                    "explanation": _get_add_urls_error_explanation((code or AddUrlsErrorCode.BLOCKED_URL).value),
                }
            )
    return allowed_urls, blocked_urls


def _ensure_add_urls_write_enabled() -> None:
    """生产环境默认关闭入库，需显式开关开启。"""
    is_prod_runtime = _is_prod_runtime()
    write_enabled = _is_truthy_env(os.getenv("ADD_URLS_WRITE_ENABLED", "false"))
    if is_prod_runtime and not write_enabled:
        raise HTTPException(
            status_code=403,
            detail={
                "message": "生产环境默认禁用 /add_urls 入库，请显式开启开关",
                "required_env": "ADD_URLS_WRITE_ENABLED=true",
            },
        )


def _collect_chunks_from_urls(clean_url_list: List[str], chunk_cfg: Dict[str, Any], strategy: str):
    """按URL抓取并切块，返回成功chunk与失败URL详情。"""
    all_chunks = []
    failed_urls = []
    verify_ssl = os.getenv("WEB_LOADER_VERIFY_SSL", "true").strip().lower() not in {"0", "false", "no", "off"}
    if not verify_ssl:
        logger.warning("WebBaseLoader SSL校验已关闭（WEB_LOADER_VERIFY_SSL=false），仅建议临时排障使用")

    for one_url in clean_url_list:
        try:
            web_loader = WebBaseLoader(one_url, verify_ssl=verify_ssl)
            documents = web_loader.load()
            all_chunks.extend(_chunk_documents(documents, one_url, chunk_cfg, strategy))
        except Exception as e:
            failed_urls.append(
                {
                    "url": one_url,
                    "code": AddUrlsErrorCode.FETCH_ERROR.value,
                    "error": str(e)[:400],
                    "explanation": _get_add_urls_error_explanation(AddUrlsErrorCode.FETCH_ERROR.value),
                }
            )
    return all_chunks, failed_urls


def _compute_chunk_quality_report(chunks: List[Any], failed_urls: List[dict], min_len: int = 120) -> Dict[str, Any]:
    """基于规则的切块质量报告，输出简洁可读的评分与指标。"""
    total = len(chunks)
    if total == 0:
        return {
            "score": 0,
            "label": "poor",
            "stats": {
                "chunks": 0,
                "failed_urls": len(failed_urls),
            },
            "signals": ["no_chunks"],
        }

    lengths = []
    empty_chunks = 0
    short_chunks = 0
    bad_end_chunks = 0
    symbols = {",", "，", ".", "。", "!", "！", "?", "？", ";", "；", ":", "：", "…"}

    for chunk in chunks:
        content = (getattr(chunk, "page_content", "") or "").strip()
        if not content:
            empty_chunks += 1
            continue
        length = len(content)
        lengths.append(length)
        if length < min_len:
            short_chunks += 1
        if content[-1] not in symbols:
            bad_end_chunks += 1

    chunks_count = len(chunks)
    empty_ratio = empty_chunks / chunks_count
    short_ratio = short_chunks / chunks_count
    bad_end_ratio = bad_end_chunks / chunks_count

    avg_len = int(sum(lengths) / len(lengths)) if lengths else 0
    min_len_val = min(lengths) if lengths else 0
    max_len_val = max(lengths) if lengths else 0

    score = 100
    score -= int(empty_ratio * 100) * 3
    score -= int(short_ratio * 100) * 2
    score -= int(bad_end_ratio * 100) * 2
    score = max(0, min(100, score))

    if score >= 85:
        label = "good"
    elif score >= 70:
        label = "fair"
    else:
        label = "poor"

    signals = []
    if empty_ratio > 0.01:
        signals.append("empty_chunks")
    if short_ratio > 0.15:
        signals.append("too_many_short")
    if bad_end_ratio > 0.3:
        signals.append("mid_sentence_cut")
    if failed_urls:
        signals.append("fetch_failed")

    suggestions = []
    if empty_ratio > 0.01:
        suggestions.append("检查抓取来源，过滤无正文页面或广告脚本")
    if short_ratio > 0.15:
        suggestions.append("尝试提高 chunk_size 或降低 chunk_overlap")
    if bad_end_ratio > 0.3:
        suggestions.append("断句偏多，调整 separators，使切分优先落在句号/问号/换行")
    if 0.2 < bad_end_ratio <= 0.3:
        suggestions.append(
            f"bad_end_ratio={bad_end_ratio:.3f}；可将 chunk_size 上调 15%-25%，并把强分隔符(\\n\\n, \\n, 。, ！, ？)放在 separators 前部"
        )
    if min_len_val and min_len_val < 60 and short_ratio > 0.01:
        suggestions.append(f"min_len={min_len_val}；建议过滤 <80 字碎片，或适度上调 chunk_size")
    if avg_len and avg_len < 300:
        suggestions.append(f"avg_len={avg_len}；可增加 chunk_size 或做正文抽取(去导航/页脚)")
    if failed_urls:
        suggestions.append("检查失败 URL 的可访问性与 SSL 配置")
    if score < 70 and not suggestions:
        suggestions.append("整体质量偏低，建议微调 chunk_size 与 separators 再评估")

    return {
        "score": score,
        "label": label,
        "stats": {
            "chunks": chunks_count,
            "failed_urls": len(failed_urls),
            "avg_len": avg_len,
            "min_len": min_len_val,
            "max_len": max_len_val,
            "empty_ratio": round(empty_ratio, 3),
            "short_ratio": round(short_ratio, 3),
            "bad_end_ratio": round(bad_end_ratio, 3),
        },
        "signals": signals,
        "suggestions": suggestions,
    }


def _resolve_embedding_output_dim(embeddings_client: Any, default_size: int) -> int:
    """尽量解析Embedding实际输出维度，避免Qdrant维度错配。"""
    dimensions = getattr(embeddings_client, "dimensions", None)
    if isinstance(dimensions, int) and dimensions > 0:
        return dimensions

    try:
        probe_vector = embeddings_client.embed_query("dimension_probe")
        if isinstance(probe_vector, (list, tuple)) and len(probe_vector) > 0:
            return len(probe_vector)
    except Exception as e:
        logger.warning("Embedding维度探测失败，使用默认值: %s | err: %s", default_size, str(e)[:120])

    return default_size


def _extract_collection_vector_size(collection_info: Any) -> Optional[int]:
    """从Qdrant collection信息中提取向量维度。"""
    config = getattr(collection_info, "config", None)
    params = getattr(config, "params", None)
    vectors = getattr(params, "vectors", None)
    if vectors is None:
        return None

    direct_size = getattr(vectors, "size", None)
    if isinstance(direct_size, int) and direct_size > 0:
        return direct_size

    if isinstance(vectors, dict):
        for vector_cfg in vectors.values():
            size = getattr(vector_cfg, "size", None)
            if isinstance(size, int) and size > 0:
                return size

    return None


def _fetch_collection_vector_size_via_http(collection_name: str) -> Optional[int]:
    qdrant_url = os.getenv("QDRANT_URL", "").strip()
    if not qdrant_url:
        return None

    api_key = os.getenv("QDRANT_API_KEY", "").strip()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["api-key"] = api_key

    url = f"{qdrant_url.rstrip('/')}/collections/{collection_name}"
    try:
        resp = requests.get(url, headers=headers, timeout=6)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        logger.warning("Qdrant HTTP元数据读取失败 | collection: %s | err: %s", collection_name, str(e)[:160])
        return None

    result = payload.get("result", payload)
    config = result.get("config") if isinstance(result, dict) else None
    params = config.get("params") if isinstance(config, dict) else None
    vectors = params.get("vectors") if isinstance(params, dict) else None
    if vectors is None:
        return None

    if isinstance(vectors, dict):
        size = vectors.get("size")
        if isinstance(size, int) and size > 0:
            return size
        for vector_cfg in vectors.values():
            if isinstance(vector_cfg, dict):
                vsize = vector_cfg.get("size")
                if isinstance(vsize, int) and vsize > 0:
                    return vsize
    return None


class VectorSizeMismatchError(RuntimeError):
    def __init__(self, collection_name: str, existing_size: int, expected_size: int) -> None:
        message = (
            "Qdrant collection vector size mismatch: "
            f"collection={collection_name} existing={existing_size} expected={expected_size}"
        )
        super().__init__(message)
        self.collection_name = collection_name
        self.existing_size = existing_size
        self.expected_size = expected_size


def _recreate_collection_with_dim(client: QdrantClient, collection_name: str, vector_size: int, old_dim: Optional[int] = None) -> None:
    """删除并重建集合，确保向量维度与当前Embedding一致。"""
    distance = _resolve_qdrant_distance()
    try:
        client.delete_collection(collection_name=collection_name)
    except TypeError:
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=rest.VectorParams(size=vector_size, distance=distance),
    )
    logger.warning(
        "Qdrant集合已重建 | collection: %s | old_dim: %s | new_dim: %s",
        collection_name,
        old_dim,
        vector_size,
    )


def _is_vector_dim_mismatch_error(error: Exception) -> bool:
    msg = str(error)
    return "Vector dimension error" in msg or "expected dim" in msg and "got" in msg


def _resolve_qdrant_distance() -> Any:
    """解析Qdrant距离度量，默认使用Cosine。"""
    distance_name = os.getenv("QDRANT_DISTANCE", "Cosine").strip().lower()
    distance_map = {
        "cosine": rest.Distance.COSINE,
        "dot": rest.Distance.DOT,
        "euclid": rest.Distance.EUCLID,
    }
    return distance_map.get(distance_name, rest.Distance.COSINE)


def _ensure_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int) -> Optional[int]:
    """确保目标collection存在且维度匹配。"""
    try:
        exists = client.collection_exists(collection_name)
    except Exception:
        # 兼容旧版本/异常场景：退化为列举集合判断。
        info = client.get_collections()
        collections = [c.name for c in getattr(info, "collections", []) or []]
        exists = collection_name in collections

    distance = _resolve_qdrant_distance()

    if exists:
        existing_size: Optional[int] = None
        try:
            info = client.get_collection(collection_name=collection_name)
            existing_size = _extract_collection_vector_size(info)
        except TypeError:
            info = client.get_collection(collection_name)
            existing_size = _extract_collection_vector_size(info)
        except Exception as e:
            logger.warning(
                "读取Qdrant集合元数据失败，尝试HTTP兜底 | collection: %s | err: %s",
                collection_name,
                str(e)[:160],
            )
            existing_size = _fetch_collection_vector_size_via_http(collection_name)
            if existing_size is None:
                logger.warning(
                    "HTTP兜底仍失败，跳过预检并在写入阶段兜底 | collection: %s",
                    collection_name,
                )
                return

        if existing_size and existing_size != vector_size:
            raise VectorSizeMismatchError(
                collection_name=collection_name,
                existing_size=existing_size,
                expected_size=vector_size,
            )
        return existing_size

    client.create_collection(
        collection_name=collection_name,
        vectors_config=rest.VectorParams(size=vector_size, distance=distance),
    )
    logger.info(
        "Qdrant collection自动创建成功 | collection: %s | vector_size: %s | distance: %s",
        collection_name,
        vector_size,
        distance.name,
    )
    return vector_size


def _build_embeddings_client(vector_size: int):
    """统一走 LiteLLM embeddings 模型（默认 bge-m3）。"""
    embedding_cfg = resolve_embedding_config(default_dimension=vector_size)
    return build_litellm_embeddings_client(default_dimensions=embedding_cfg.dimensions)

@app.get("/", summary="根路径", description="基础连通性检查，返回简单响应。")
def read_root():
    """根路径响应，用于快速确认服务存活。"""
    logger.info("访问根路径")
    return {"Hello": "World"}

@app.post("/chat", summary="对话接口", description="主对话入口。query 为用户输入，session_id 用于会话隔离。")
def chat(query: str, session_id: str):
    """处理一次对话请求并返回模型回复。"""
    logger.info(f"接收Chat API请求 | session_id: {session_id} | 查询: {query[:100]}")
    if not session_id.strip():
        raise HTTPException(
            status_code=400,
            detail={
                "message": "session_id不能为空",
                "error_code": "INVALID_SESSION_ID",
                "explanation": _COMMON_ERROR_EXPLANATIONS["INVALID_SESSION_ID"],
            },
        )
    try:
        res = master.run(query, session_id=session_id)
        response_text = res.get("output", "")
        logger.info(f"Chat API响应成功 | session_id: {session_id} | 查询: {query[:50]} | 响应长度: {len(response_text)}")
        return {"code": 200, "session_id": session_id, "query": query, "response": response_text}
    except Exception as e:
        error_msg = str(e)[:100]
        logger.error(f"Chat API执行异常 | 查询: {query[:50]} | 错误: {error_msg}", exc_info=True)
        return {
            "code": 500,
            "error_code": "CHAT_RUNTIME_ERROR",
            "explanation": _COMMON_ERROR_EXPLANATIONS["CHAT_RUNTIME_ERROR"],
            "error": error_msg,
            "session_id": session_id,
            "query": query,
            "response": f"错误：{error_msg}",
        }


@app.post("/tools/test", summary="工具调试：系统自检", description="调用 test 工具，支持 scope=all|astro|vector|search 用于快速排查配置。")
def debug_tool_test(payload: ToolTestRequest = Body(...)):
    _ensure_debug_tools_enabled()
    result = test.invoke({"scope": payload.scope or "all"})
    return _wrap_tool_result("test", result)


@app.post("/tools/search", summary="工具调试：联网搜索", description="调用 search 工具，使用 SERPAPI 查询外部信息。")
def debug_tool_search(payload: ToolSearchRequest = Body(...)):
    _ensure_debug_tools_enabled()
    result = search.invoke({"query": payload.query})
    return _wrap_tool_result("search", result)


@app.post("/tools/vector_search", summary="工具调试：向量检索", description="调用 vector_search 工具，从已入库文档中检索相似内容。")
def debug_tool_vector_search(payload: ToolVectorSearchRequest = Body(...)):
    _ensure_debug_tools_enabled()
    result = vector_search.invoke({"query": payload.query})
    return _wrap_tool_result("vector_search", result)


@app.post("/tools/xingpan", summary="工具调试：星盘", description="调用 xingpan 工具，依据姓名、出生时间、经纬度查询星盘。")
def debug_tool_xingpan(payload: ToolXingpanRequest = Body(...)):
    _ensure_debug_tools_enabled()
    result = xingpan.invoke(
        {
            "name": payload.name,
            "birth_dt": payload.birth_dt,
            "longitude": payload.longitude,
            "latitude": payload.latitude,
        }
    )
    return _wrap_tool_result("xingpan", result)


@app.post("/tools/astro/my_sign", summary="工具调试：星座信息", description="调用 astro_my_sign 工具，读取 ASTRO_UID 并查询星座信息。")
def debug_tool_astro_my_sign(payload: ToolAstroChartRequest = Body(...)):
    _ensure_debug_tools_enabled()
    result = astro_my_sign.invoke(
        {
            "birth_dt": payload.birth_dt,
            "longitude": payload.longitude,
            "latitude": payload.latitude,
        }
    )
    return _wrap_tool_result("astro_my_sign", result)


@app.post("/tools/astro/day", summary="工具调试：日运势", description="调用 astro_day_scope 工具，读取 ASTRO_UID 并查询日运势。")
def debug_tool_astro_day():
    _ensure_debug_tools_enabled()
    result = astro_day_scope.invoke({})
    return _wrap_tool_result("astro_day_scope", result)


@app.post("/tools/astro/week", summary="工具调试：周运势", description="调用 astro_week_scope 工具，读取 ASTRO_UID 并查询周运势。")
def debug_tool_astro_week():
    _ensure_debug_tools_enabled()
    result = astro_week_scope.invoke({})
    return _wrap_tool_result("astro_week_scope", result)


@app.post("/tools/astro/month", summary="工具调试：月运势", description="调用 astro_month_scope 工具，读取 ASTRO_UID 并查询月运势。")
def debug_tool_astro_month():
    _ensure_debug_tools_enabled()
    result = astro_month_scope.invoke({})
    return _wrap_tool_result("astro_month_scope", result)


@app.post("/tools/astro/natal_chart", summary="工具调试：本命盘", description="调用 astro_natal_chart 工具，依据出生时间与经纬度查询本命盘。")
def debug_tool_astro_natal(payload: ToolAstroChartRequest = Body(...)):
    _ensure_debug_tools_enabled()
    result = astro_natal_chart.invoke(
        {
            "birth_dt": payload.birth_dt,
            "longitude": payload.longitude,
            "latitude": payload.latitude,
        }
    )
    return _wrap_tool_result("astro_natal_chart", result)


@app.post("/tools/astro/current_chart", summary="工具调试：当前天象盘", description="调用 astro_current_chart 工具，查询当前天象盘。")
def debug_tool_astro_current():
    _ensure_debug_tools_enabled()
    result = astro_current_chart.invoke({})
    return _wrap_tool_result("astro_current_chart", result)


@app.post("/tools/astro/transit_chart", summary="工具调试：行运盘", description="调用 astro_transit_chart 工具，依据出生时间与经纬度查询行运盘。")
def debug_tool_astro_transit(payload: ToolAstroChartRequest = Body(...)):
    _ensure_debug_tools_enabled()
    result = astro_transit_chart.invoke(
        {
            "birth_dt": payload.birth_dt,
            "longitude": payload.longitude,
            "latitude": payload.latitude,
        }
    )
    return _wrap_tool_result("astro_transit_chart", result)


@app.get("/debug/ui", response_class=HTMLResponse, summary="调试界面", description="简易工具调试界面，仅非生产可用。")
def debug_ui():
        _ensure_debug_tools_enabled()
        return """<!doctype html>
<html lang="zh-CN">
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>工具调试台</title>
        <style>
            :root {
                --bg: #f4f1ea;
                --ink: #1e1e1e;
                --accent: #2f5d50;
                --card: #ffffff;
                --border: #e3ddcf;
            }
            body {
                margin: 0;
                font-family: "Georgia", "Times New Roman", serif;
                background: radial-gradient(circle at top left, #f7efe0 0%, #f4f1ea 45%, #efe7d6 100%);
                color: var(--ink);
            }
            header {
                padding: 24px 32px 12px;
            }
            h1 {
                margin: 0 0 6px;
                font-size: 28px;
                letter-spacing: 0.5px;
            }
            p.sub {
                margin: 0;
                color: #5a5348;
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 16px;
                padding: 16px 32px 32px;
            }
            .card {
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 16px;
                box-shadow: 0 6px 16px rgba(40, 34, 26, 0.08);
            }
            .card h2 {
                margin: 0 0 8px;
                font-size: 18px;
            }
            label {
                display: block;
                font-size: 13px;
                color: #5a5348;
                margin-bottom: 6px;
            }
            input, textarea, select {
                width: 100%;
                padding: 8px 10px;
                border: 1px solid var(--border);
                border-radius: 10px;
                font-size: 14px;
                background: #fcfbf8;
                box-sizing: border-box;
            }
            textarea {
                min-height: 90px;
                resize: vertical;
            }
            button {
                margin-top: 10px;
                background: var(--accent);
                color: #fff;
                border: none;
                border-radius: 10px;
                padding: 8px 14px;
                cursor: pointer;
            }
            pre {
                background: #1b1a17;
                color: #e8e5de;
                padding: 10px;
                border-radius: 10px;
                overflow: auto;
                max-height: 240px;
                font-size: 12px;
            }
            .row {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 8px;
            }
            @media (max-width: 720px) {
                header, .grid { padding: 16px; }
                .row { grid-template-columns: 1fr; }
            }
        </style>
    </head>
    <body>
        <header>
            <h1>工具调试台</h1>
            <p class="sub">仅用于本地/测试环境。返回结构统一为 {tool, ok, code, data, error}。</p>
        </header>
        <section class="grid">
            <div class="card">
                <h2>系统自检</h2>
                <label>scope</label>
                <select id="test-scope">
                    <option value="all">all</option>
                    <option value="astro">astro</option>
                    <option value="vector">vector</option>
                    <option value="search">search</option>
                </select>
                <button onclick="callTool('/tools/test', {scope: byId('test-scope').value}, 'out-test')">执行</button>
                <pre id="out-test"></pre>
            </div>

            <div class="card">
                <h2>Chat</h2>
                <label>query</label>
                <textarea id="chat-query" placeholder="输入对话内容"></textarea>
                <label>session_id</label>
                <input id="chat-session" placeholder="session id" />
                <button onclick="callToolQuery('/chat', {query: byId('chat-query').value, session_id: byId('chat-session').value}, {}, 'out-chat')">执行</button>
                <pre id="out-chat"></pre>
            </div>

            <div class="card">
                <h2>联网搜索</h2>
                <label>query</label>
                <input id="search-query" placeholder="输入搜索关键词" />
                <button onclick="callTool('/tools/search', {query: byId('search-query').value}, 'out-search')">执行</button>
                <pre id="out-search"></pre>
            </div>

            <div class="card">
                <h2>向量检索</h2>
                <label>query</label>
                <input id="vector-query" placeholder="输入检索关键词" />
                <button onclick="callTool('/tools/vector_search', {query: byId('vector-query').value}, 'out-vector')">执行</button>
                <pre id="out-vector"></pre>
            </div>

            <div class="card">
                <h2>add_urls / dry_run</h2>
                <label>url</label>
                <input id="add-url" placeholder="单个URL" />
                <label>urls（每行一个）</label>
                <textarea id="add-urls" placeholder="https://example.com\nhttps://example.org"></textarea>
                <label>chunk_strategy</label>
                <select id="add-strategy">
                    <option value="balanced">balanced</option>
                    <option value="faq">faq</option>
                    <option value="article">article</option>
                    <option value="custom">custom</option>
                </select>
                <div class="row">
                    <div>
                        <label>chunk_size</label>
                        <input id="add-size" placeholder="可选" />
                    </div>
                    <div>
                        <label>chunk_overlap</label>
                        <input id="add-overlap" placeholder="可选" />
                    </div>
                    <div>
                        <label>preview_limit</label>
                        <input id="add-preview" placeholder="默认3" />
                    </div>
                    <div>
                        <label>separators（每行一个）</label>
                        <input id="add-seps" placeholder="仅custom建议填写" />
                    </div>
                </div>
                <button onclick="callTool('/add_urls/dry_run', buildAddUrlsPayload(), 'out-add-dry')">执行</button>
                <pre id="out-add-dry"></pre>
            </div>

            <div class="card">
                <h2>星盘</h2>
                <div class="row">
                    <div>
                        <label>name</label>
                        <input id="xingpan-name" placeholder="姓名" />
                    </div>
                    <div>
                        <label>birth_dt</label>
                        <input id="xingpan-birth" placeholder="1999-10-17 21:00:00" />
                    </div>
                    <div>
                        <label>longitude</label>
                        <input id="xingpan-lng" placeholder="120.1" />
                    </div>
                    <div>
                        <label>latitude</label>
                        <input id="xingpan-lat" placeholder="30.2" />
                    </div>
                </div>
                <button onclick="callTool('/tools/xingpan', {name: byId('xingpan-name').value, birth_dt: byId('xingpan-birth').value, longitude: numVal('xingpan-lng'), latitude: numVal('xingpan-lat')}, 'out-xingpan')">执行</button>
                <pre id="out-xingpan"></pre>
            </div>

            <div class="card">
                <h2>星座信息</h2>
                <button onclick="callTool('/tools/astro/my_sign', {}, 'out-my-sign')">执行</button>
                <pre id="out-my-sign"></pre>
            </div>

            <div class="card">
                <h2>日运势</h2>
                <button onclick="callTool('/tools/astro/day', {}, 'out-day-scope')">执行</button>
                <pre id="out-day-scope"></pre>
            </div>

            <div class="card">
                <h2>周运势</h2>
                <button onclick="callTool('/tools/astro/week', {}, 'out-week-scope')">执行</button>
                <pre id="out-week-scope"></pre>
            </div>

            <div class="card">
                <h2>月运势</h2>
                <button onclick="callTool('/tools/astro/month', {}, 'out-month-scope')">执行</button>
                <pre id="out-month-scope"></pre>
            </div>

            <div class="card">
                <h2>本命盘</h2>
                <div class="row">
                    <div>
                        <label>birth_dt</label>
                        <input id="natal-birth" placeholder="1999-10-17 21:00:00" />
                    </div>
                    <div>
                        <label>longitude</label>
                        <input id="natal-lng" placeholder="120.1" />
                    </div>
                    <div>
                        <label>latitude</label>
                        <input id="natal-lat" placeholder="30.2" />
                    </div>
                </div>
                <button onclick="callTool('/tools/astro/natal_chart', {birth_dt: byId('natal-birth').value, longitude: numVal('natal-lng'), latitude: numVal('natal-lat')}, 'out-natal')">执行</button>
                <pre id="out-natal"></pre>
            </div>

            <div class="card">
                <h2>当前天象盘</h2>
                <button onclick="callTool('/tools/astro/current_chart', {}, 'out-current')">执行</button>
                <pre id="out-current"></pre>
            </div>

            <div class="card">
                <h2>行运盘</h2>
                <div class="row">
                    <div>
                        <label>birth_dt</label>
                        <input id="transit-birth" placeholder="1999-10-17 21:00:00" />
                    </div>
                    <div>
                        <label>longitude</label>
                        <input id="transit-lng" placeholder="120.1" />
                    </div>
                    <div>
                        <label>latitude</label>
                        <input id="transit-lat" placeholder="30.2" />
                    </div>
                </div>
                <button onclick="callTool('/tools/astro/transit_chart', {birth_dt: byId('transit-birth').value, longitude: numVal('transit-lng'), latitude: numVal('transit-lat')}, 'out-transit')">执行</button>
                <pre id="out-transit"></pre>
            </div>
        </section>
        <script>
            function byId(id) { return document.getElementById(id); }
            function linesToList(text) {
                return (text || "")
                    .split(/\r?\n/)
                    .map(function (line) { return line.trim(); })
                    .filter(Boolean);
            }
            function numVal(id) {
                var raw = byId(id).value;
                var val = parseFloat(raw);
                return isNaN(val) ? raw : val;
            }
            function numOrNull(id) {
                var raw = (byId(id).value || "").trim();
                if (!raw) return null;
                var val = parseInt(raw, 10);
                return isNaN(val) ? raw : val;
            }
            async function callTool(path, payload, outputId) {
                var out = byId(outputId);
                out.textContent = "loading...";
                try {
                    var res = await fetch(path, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(payload || {}),
                    });
                    var text = await res.text();
                    try {
                        out.textContent = JSON.stringify(JSON.parse(text), null, 2);
                    } catch (e) {
                        out.textContent = text;
                    }
                } catch (err) {
                    out.textContent = String(err);
                }
            }
            async function callToolQuery(path, queryParams, payload, outputId) {
                var out = byId(outputId);
                out.textContent = "loading...";
                try {
                    var query = new URLSearchParams(queryParams || {}).toString();
                    var url = query ? path + "?" + query : path;
                    var res = await fetch(url, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(payload || {}),
                    });
                    var text = await res.text();
                    try {
                        out.textContent = JSON.stringify(JSON.parse(text), null, 2);
                    } catch (e) {
                        out.textContent = text;
                    }
                } catch (err) {
                    out.textContent = String(err);
                }
            }
            function buildAddUrlsPayload() {
                var payload = {
                    chunk_strategy: byId('add-strategy').value,
                };
                var url = (byId('add-url').value || "").trim();
                if (url) payload.url = url;
                var urls = linesToList(byId('add-urls').value);
                if (urls.length) payload.urls = urls;
                var chunkSize = numOrNull('add-size');
                if (chunkSize !== null) payload.chunk_size = chunkSize;
                var chunkOverlap = numOrNull('add-overlap');
                if (chunkOverlap !== null) payload.chunk_overlap = chunkOverlap;
                var previewLimit = numOrNull('add-preview');
                if (previewLimit !== null) payload.preview_limit = previewLimit;
                var separators = linesToList(byId('add-seps').value);
                if (separators.length) payload.separators = separators;
                return payload;
            }
        </script>
    </body>
</html>
"""

@app.post(
    "/add_urls",
    response_model=AddUrlsResponse,
    summary="URL学习入库",
    description="抓取URL内容、切块并写入向量库。支持 JSON 与 Query 参数。",
)
def add_urls(payload: AddUrlsRequest = Depends(_resolve_add_urls_payload)) -> AddUrlsResponse:
    """抓取并切块后写入向量库。"""
    _ensure_add_urls_write_enabled()

    overall_start = time.perf_counter()
    clean_url_list = _normalize_urls(payload)

    if not clean_url_list:
        raise HTTPException(status_code=400, detail="请提供url或urls参数")

    clean_url_list, blocked_urls = _partition_safe_urls(clean_url_list)
    blocked_urls = _normalize_failed_urls(blocked_urls)
    if not clean_url_list:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "URL全部被安全策略拦截",
                "failed_urls": blocked_urls,
            },
        )

    log_event(
        logging.INFO,
        "add_urls.start",
        url_count=len(clean_url_list),
        strategy=payload.chunk_strategy,
    )

    chunk_cfg = _build_chunking_config(
        strategy=payload.chunk_strategy,
        chunk_size=payload.chunk_size,
        chunk_overlap=payload.chunk_overlap,
        separators=payload.separators,
    )

    collect_start = time.perf_counter()
    all_chunks, failed_urls = _collect_chunks_from_urls(clean_url_list, chunk_cfg, payload.chunk_strategy)
    failed_urls = _normalize_failed_urls([*blocked_urls, *failed_urls])
    quality_report = _compute_chunk_quality_report(all_chunks, failed_urls)
    collect_ms = int((time.perf_counter() - collect_start) * 1000)
    log_event(
        logging.INFO,
        "add_urls.collect",
        url_count=len(clean_url_list),
        chunks=len(all_chunks),
        failed=len(failed_urls),
        elapsed_ms=collect_ms,
    )

    if not all_chunks:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "URL内容加载失败，未生成可写入的文本块",
                "failed_urls": failed_urls,
            },
        )

    qdrant_url = os.getenv("QDRANT_URL", "").strip()
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip() or None
    qdrant_path = os.getenv("QDRANT_DB_PATH", "./qdrant_data/qdrant.db")
    collection_name = os.getenv("QDRANT_COLLECTION", "divination_master_collection")
    embedding_cfg = resolve_embedding_config()
    vector_size = embedding_cfg.dimensions

    if qdrant_url:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        mode = "remote"
    else:
        client = QdrantClient(path=qdrant_path)
        mode = "local"

    # 使用已创建的 QdrantClient 实例，避免 from_documents 在不同版本中的参数兼容问题。
    try:
        embeddings_start = time.perf_counter()
        embeddings_client = _build_embeddings_client(vector_size=vector_size)
        effective_vector_size = _resolve_embedding_output_dim(embeddings_client, default_size=vector_size)
        log_event(
            logging.INFO,
            "add_urls.embeddings.init",
            provider="litellm",
            model=getattr(embeddings_client, "model", ""),
            vector_size=effective_vector_size,
            vector_size_source=embedding_cfg.dimension_source,
            elapsed_ms=int((time.perf_counter() - embeddings_start) * 1000),
        )

        try:
            ensure_start = time.perf_counter()
            _ensure_qdrant_collection(
                client=client,
                collection_name=collection_name,
                vector_size=effective_vector_size,
            )
            log_event(
                logging.INFO,
                "add_urls.qdrant.ensure",
                collection=collection_name,
                mode=mode,
                elapsed_ms=int((time.perf_counter() - ensure_start) * 1000),
            )
        except VectorSizeMismatchError as mismatch_err:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "Qdrant collection vector size mismatch",
                    "collection": mismatch_err.collection_name,
                    "existing_size": mismatch_err.existing_size,
                    "expected_size": mismatch_err.expected_size,
                    "hint": "请调整 EMBEDDINGS_DIMENSION 与集合一致，或调用 /qdrant/recreate 重建集合。",
                },
            )

        vector_store = Qdrant(
            client,
            collection_name,
            embeddings_client,
        )
        try:
            write_start = time.perf_counter()
            vector_store.add_documents(all_chunks)
            log_event(
                logging.INFO,
                "add_urls.qdrant.write",
                collection=collection_name,
                mode=mode,
                chunks=len(all_chunks),
                elapsed_ms=int((time.perf_counter() - write_start) * 1000),
            )
        except Exception as write_err:
            if _is_vector_dim_mismatch_error(write_err):
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": "Qdrant collection vector size mismatch",
                        "collection": collection_name,
                        "expected_size": effective_vector_size,
                        "hint": "请调整 EMBEDDINGS_DIMENSION 与集合一致，或调用 /qdrant/recreate 重建集合。",
                    },
                )
            raise
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "向量写入Qdrant失败",
                "error": str(e)[:400],
                "mode": mode,
                "collection": collection_name,
            },
        )

    logger.info(
        "调用add_urls接口 | mode: %s | source_urls: %s | collection: %s | chunks: %s | failed: %s | strategy: %s | chunk_size: %s | overlap: %s",
        mode,
        len(clean_url_list),
        collection_name,
        len(all_chunks),
        len(failed_urls),
        payload.chunk_strategy,
        chunk_cfg["chunk_size"],
        chunk_cfg["chunk_overlap"],
    )
    log_event(
        logging.INFO,
        "add_urls.done",
        mode=mode,
        collection=collection_name,
        url_count=len(clean_url_list),
        chunks=len(all_chunks),
        failed=len(failed_urls),
        elapsed_ms=int((time.perf_counter() - overall_start) * 1000),
    )
    return AddUrlsResponse(
        response="URLs added!",
        collection=collection_name,
        mode=mode,
        source_urls=len(clean_url_list),
        chunks=len(all_chunks),
        failed_urls=[FailedUrlItem(**item) for item in failed_urls],
        chunk_strategy=payload.chunk_strategy,
        chunk_config=ChunkConfigModel(
            chunk_size=chunk_cfg["chunk_size"],
            chunk_overlap=chunk_cfg["chunk_overlap"],
            separators=chunk_cfg["separators"],
        ),
        quality_report=quality_report,
    )


@app.post(
    "/add_urls/dry_run",
    response_model=AddUrlsDryRunResponse,
    summary="URL切块预览",
    description="仅抓取与切块预览，不写入向量库。支持 JSON 与 Query 参数。",
)
def add_urls_dry_run(payload: AddUrlsRequest = Depends(_resolve_add_urls_payload)) -> AddUrlsDryRunResponse:
    """仅抓取和切块预览，不写入Qdrant。"""
    overall_start = time.perf_counter()
    clean_url_list = _normalize_urls(payload)
    if not clean_url_list:
        raise HTTPException(status_code=400, detail="请提供url或urls参数")

    clean_url_list, blocked_urls = _partition_safe_urls(clean_url_list)
    blocked_urls = _normalize_failed_urls(blocked_urls)
    if not clean_url_list:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "URL全部被安全策略拦截",
                "failed_urls": blocked_urls,
            },
        )

    chunk_cfg = _build_chunking_config(
        strategy=payload.chunk_strategy,
        chunk_size=payload.chunk_size,
        chunk_overlap=payload.chunk_overlap,
        separators=payload.separators,
    )
    collect_start = time.perf_counter()
    all_chunks, failed_urls = _collect_chunks_from_urls(clean_url_list, chunk_cfg, payload.chunk_strategy)
    failed_urls = _normalize_failed_urls([*blocked_urls, *failed_urls])
    quality_report = _compute_chunk_quality_report(all_chunks, failed_urls)
    log_event(
        logging.INFO,
        "add_urls.dry_run.collect",
        url_count=len(clean_url_list),
        chunks=len(all_chunks),
        failed=len(failed_urls),
        elapsed_ms=int((time.perf_counter() - collect_start) * 1000),
    )

    if not all_chunks:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "URL内容加载失败，未生成可预览的文本块",
                "failed_urls": failed_urls,
            },
        )

    preview_limit = min(payload.preview_limit, len(all_chunks))
    chunk_preview = []
    for chunk in all_chunks[:preview_limit]:
        content = (chunk.page_content or "").strip()
        chunk_preview.append(
            {
                "source_url": (chunk.metadata or {}).get("source_url", ""),
                "chunk_index": (chunk.metadata or {}).get("chunk_index", 0),
                "content_length": len(content),
                "content_preview": content[:200],
            }
        )

    logger.info(
        "调用add_urls_dry_run接口 | source_urls: %s | chunks: %s | failed: %s | strategy: %s | chunk_size: %s | overlap: %s",
        len(clean_url_list),
        len(all_chunks),
        len(failed_urls),
        payload.chunk_strategy,
        chunk_cfg["chunk_size"],
        chunk_cfg["chunk_overlap"],
    )
    log_event(
        logging.INFO,
        "add_urls.dry_run.done",
        url_count=len(clean_url_list),
        chunks=len(all_chunks),
        failed=len(failed_urls),
        elapsed_ms=int((time.perf_counter() - overall_start) * 1000),
    )

    return AddUrlsDryRunResponse(
        response="Dry run completed",
        source_urls=len(clean_url_list),
        chunks=len(all_chunks),
        failed_urls=[FailedUrlItem(**item) for item in failed_urls],
        chunk_strategy=payload.chunk_strategy,
        chunk_config=ChunkConfigModel(
            chunk_size=chunk_cfg["chunk_size"],
            chunk_overlap=chunk_cfg["chunk_overlap"],
            separators=chunk_cfg["separators"],
        ),
        chunk_preview=[ChunkPreviewItem(**item) for item in chunk_preview],
        quality_report=quality_report,
    )

@app.post("/add_pdfs", summary="PDF入库占位", description="占位接口：未来支持PDF解析与入库。")
def add_pdfs():
    """占位接口：PDF解析与入库功能待扩展。"""
    logger.info("调用add_pdfs接口")
    return {"response": "PDFs added!"}

@app.post("/add_texts", summary="文本入库占位", description="占位接口：未来支持直接文本入库。")
def add_texts():
    """占位接口：纯文本入库功能待扩展。"""
    logger.info("调用add_texts接口")
    return {"response": "Texts added!"}

@app.post(
    "/qdrant/init",
    summary="Qdrant初始化",
    description="初始化集合，必要时可先删除旧集合再重建。",
)
def init_qdrant(
    collection: Optional[str] = Query(default=None, description="可选：指定要初始化的collection，默认使用QDRANT_COLLECTION"),
    recreate: bool = Query(default=False, description="是否先删除旧集合再重建"),
):
    """初始化Qdrant collection。"""
    try:
        result = init_qdrant_collection(collection_name=collection, force_recreate=recreate)
        logger.info("Qdrant初始化完成 | result: %s", result)
        return {"code": 200, "data": result}
    except Exception as e:
        err = str(e)[:120]
        logger.error("Qdrant初始化失败 | err: %s", err, exc_info=True)
        return {
            "code": 500,
            "error_code": "QDRANT_INIT_ERROR",
            "error": err,
            "explanation": _COMMON_ERROR_EXPLANATIONS["QDRANT_INIT_ERROR"],
        }


@app.post(
    "/qdrant/recreate",
    summary="Qdrant重建",
    description="删除并重建指定集合，默认使用 QDRANT_COLLECTION。",
)
def recreate_qdrant(
    collection: Optional[str] = Query(default=None, description="可选：指定要重建的collection，默认使用QDRANT_COLLECTION"),
):
    """删除并重建Qdrant collection。"""
    target = (collection or os.getenv("QDRANT_COLLECTION", "divination_master_collection")).strip()
    try:
        result = recreate_qdrant_collection(collection_name=target)
        logger.info("Qdrant重建完成 | result: %s", result)
        return {"code": 200, "data": result}
    except Exception as e:
        err = str(e)[:120]
        logger.error("Qdrant重建失败 | err: %s", err, exc_info=True)
        return {
            "code": 500,
            "error_code": "QDRANT_RECREATE_ERROR",
            "error": err,
            "explanation": _COMMON_ERROR_EXPLANATIONS["QDRANT_RECREATE_ERROR"],
        }

@app.get("/qdrant/health", summary="Qdrant健康检查", description="检查Qdrant连通性与可用性。")
def qdrant_health_check():
    """Qdrant连通性检查。"""
    try:
        result = qdrant_health()
        logger.info("Qdrant健康检查 | result: %s", result)
        return {"code": 200, "data": result}
    except Exception as e:
        err = str(e)[:120]
        logger.error("Qdrant健康检查失败 | err: %s", err, exc_info=True)
        return {
            "code": 500,
            "error_code": "QDRANT_HEALTH_ERROR",
            "error": err,
            "explanation": _COMMON_ERROR_EXPLANATIONS["QDRANT_HEALTH_ERROR"],
        }

@app.get("/qdrant/collections", summary="Qdrant集合列表", description="列出当前Qdrant中的集合。")
def qdrant_collections():
    """列出Qdrant collections。"""
    try:
        result = qdrant_list_collections()
        logger.info("Qdrant collections | result: %s", result)
        return {"code": 200, "data": result}
    except Exception as e:
        err = str(e)[:120]
        logger.error("Qdrant collections失败 | err: %s", err, exc_info=True)
        return {
            "code": 500,
            "error_code": "QDRANT_COLLECTIONS_ERROR",
            "error": err,
            "explanation": _COMMON_ERROR_EXPLANATIONS["QDRANT_COLLECTIONS_ERROR"],
        }


@app.get("/qdrant/status", summary="Qdrant状态", description="获取Qdrant仓库的轻量状态信息。")
def qdrant_status():
    """简单监视Qdrant仓库状态。"""
    try:
        result = qdrant_repo_status()
        logger.info("Qdrant status | result: %s", result)
        return {"code": 200, "data": result}
    except Exception as e:
        err = str(e)[:120]
        logger.error("Qdrant status失败 | err: %s", err, exc_info=True)
        return {
            "code": 500,
            "error_code": "QDRANT_STATUS_ERROR",
            "error": err,
            "explanation": _COMMON_ERROR_EXPLANATIONS["QDRANT_STATUS_ERROR"],
        }


@app.get(
    "/embedding/config",
    response_model=EmbeddingConfigResponse,
    summary="Embedding配置",
    description="返回当前生效的 embedding 模型、维度与维度来源。",
)
def embedding_config() -> EmbeddingConfigResponse:
    """读取当前生效 embedding 配置，用于排查维度不一致问题。"""
    cfg = resolve_embedding_config()
    return EmbeddingConfigResponse(
        model=cfg.model,
        dimensions=cfg.dimensions,
        dimension_source=cfg.dimension_source,
        collection=(os.getenv("QDRANT_COLLECTION", "divination_master_collection").strip() or "divination_master_collection"),
        qdrant_distance=(os.getenv("QDRANT_DISTANCE", "cosine").strip().lower() or "cosine"),
    )


@app.get(
    "/rerank/config",
    response_model=RerankConfigResponse,
    summary="Rerank配置",
    description="返回当前 rerank 是否直连上游、路由模型与基础地址。",
)
def rerank_config() -> RerankConfigResponse:
    """读取当前生效 rerank 配置，用于排查 rerank 路由与直连状态。"""
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

@app.get("/health", summary="服务健康检查", description="检查服务与LLM连通性，返回运行状态。")
async def health_check():
    """健康检查接口"""
    try:
        import asyncio

        llm_ok = await asyncio.to_thread(master.check_llm_health)
        llm_status = "ok" if llm_ok else "error"
        health_info = {
            "status": "healthy",
            "timestamp": time.time(),
            "llm_status": llm_status,
            "env": "production" if _is_prod_runtime() else "development"
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
            "explanation": _COMMON_ERROR_EXPLANATIONS["HEALTH_CHECK_ERROR"],
        }


@app.get("/health/live", summary="服务存活检查", description="轻量存活检查，不依赖LLM。")
def health_live():
    """用于容器探针的轻量健康接口，避免外部依赖导致误判。"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "env": "production" if _is_prod_runtime() else "development",
    }

@app.get("/memory/status", summary="会话记忆状态", description="按 session_id 查询会话记忆状态。")
def memory_status(session_id: str):
    """查看指定session的memory状态。"""
    if not session_id.strip():
        raise HTTPException(
            status_code=400,
            detail={
                "message": "session_id不能为空",
                "error_code": "INVALID_SESSION_ID",
                "explanation": _COMMON_ERROR_EXPLANATIONS["INVALID_SESSION_ID"],
            },
        )
    try:
        status = master.get_memory_status(session_id)
        logger.info("memory状态查询成功 | session_id: %s | count: %s", status["session_id"], status["message_count"])
        return {"code": 200, "data": status}
    except Exception as e:
        err = str(e)[:120]
        logger.error("memory状态查询失败 | session_id: %s | err: %s", session_id, err, exc_info=True)
        return {
            "code": 500,
            "error_code": "MEMORY_STATUS_ERROR",
            "session_id": session_id,
            "error": err,
            "explanation": _COMMON_ERROR_EXPLANATIONS["MEMORY_STATUS_ERROR"],
        }

@app.websocket("/ws")
async def ws(websocket: WebSocket):
    """WebSocket对话通道，使用 query 参数 session_id 进行会话隔离。"""
    if _is_prod_runtime():
        await websocket.close(code=1008, reason="Not Found")
        return

    await websocket.accept()
    client_ip = websocket.client.host
    session_id = (websocket.query_params.get("session_id") or "").strip()
    if not session_id:
        logger.warning("WebSocket连接拒绝 | 客户端IP: %s | 原因: session_id不能为空", client_ip)
        await websocket.close(code=1008, reason="session_id不能为空")
        return
    ws_start = time.perf_counter()
    logger.info(f"WebSocket连接建立 | 客户端IP: {client_ip} | session_id: {session_id}")
    log_event(
        logging.INFO,
        "ws.open",
        client_ip=client_ip,
        session_id=session_id,
    )
    
    try:
        while True:
            data = await websocket.receive_text()
            msg_start = time.perf_counter()
            logger.info(f"WebSocket接收消息 | 客户端IP: {client_ip} | session_id: {session_id} | 消息: {data[:100]}")
            
            res = await master.run_async(data, session_id=session_id)
            clean_response = res.get("output", USER_MESSAGES["ws_default"])
            
            await websocket.send_text(clean_response)
            logger.info(f"WebSocket发送消息 | 客户端IP: {client_ip} | 响应长度: {len(clean_response)}")
            log_event(
                logging.INFO,
                "ws.message",
                client_ip=client_ip,
                session_id=session_id,
                input_len=len(data),
                output_len=len(clean_response),
                elapsed_ms=int((time.perf_counter() - msg_start) * 1000),
            )
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket连接断开 | 客户端IP: {client_ip}")
        log_event(
            logging.INFO,
            "ws.close",
            client_ip=client_ip,
            session_id=session_id,
            elapsed_ms=int((time.perf_counter() - ws_start) * 1000),
        )
    except Exception as e:
        logger.error(f"WebSocket异常 | 客户端IP: {client_ip} | 错误: {str(e)[:100]}", exc_info=True)
        log_event(
            logging.ERROR,
            "ws.error",
            client_ip=client_ip,
            session_id=session_id,
            error=str(e)[:160],
            elapsed_ms=int((time.perf_counter() - ws_start) * 1000),
        )
        await websocket.close(code=1011, reason="服务器内部错误")

if __name__ == "__main__":
    import uvicorn
    api_host = os.getenv("API_HOST", "127.0.0.1").strip() or "127.0.0.1"
    try:
        api_port = int(os.getenv("API_PORT", "8000"))
    except ValueError:
        api_port = 8000
    logger.info("启动FastAPI服务 | 地址: %s:%s", api_host, api_port)
    uvicorn.run(app, host=api_host, port=api_port)