import os
import json
import sys
import time
import logging
import requests
from typing import Any, Dict, List, Literal, Optional
from fastapi import Body, Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

# 兼容 `python app/main.py` 直跑场景：把项目根目录加入模块搜索路径。
if __package__ in (None, ""):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from app.core.config import IS_PROD
from app.core.logger_setup import logger, log_event
from app.services.master_service import Master
from app.tools.mytools import (
    astro_current_chart,
    astro_my_sign,
    astro_natal_chart,
    astro_transit_chart,
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

app = FastAPI()
master = Master()


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
    error: str = Field(description="失败原因")


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
    if IS_PROD:
        raise HTTPException(status_code=404, detail="Not Found")


def _wrap_tool_result(tool_name: str, raw_result: Any) -> Dict[str, Any]:
    parsed = None
    if isinstance(raw_result, str):
        try:
            parsed = json.loads(raw_result)
        except Exception:
            parsed = None

    if isinstance(parsed, dict) and {"ok", "code", "data", "error"}.issubset(parsed.keys()):
        return {"tool": tool_name, **parsed}

    return {
        "tool": tool_name,
        "ok": True,
        "code": "OK",
        "data": parsed if parsed is not None else raw_result,
        "error": "",
    }


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
            failed_urls.append({"url": one_url, "error": str(e)[:400]})
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


def _resolve_embeddings_dimension(default_value: int) -> int:
    raw = os.getenv("EMBEDDINGS_DIMENSION", "").strip()
    if not raw:
        return default_value
    try:
        value = int(raw)
        if value > 0:
            return value
    except ValueError:
        pass
    logger.warning("EMBEDDINGS_DIMENSION 非法，已回退默认值: %s", default_value)
    return default_value


def _resolve_target_vector_size(default_value: int) -> int:
    """仅读取EMBEDDINGS_DIMENSION作为向量维度来源。"""
    raw = os.getenv("EMBEDDINGS_DIMENSION", "").strip()
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            logger.warning("EMBEDDINGS_DIMENSION 非法，已回退默认值: %s", default_value)

    return default_value


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


def _normalize_openai_base_url(raw_base_url: str) -> str:
    """规范化 OPENAI_API_BASE，确保配置不包含 /v1。"""
    base = (raw_base_url or "").strip().rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base.rstrip("/")


def _resolve_local_embedding_model(model_name: str, cache_dir: str) -> str:
    """本地Embedding模型路径解析。

    兼容两种输入：
    - 直接传本地目录（推荐）
    - 传仓库名（如 BAAI/bge-small-zh-v1.5），若 cache_dir 下存在同名目录则自动映射
    """
    clean_model = (model_name or "").strip()
    if not clean_model:
        return clean_model

    if os.path.isdir(clean_model):
        return clean_model

    if cache_dir and "/" in clean_model:
        candidate = os.path.join(cache_dir, *clean_model.split("/"))
        if os.path.isdir(candidate):
            return candidate

    return clean_model


def _build_embeddings_client(vector_size: int):
    """按 EMBEDDINGS_API 构造 Embeddings 客户端。"""
    provider = os.getenv("EMBEDDINGS_API", "openai").strip().lower()
    embedding_model = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small").strip() or "text-embedding-3-small"

    if provider == "local":
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except Exception as e:
            raise RuntimeError("EMBEDDINGS_API=local 需要安装 sentence-transformers") from e
        hf_endpoint = os.getenv("EMBEDDINGS_HF_ENDPOINT", "").strip()
        local_files_only = os.getenv("EMBEDDINGS_LOCAL_FILES_ONLY", "false").strip().lower() in {"1", "true", "yes", "on"}
        cache_dir = os.getenv("EMBEDDINGS_CACHE_DIR", "").strip()
        resolved_model = _resolve_local_embedding_model(embedding_model, cache_dir)

        # 仅在 Embedding 初始化路径下设置 HF 运行时环境，避免影响其他模块。
        if hf_endpoint and not local_files_only:
            os.environ["HF_ENDPOINT"] = hf_endpoint.rstrip("/")
        if local_files_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if cache_dir:
            os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_dir

        model_kwargs: Dict[str, Any] = {"local_files_only": local_files_only}

        try:
            return HuggingFaceEmbeddings(
                model_name=resolved_model,
                cache_folder=(cache_dir or None),
                model_kwargs=model_kwargs,
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as e:
            expected_path = os.path.join(cache_dir, "BAAI", "bge-small-zh-v1.5") if cache_dir else ""
            hint = (
                "本地模型不可用。请先预下载模型到缓存目录，"
                "或设置 EMBEDDINGS_HF_ENDPOINT 到可访问镜像站，"
                "或将 EMBEDDINGS_API 切回 openai。"
            )
            raise RuntimeError(
                f"初始化本地Embedding失败: {str(e)[:240]} | model={resolved_model} | expected_dir={expected_path} | {hint}"
            ) from e

    api_base = _normalize_openai_base_url(os.getenv("OPENAI_API_BASE", ""))
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    embedding_kwargs: Dict[str, Any] = {
        "model": embedding_model,
        "dimensions": _resolve_embeddings_dimension(vector_size),
        # 某些代理模型不在 tiktoken 映射表中，固定编码可避免无意义告警。
        "tiktoken_model_name": "cl100k_base",
    }

    if api_base:
        embedding_kwargs["openai_api_base"] = f"{api_base}/v1"

    if api_key:
        embedding_kwargs["openai_api_key"] = api_key

    return OpenAIEmbeddings(**embedding_kwargs)

@app.get("/")
def read_root():
    logger.info("访问根路径")
    return {"Hello": "World"}

@app.post("/chat")
def chat(query: str, session_id: str):
    logger.info(f"接收Chat API请求 | session_id: {session_id} | 查询: {query[:100]}")
    if not session_id.strip():
        raise HTTPException(status_code=400, detail="session_id不能为空")
    try:
        res = master.run(query, session_id=session_id)
        response_text = res.get("output", "")
        logger.info(f"Chat API响应成功 | session_id: {session_id} | 查询: {query[:50]} | 响应长度: {len(response_text)}")
        return {"code": 200, "session_id": session_id, "query": query, "response": response_text}
    except Exception as e:
        error_msg = str(e)[:100]
        logger.error(f"Chat API执行异常 | 查询: {query[:50]} | 错误: {error_msg}", exc_info=True)
        return {"code": 500, "session_id": session_id, "query": query, "response": f"错误：{error_msg}"}


@app.post("/tools/test", summary="工具调试：系统自检", description="调用 test 工具，支持 scope=all|astro|vector|search 用于快速排查配置。")
def debug_tool_test(payload: ToolTestRequest = Body(...)):
    _ensure_debug_tools_enabled()
    result = test(scope=payload.scope or "all")
    return _wrap_tool_result("test", result)


@app.post("/tools/search", summary="工具调试：联网搜索", description="调用 search 工具，使用 SERPAPI 查询外部信息。")
def debug_tool_search(payload: ToolSearchRequest = Body(...)):
    _ensure_debug_tools_enabled()
    result = search(query=payload.query)
    return _wrap_tool_result("search", result)


@app.post("/tools/vector_search", summary="工具调试：向量检索", description="调用 vector_search 工具，从已入库文档中检索相似内容。")
def debug_tool_vector_search(payload: ToolVectorSearchRequest = Body(...)):
    _ensure_debug_tools_enabled()
    result = vector_search(query=payload.query)
    return _wrap_tool_result("vector_search", result)


@app.post("/tools/xingpan", summary="工具调试：星盘", description="调用 xingpan 工具，依据姓名、出生时间、经纬度查询星盘。")
def debug_tool_xingpan(payload: ToolXingpanRequest = Body(...)):
    _ensure_debug_tools_enabled()
    result = xingpan(
        name=payload.name,
        birth_dt=payload.birth_dt,
        longitude=payload.longitude,
        latitude=payload.latitude,
    )
    return _wrap_tool_result("xingpan", result)


@app.post("/tools/astro/my_sign", summary="工具调试：星座信息", description="调用 astro_my_sign 工具，读取 ASTRO_UID/UID 并查询星座信息。")
def debug_tool_astro_my_sign():
    _ensure_debug_tools_enabled()
    result = astro_my_sign()
    return _wrap_tool_result("astro_my_sign", result)


@app.post("/tools/astro/natal_chart", summary="工具调试：本命盘", description="调用 astro_natal_chart 工具，依据出生时间与经纬度查询本命盘。")
def debug_tool_astro_natal(payload: ToolAstroChartRequest = Body(...)):
    _ensure_debug_tools_enabled()
    result = astro_natal_chart(
        birth_dt=payload.birth_dt,
        longitude=payload.longitude,
        latitude=payload.latitude,
    )
    return _wrap_tool_result("astro_natal_chart", result)


@app.post("/tools/astro/current_chart", summary="工具调试：当前天象盘", description="调用 astro_current_chart 工具，查询当前天象盘。")
def debug_tool_astro_current():
    _ensure_debug_tools_enabled()
    result = astro_current_chart()
    return _wrap_tool_result("astro_current_chart", result)


@app.post("/tools/astro/transit_chart", summary="工具调试：行运盘", description="调用 astro_transit_chart 工具，依据出生时间与经纬度查询行运盘。")
def debug_tool_astro_transit(payload: ToolAstroChartRequest = Body(...)):
    _ensure_debug_tools_enabled()
    result = astro_transit_chart(
        birth_dt=payload.birth_dt,
        longitude=payload.longitude,
        latitude=payload.latitude,
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
            function numVal(id) {
                var raw = byId(id).value;
                var val = parseFloat(raw);
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
        </script>
    </body>
</html>
"""

@app.post("/add_urls", response_model=AddUrlsResponse)
def add_urls(payload: AddUrlsRequest = Depends(_resolve_add_urls_payload)) -> AddUrlsResponse:
    overall_start = time.perf_counter()
    clean_url_list = _normalize_urls(payload)

    if not clean_url_list:
        raise HTTPException(status_code=400, detail="请提供url或urls参数")

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
    vector_size = _resolve_target_vector_size(default_value=384)

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
            provider=os.getenv("EMBEDDINGS_API", "openai").strip().lower(),
            vector_size=effective_vector_size,
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


@app.post("/add_urls/dry_run", response_model=AddUrlsDryRunResponse)
def add_urls_dry_run(payload: AddUrlsRequest = Depends(_resolve_add_urls_payload)) -> AddUrlsDryRunResponse:
    """仅抓取和切块预览，不写入Qdrant。"""
    overall_start = time.perf_counter()
    clean_url_list = _normalize_urls(payload)
    if not clean_url_list:
        raise HTTPException(status_code=400, detail="请提供url或urls参数")

    chunk_cfg = _build_chunking_config(
        strategy=payload.chunk_strategy,
        chunk_size=payload.chunk_size,
        chunk_overlap=payload.chunk_overlap,
        separators=payload.separators,
    )
    collect_start = time.perf_counter()
    all_chunks, failed_urls = _collect_chunks_from_urls(clean_url_list, chunk_cfg, payload.chunk_strategy)
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

@app.post("/add_pdfs")
def add_pdfs():
    logger.info("调用add_pdfs接口")
    return {"response": "PDFs added!"}

@app.post("/add_texts")
def add_texts():
    logger.info("调用add_texts接口")
    return {"response": "Texts added!"}

@app.post("/qdrant/init")
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
        return {"code": 500, "error": err}


@app.post("/qdrant/recreate")
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
        return {"code": 500, "error": err}

@app.get("/qdrant/health")
def qdrant_health_check():
    """Qdrant连通性检查。"""
    try:
        result = qdrant_health()
        logger.info("Qdrant健康检查 | result: %s", result)
        return {"code": 200, "data": result}
    except Exception as e:
        err = str(e)[:120]
        logger.error("Qdrant健康检查失败 | err: %s", err, exc_info=True)
        return {"code": 500, "error": err}

@app.get("/qdrant/collections")
def qdrant_collections():
    """列出Qdrant collections。"""
    try:
        result = qdrant_list_collections()
        logger.info("Qdrant collections | result: %s", result)
        return {"code": 200, "data": result}
    except Exception as e:
        err = str(e)[:120]
        logger.error("Qdrant collections失败 | err: %s", err, exc_info=True)
        return {"code": 500, "error": err}


@app.get("/qdrant/status")
def qdrant_status():
    """简单监视Qdrant仓库状态。"""
    try:
        result = qdrant_repo_status()
        logger.info("Qdrant status | result: %s", result)
        return {"code": 200, "data": result}
    except Exception as e:
        err = str(e)[:120]
        logger.error("Qdrant status失败 | err: %s", err, exc_info=True)
        return {"code": 500, "error": err}

@app.get("/health")
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
            "env": "production" if IS_PROD else "development"
        }
        logger.info(f"健康检查 | 状态: {health_info}")
        return health_info
    except Exception as e:
        logger.error(f"健康检查异常 | 错误: {str(e)[:100]}", exc_info=True)
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e)[:100]
        }

@app.get("/memory/status")
def memory_status(session_id: str):
    """查看指定session的memory状态。"""
    if not session_id.strip():
        raise HTTPException(status_code=400, detail="session_id不能为空")
    try:
        status = master.get_memory_status(session_id)
        logger.info("memory状态查询成功 | session_id: %s | count: %s", status["session_id"], status["message_count"])
        return {"code": 200, "data": status}
    except Exception as e:
        err = str(e)[:120]
        logger.error("memory状态查询失败 | session_id: %s | err: %s", session_id, err, exc_info=True)
        return {"code": 500, "session_id": session_id, "error": err}

@app.websocket("/ws")
async def ws(websocket: WebSocket):
    """WebSocket接口"""
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