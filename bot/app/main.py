import os
import asyncio
import sys
import time
import logging
import socket
import ipaddress
import uuid
import requests
from urllib.parse import urlparse
from typing import Any, Dict, List, Literal, Optional
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
from app.core.config import (
    build_config_health_summary,
    get_env_float,
    get_env_int,
    get_gateway_security_settings,
    get_qdrant_settings,
    get_runtime_settings,
    get_server_settings,
    is_prod_runtime,
)
from app.core.gateway_security import (
    FixedWindowRateLimiter,
    build_http_audit_context,
    ensure_body_size_limit,
    ensure_http_auth,
    ensure_http_rate_limit,
    ensure_websocket_auth,
    ensure_ws_rate_limit,
    log_http_gateway_audit,
)
from app.core.logger_setup import (
    clear_trace_id,
    get_trace_id,
    logger,
    log_event,
    set_trace_id,
)
from app.core.litellm_adapters import build_litellm_embeddings_client
from app.api.routers import conversation as conversation_router
from app.api.routers import ingestion as ingestion_router
from app.api.routers import ops as ops_router
from app.api.routers import tools as tools_router
from app.schemas.add_urls import (
    AddUrlsErrorCode,
    AddUrlsRequest,
)
from app.services import add_urls_service
from app.services.master_service import Master
from app.services.qdrant_service import (
    init_qdrant_collection,
    qdrant_health,
    qdrant_list_collections,
    qdrant_repo_status,
    recreate_qdrant_collection,
)


def _is_prod_runtime() -> bool:
    return is_prod_runtime()

app = FastAPI(
    docs_url=None if _is_prod_runtime() else "/docs",
    redoc_url=None if _is_prod_runtime() else "/redoc",
    openapi_url=None if _is_prod_runtime() else "/openapi.json",
)

app.include_router(conversation_router.router)
app.include_router(tools_router.router)
app.include_router(ingestion_router.router)
app.include_router(ops_router.router)

gateway_security_settings = get_gateway_security_settings()
gateway_rate_limiter = FixedWindowRateLimiter()

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(gateway_security_settings.cors_allow_origins),
    allow_credentials=gateway_security_settings.cors_allow_credentials,
    allow_methods=list(gateway_security_settings.cors_allow_methods),
    allow_headers=list(gateway_security_settings.cors_allow_headers),
)

master = Master()


def _log_config_health_summary() -> None:
    summary = build_config_health_summary()
    level = logging.INFO if summary.get("ok", False) else logging.ERROR
    log_event(
        level,
        "config.health.summary",
        ok=bool(summary.get("ok", False)),
        warnings=list(summary.get("warnings", ())),
        errors=list(summary.get("errors", ())),
        **dict(summary.get("highlights", {})),
    )


_log_config_health_summary()

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
    "HTTP_429": "请求频率过高，请稍后重试。",
    "HTTP_413": "请求体超出限制，请缩减请求大小。",
    "HTTP_500": "服务内部错误，请稍后重试。",
    "HTTP_504": "请求处理超时，请稍后重试。",
    "INTERNAL_SERVER_ERROR": "服务内部发生未预期错误，请稍后重试。",
    "GATEWAY_AUTH_FAILED": "网关鉴权失败，请检查访问凭据。",
    "GATEWAY_RATE_LIMIT_IP": "请求频率过高（IP维度），请稍后重试。",
    "GATEWAY_RATE_LIMIT_SESSION": "请求频率过高（会话维度），请稍后重试。",
    "GATEWAY_BODY_TOO_LARGE": "请求体过大，请缩减后重试。",
    "GATEWAY_TIMEOUT": "网关请求超时，请稍后重试。",
}


def _resolve_request_id(request: Request) -> str:
    existing = getattr(request.state, "request_id", "")
    if existing:
        return existing

    incoming = (request.headers.get("x-request-id") or "").strip()
    request_id = incoming or uuid.uuid4().hex[:16]
    request.state.request_id = request_id
    return request_id


def _resolve_trace_id(request: Request) -> str:
    existing = getattr(request.state, "trace_id", "")
    if existing:
        return existing

    incoming = (request.headers.get("x-trace-id") or "").strip()
    if not incoming:
        incoming = (request.headers.get("x-request-id") or "").strip()
    trace_id = set_trace_id(incoming)
    request.state.trace_id = trace_id
    return trace_id


def _attach_trace_headers(response: JSONResponse, request_id: str, trace_id: str) -> JSONResponse:
    if request_id:
        response.headers["X-Request-Id"] = request_id
    if trace_id:
        response.headers["X-Trace-Id"] = trace_id
    return response


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


_GATEWAY_EXEMPT_PATHS = {
    "/",
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/favicon.ico",
}


@app.middleware("http")
async def _gateway_guard(request: Request, call_next):
    request_id = _resolve_request_id(request)
    trace_id = _resolve_trace_id(request)

    # CORS preflight must pass through auth/rate-limit checks.
    if request.method.upper() == "OPTIONS":
        response = await call_next(request)
        if request_id:
            response.headers["X-Request-Id"] = request_id
        if trace_id:
            response.headers["X-Trace-Id"] = trace_id
        clear_trace_id()
        return response

    path = request.url.path
    if path in _GATEWAY_EXEMPT_PATHS:
        response = await call_next(request)
        if request_id:
            response.headers["X-Request-Id"] = request_id
        if trace_id:
            response.headers["X-Trace-Id"] = trace_id
        clear_trace_id()
        return response

    # Keep existing prod 404 behavior for non-exposed routes.
    if _is_prod_runtime() and path not in _PROD_ALLOWED_HTTP_PATHS:
        response = await call_next(request)
        if request_id:
            response.headers["X-Request-Id"] = request_id
        if trace_id:
            response.headers["X-Trace-Id"] = trace_id
        clear_trace_id()
        return response

    start = time.perf_counter()
    auth_scheme = "none"
    audit_ctx = None
    try:
        ensure_body_size_limit(gateway_security_settings, request)
        auth_scheme = ensure_http_auth(gateway_security_settings, request)
        ensure_http_rate_limit(gateway_security_settings, request, gateway_rate_limiter)

        audit_ctx = build_http_audit_context(
            request=request,
            auth_scheme=auth_scheme,
            trust_proxy_headers=gateway_security_settings.trust_proxy_headers,
        )
        request.state.request_id = audit_ctx.request_id
        request.state.trace_id = trace_id

        timeout_seconds = gateway_security_settings.request_timeout_seconds
        if timeout_seconds and timeout_seconds > 0:
            response = await asyncio.wait_for(call_next(request), timeout=timeout_seconds)
        else:
            response = await call_next(request)
        if audit_ctx is not None:
            response.headers["X-Request-Id"] = audit_ctx.request_id
            if trace_id:
                response.headers["X-Trace-Id"] = trace_id
            log_http_gateway_audit(
                logging.INFO,
                "gateway.request",
                ctx=audit_ctx,
                status_code=response.status_code,
                elapsed_ms=int((time.perf_counter() - start) * 1000),
            )
        return response
    except HTTPException as exc:
        if audit_ctx is None:
            audit_ctx = build_http_audit_context(
                request=request,
                auth_scheme=auth_scheme,
                trust_proxy_headers=gateway_security_settings.trust_proxy_headers,
            )
        log_http_gateway_audit(
            logging.WARNING,
            "gateway.request.blocked",
            ctx=audit_ctx,
            status_code=exc.status_code,
            elapsed_ms=int((time.perf_counter() - start) * 1000),
        )
        raise
    except asyncio.TimeoutError:
        if audit_ctx is None:
            audit_ctx = build_http_audit_context(
                request=request,
                auth_scheme=auth_scheme,
                trust_proxy_headers=gateway_security_settings.trust_proxy_headers,
            )
        log_http_gateway_audit(
            logging.WARNING,
            "gateway.request.timeout",
            ctx=audit_ctx,
            status_code=504,
            elapsed_ms=int((time.perf_counter() - start) * 1000),
        )
        raise HTTPException(
            status_code=504,
            detail={
                "message": "网关请求超时",
                "error_code": "GATEWAY_TIMEOUT",
            },
        )
    finally:
        clear_trace_id()


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


def _build_error_response_payload(request: Request, status_code: int, normalized_payload: Dict[str, Any]) -> Dict[str, Any]:
    error_code = str(normalized_payload.get("error_code") or f"HTTP_{status_code}")
    explanation = normalized_payload.get("explanation") or _resolve_explanation_by_error_code(error_code)
    request_id = getattr(request.state, "request_id", "")
    trace_id = getattr(request.state, "trace_id", "") or get_trace_id()

    detail = normalized_payload.get("detail", "请求处理失败")
    message = None
    if isinstance(detail, dict):
        message = detail.get("message") or detail.get("detail")
    if not message:
        message = str(detail)

    response_payload: Dict[str, Any] = {
        "ok": False,
        "code": error_code,
        "message": message,
        "data": None,
        "error": {
            "status_code": status_code,
            "error_code": error_code,
            "explanation": explanation,
            "detail": detail,
        },
        # 兼容历史字段
        "detail": detail,
        "error_code": error_code,
        "explanation": explanation,
    }

    errors = normalized_payload.get("errors")
    if errors is not None:
        response_payload["errors"] = errors
        response_payload["error"]["errors"] = errors

    if request_id:
        response_payload["request_id"] = request_id
    if trace_id:
        response_payload["trace_id"] = trace_id

    return response_payload


@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException):
    payload = _normalize_http_exception_payload(exc.status_code, exc.detail)
    response_payload = _build_error_response_payload(request, exc.status_code, payload)
    response = JSONResponse(status_code=exc.status_code, content=response_payload)
    return _attach_trace_headers(
        response,
        request_id=getattr(request.state, "request_id", ""),
        trace_id=getattr(request.state, "trace_id", "") or get_trace_id(),
    )


@app.exception_handler(RequestValidationError)
async def _request_validation_exception_handler(request: Request, exc: RequestValidationError):
    payload = {
        "detail": "请求参数校验失败",
        "error_code": "REQUEST_VALIDATION_ERROR",
        "explanation": _resolve_explanation_by_error_code("REQUEST_VALIDATION_ERROR"),
        "errors": exc.errors(),
    }
    response_payload = _build_error_response_payload(request, 422, payload)
    response = JSONResponse(status_code=422, content=response_payload)
    return _attach_trace_headers(
        response,
        request_id=getattr(request.state, "request_id", ""),
        trace_id=getattr(request.state, "trace_id", "") or get_trace_id(),
    )


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("未捕获异常 | path: %s | err: %s", request.url.path, str(exc)[:200], exc_info=True)
    payload = {
        "detail": "Internal Server Error",
        "error_code": "INTERNAL_SERVER_ERROR",
        "explanation": _resolve_explanation_by_error_code("INTERNAL_SERVER_ERROR"),
    }
    response_payload = _build_error_response_payload(request, 500, payload)
    response = JSONResponse(status_code=500, content=response_payload)
    return _attach_trace_headers(
        response,
        request_id=getattr(request.state, "request_id", ""),
        trace_id=getattr(request.state, "trace_id", "") or get_trace_id(),
    )


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


def _get_add_urls_fetch_timeout_seconds() -> float:
    return get_env_float("ADD_URLS_FETCH_TIMEOUT_SECONDS", default=10.0, min_value=1.0)


def _get_add_urls_fetch_retry_count() -> int:
    return get_env_int("ADD_URLS_FETCH_RETRY_COUNT", default=2, min_value=0)


def _get_add_urls_fetch_backoff_seconds() -> float:
    return get_env_float("ADD_URLS_FETCH_BACKOFF_SECONDS", default=1.0, min_value=0.0)


def _get_add_urls_max_content_chars() -> int:
    return get_env_int("ADD_URLS_MAX_CONTENT_CHARS", default=20000, min_value=1)


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
    runtime_settings = get_runtime_settings()
    if runtime_settings.is_prod and not runtime_settings.add_urls_write_enabled:
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
    verify_ssl = get_runtime_settings().web_loader_verify_ssl
    fetch_timeout_seconds = _get_add_urls_fetch_timeout_seconds()
    fetch_retry_count = _get_add_urls_fetch_retry_count()
    fetch_backoff_seconds = _get_add_urls_fetch_backoff_seconds()
    max_content_chars = _get_add_urls_max_content_chars()
    if not verify_ssl:
        logger.warning("WebBaseLoader SSL校验已关闭（WEB_LOADER_VERIFY_SSL=false），仅建议临时排障使用")

    for one_url in clean_url_list:
        last_error: Optional[Exception] = None
        documents = None
        for attempt in range(fetch_retry_count + 1):
            try:
                web_loader = WebBaseLoader(
                    one_url,
                    verify_ssl=verify_ssl,
                    continue_on_failure=False,
                    raise_for_status=True,
                    requests_kwargs={"timeout": fetch_timeout_seconds},
                )
                documents = web_loader.load()
                break
            except Exception as exc:
                last_error = exc
                if attempt < fetch_retry_count:
                    sleep_seconds = fetch_backoff_seconds * (2 ** attempt)
                    logger.warning(
                        "URL抓取失败，准备重试 | url: %s | attempt: %s/%s | sleep_seconds: %.2f | err: %s",
                        one_url,
                        attempt + 1,
                        fetch_retry_count + 1,
                        sleep_seconds,
                        str(exc)[:200],
                    )
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)
                    continue

        if documents is None:
            failed_urls.append(
                {
                    "url": one_url,
                    "code": AddUrlsErrorCode.FETCH_ERROR.value,
                    "error": str(last_error)[:400] if last_error else "抓取失败",
                    "explanation": _get_add_urls_error_explanation(AddUrlsErrorCode.FETCH_ERROR.value),
                }
            )
            continue

        truncated_docs = 0
        normalized_documents = []
        for doc in documents:
            page_content = getattr(doc, "page_content", "")
            if not isinstance(page_content, str):
                page_content = str(page_content or "")
            original_length = len(page_content)
            if original_length > max_content_chars:
                page_content = page_content[:max_content_chars]
                truncated_docs += 1
            doc.page_content = page_content
            if original_length > max_content_chars:
                metadata = dict(getattr(doc, "metadata", {}) or {})
                metadata["content_truncated"] = True
                metadata["content_original_length"] = original_length
                metadata["content_max_length"] = max_content_chars
                doc.metadata = metadata
            normalized_documents.append(doc)

        if truncated_docs:
            logger.info(
                "URL内容已截断以控制最大长度 | url: %s | truncated_docs: %s | max_content_chars: %s",
                one_url,
                truncated_docs,
                max_content_chars,
            )

        all_chunks.extend(_chunk_documents(normalized_documents, one_url, chunk_cfg, strategy))
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
    qdrant_settings = get_qdrant_settings()
    qdrant_url = qdrant_settings.url
    if not qdrant_url:
        return None

    api_key = qdrant_settings.api_key
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
    distance_name = get_qdrant_settings().distance
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


# 兼容导出说明：
# 下列符号由路由层与测试用例通过 app.main.* 访问（含 monkeypatch）。
# 这里保留稳定的符号名，并将实现委托给 service 层，避免重构期间破坏契约。
_build_chunking_config = add_urls_service._build_chunking_config
_chunk_documents = add_urls_service._chunk_documents
_normalize_urls = add_urls_service._normalize_urls
_get_add_urls_fetch_timeout_seconds = add_urls_service._get_add_urls_fetch_timeout_seconds
_get_add_urls_fetch_retry_count = add_urls_service._get_add_urls_fetch_retry_count
_get_add_urls_fetch_backoff_seconds = add_urls_service._get_add_urls_fetch_backoff_seconds
_get_add_urls_max_content_chars = add_urls_service._get_add_urls_max_content_chars
_get_add_urls_error_explanation = add_urls_service._get_add_urls_error_explanation
_normalize_failed_urls = add_urls_service._normalize_failed_urls
_is_public_http_url = add_urls_service._is_public_http_url
_partition_safe_urls = add_urls_service._partition_safe_urls
_ensure_add_urls_write_enabled = add_urls_service._ensure_add_urls_write_enabled
_collect_chunks_from_urls = add_urls_service._collect_chunks_from_urls
_compute_chunk_quality_report = add_urls_service._compute_chunk_quality_report
_resolve_embedding_output_dim = add_urls_service._resolve_embedding_output_dim
VectorSizeMismatchError = add_urls_service.VectorSizeMismatchError
_extract_collection_vector_size = add_urls_service._extract_collection_vector_size
_fetch_collection_vector_size_via_http = add_urls_service._fetch_collection_vector_size_via_http
_recreate_collection_with_dim = add_urls_service._recreate_collection_with_dim
_is_vector_dim_mismatch_error = add_urls_service._is_vector_dim_mismatch_error
_resolve_qdrant_distance = add_urls_service._resolve_qdrant_distance
_ensure_qdrant_collection = add_urls_service._ensure_qdrant_collection
_build_embeddings_client = add_urls_service._build_embeddings_client

if __name__ == "__main__":
    import uvicorn
    server_settings = get_server_settings()
    api_host = server_settings.host
    api_port = server_settings.port
    logger.info("启动FastAPI服务 | 地址: %s:%s", api_host, api_port)
    uvicorn.run(app, host=api_host, port=api_port)