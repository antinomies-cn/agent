import hashlib
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

try:
    import redis
except Exception:  # pragma: no cover - optional in minimal test env
    redis = None
from fastapi import HTTPException, Request, WebSocket

from app.core.config import GatewaySecuritySettings, get_redis_settings
from app.core.logger_setup import log_event


@dataclass(frozen=True)
class GatewayAuditContext:
    request_id: str
    client_ip: str
    method: str
    path: str
    session_hash: str
    auth_scheme: str


def _build_redis_url_from_env() -> str:
    return get_redis_settings().url


def _extract_client_ip_from_headers(headers: Dict[str, str], fallback_ip: str, trust_proxy_headers: bool) -> str:
    if trust_proxy_headers:
        forwarded_for = (headers.get("x-forwarded-for") or "").strip()
        if forwarded_for:
            first = forwarded_for.split(",", 1)[0].strip()
            if first:
                return first
        real_ip = (headers.get("x-real-ip") or "").strip()
        if real_ip:
            return real_ip
    return fallback_ip or "unknown"


def _extract_token_from_headers(headers: Dict[str, str]) -> Tuple[str, str]:
    auth_header = (headers.get("authorization") or "").strip()
    if auth_header:
        parts = auth_header.split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip(), "bearer"

    api_key = (headers.get("x-api-key") or "").strip()
    if api_key:
        return api_key, "x-api-key"

    return "", "none"


def _hash_session_id(session_id: str) -> str:
    text = (session_id or "").strip()
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _extract_session_id_for_guard(request: Request) -> str:
    session_id = (request.query_params.get("session_id") or "").strip()
    if session_id:
        return session_id

    header_val = (request.headers.get("x-session-id") or "").strip()
    if header_val:
        return header_val

    return ""


def _normalize_headers_map(raw_headers) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for key, value in raw_headers.items():
        result[str(key).lower()] = str(value)
    return result


class FixedWindowRateLimiter:
    def __init__(self) -> None:
        self._mem_lock = threading.Lock()
        self._mem_counters: Dict[str, Tuple[int, int]] = {}
        self._redis_client = None

        redis_url = _build_redis_url_from_env()
        try:
            if redis is not None:
                self._redis_client = redis.Redis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_connect_timeout=0.2,
                    socket_timeout=0.2,
                )
                self._redis_client.ping()
        except Exception:
            self._redis_client = None

    def _redis_check(self, key: str, limit: int, window_seconds: int) -> Tuple[bool, int]:
        assert self._redis_client is not None
        window_id = int(time.time() // window_seconds)
        redis_key = f"gw:rl:{key}:{window_id}"
        count = int(self._redis_client.incr(redis_key))
        if count == 1:
            self._redis_client.expire(redis_key, window_seconds)
        return count <= limit, count

    def _memory_check(self, key: str, limit: int, window_seconds: int) -> Tuple[bool, int]:
        now = int(time.time())
        window_id = now // window_seconds
        with self._mem_lock:
            existing = self._mem_counters.get(key)
            if not existing or existing[0] != window_id:
                self._mem_counters[key] = (window_id, 1)
                return True, 1

            count = existing[1] + 1
            self._mem_counters[key] = (window_id, count)
            return count <= limit, count

    def check(self, key: str, limit: int, window_seconds: int) -> Tuple[bool, int]:
        if self._redis_client is not None:
            try:
                return self._redis_check(key=key, limit=limit, window_seconds=window_seconds)
            except Exception:
                pass
        return self._memory_check(key=key, limit=limit, window_seconds=window_seconds)


def ensure_http_auth(settings: GatewaySecuritySettings, request: Request) -> str:
    if not settings.auth_enabled:
        return "none"

    if not settings.auth_tokens:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "网关鉴权已启用，但未配置令牌",
                "error_code": "GATEWAY_AUTH_CONFIG_ERROR",
            },
        )

    token, scheme = _extract_token_from_headers(_normalize_headers_map(request.headers))
    if not token or token not in settings.auth_tokens:
        raise HTTPException(
            status_code=401,
            detail={
                "message": "网关鉴权失败",
                "error_code": "GATEWAY_AUTH_FAILED",
            },
        )
    return scheme


def ensure_websocket_auth(settings: GatewaySecuritySettings, websocket: WebSocket) -> str:
    if not settings.auth_enabled:
        return "none"

    if not settings.auth_tokens:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "网关鉴权已启用，但未配置令牌",
                "error_code": "GATEWAY_AUTH_CONFIG_ERROR",
            },
        )

    token, scheme = _extract_token_from_headers(_normalize_headers_map(websocket.headers))
    if not token:
        token = (websocket.query_params.get("token") or "").strip()
        if token:
            scheme = "query-token"

    if not token or token not in settings.auth_tokens:
        raise HTTPException(
            status_code=401,
            detail={
                "message": "网关鉴权失败",
                "error_code": "GATEWAY_AUTH_FAILED",
            },
        )
    return scheme


def ensure_body_size_limit(settings: GatewaySecuritySettings, request: Request) -> None:
    max_bytes = settings.max_request_body_bytes
    if max_bytes <= 0:
        return

    content_length = (request.headers.get("content-length") or "").strip()
    if not content_length:
        return

    try:
        actual = int(content_length)
    except ValueError:
        return

    if actual > max_bytes:
        raise HTTPException(
            status_code=413,
            detail={
                "message": "请求体过大",
                "error_code": "GATEWAY_BODY_TOO_LARGE",
                "max_bytes": max_bytes,
                "actual_bytes": actual,
            },
        )


def ensure_http_rate_limit(settings: GatewaySecuritySettings, request: Request, limiter: FixedWindowRateLimiter) -> None:
    if not settings.rate_limit_enabled:
        return

    headers = _normalize_headers_map(request.headers)
    client_ip = _extract_client_ip_from_headers(
        headers=headers,
        fallback_ip=(request.client.host if request.client else "unknown"),
        trust_proxy_headers=settings.trust_proxy_headers,
    )

    allowed_ip, ip_count = limiter.check(
        key=f"ip:{client_ip}",
        limit=settings.rate_limit_ip_requests,
        window_seconds=settings.rate_limit_window_seconds,
    )
    if not allowed_ip:
        raise HTTPException(
            status_code=429,
            detail={
                "message": "请求过于频繁（IP限流）",
                "error_code": "GATEWAY_RATE_LIMIT_IP",
                "count": ip_count,
            },
        )

    session_id = _extract_session_id_for_guard(request)
    if not session_id:
        return

    session_hash = _hash_session_id(session_id)
    allowed_session, session_count = limiter.check(
        key=f"sid:{session_hash}",
        limit=settings.rate_limit_session_requests,
        window_seconds=settings.rate_limit_window_seconds,
    )
    if not allowed_session:
        raise HTTPException(
            status_code=429,
            detail={
                "message": "请求过于频繁（会话限流）",
                "error_code": "GATEWAY_RATE_LIMIT_SESSION",
                "count": session_count,
            },
        )


def ensure_ws_rate_limit(
    settings: GatewaySecuritySettings,
    websocket: WebSocket,
    limiter: FixedWindowRateLimiter,
    session_id: str,
) -> None:
    if not settings.rate_limit_enabled:
        return

    headers = _normalize_headers_map(websocket.headers)
    client_ip = _extract_client_ip_from_headers(
        headers=headers,
        fallback_ip=(websocket.client.host if websocket.client else "unknown"),
        trust_proxy_headers=settings.trust_proxy_headers,
    )
    allowed_ip, _ = limiter.check(
        key=f"ws:ip:{client_ip}",
        limit=settings.rate_limit_ip_requests,
        window_seconds=settings.rate_limit_window_seconds,
    )
    if not allowed_ip:
        raise HTTPException(
            status_code=429,
            detail={"message": "连接过于频繁（IP限流）", "error_code": "GATEWAY_RATE_LIMIT_IP"},
        )

    sid_hash = _hash_session_id(session_id)
    allowed_sid, _ = limiter.check(
        key=f"ws:sid:{sid_hash}",
        limit=settings.rate_limit_session_requests,
        window_seconds=settings.rate_limit_window_seconds,
    )
    if not allowed_sid:
        raise HTTPException(
            status_code=429,
            detail={"message": "连接过于频繁（会话限流）", "error_code": "GATEWAY_RATE_LIMIT_SESSION"},
        )


def build_http_audit_context(
    request: Request,
    auth_scheme: str,
    request_id: Optional[str] = None,
    trust_proxy_headers: bool = True,
) -> GatewayAuditContext:
    headers = _normalize_headers_map(request.headers)
    session_id = _extract_session_id_for_guard(request)
    client_ip = _extract_client_ip_from_headers(
        headers=headers,
        fallback_ip=(request.client.host if request.client else "unknown"),
        trust_proxy_headers=trust_proxy_headers,
    )
    return GatewayAuditContext(
        request_id=(request_id or uuid.uuid4().hex[:16]),
        client_ip=client_ip,
        method=request.method,
        path=request.url.path,
        session_hash=_hash_session_id(session_id),
        auth_scheme=auth_scheme,
    )


def log_http_gateway_audit(level: int, event: str, ctx: GatewayAuditContext, status_code: int, elapsed_ms: int) -> None:
    log_event(
        level,
        event,
        request_id=ctx.request_id,
        client_ip=ctx.client_ip,
        method=ctx.method,
        path=ctx.path,
        session_hash=ctx.session_hash,
        auth_scheme=ctx.auth_scheme,
        status_code=status_code,
        elapsed_ms=elapsed_ms,
    )
