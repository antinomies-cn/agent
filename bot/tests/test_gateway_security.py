from fastapi import HTTPException, Request

from app.core.config import get_gateway_security_settings
from app.core.gateway_security import (
    FixedWindowRateLimiter,
    build_http_audit_context,
    ensure_body_size_limit,
    ensure_http_auth,
    ensure_http_rate_limit,
)


def _make_request(headers=None, path="/chat", method="POST", query_string=""):
    raw_headers = []
    for key, value in (headers or {}).items():
        raw_headers.append((key.lower().encode("utf-8"), str(value).encode("utf-8")))
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": query_string.encode("utf-8"),
        "headers": raw_headers,
        "client": ("127.0.0.1", 12345),
        "scheme": "http",
        "server": ("testserver", 80),
    }
    return Request(scope)


def test_gateway_auth_accepts_bearer(monkeypatch):
    monkeypatch.setenv("GATEWAY_AUTH_ENABLED", "true")
    monkeypatch.setenv("GATEWAY_AUTH_TOKENS", "token-1,token-2")
    settings = get_gateway_security_settings()

    request = _make_request(headers={"Authorization": "Bearer token-2"})
    scheme = ensure_http_auth(settings, request)

    assert scheme == "bearer"


def test_gateway_auth_rejects_missing_token(monkeypatch):
    monkeypatch.setenv("GATEWAY_AUTH_ENABLED", "true")
    monkeypatch.setenv("GATEWAY_AUTH_TOKENS", "token-1")
    settings = get_gateway_security_settings()

    request = _make_request()
    try:
        ensure_http_auth(settings, request)
    except HTTPException as exc:
        assert exc.status_code == 401
        detail = exc.detail if isinstance(exc.detail, dict) else {}
        assert detail.get("error_code") == "GATEWAY_AUTH_FAILED"
    else:
        raise AssertionError("expected HTTPException when gateway token is missing")


def test_gateway_body_limit_rejects_large_content_length(monkeypatch):
    monkeypatch.setenv("GATEWAY_MAX_REQUEST_BODY", "1024")
    settings = get_gateway_security_settings()

    request = _make_request(headers={"Content-Length": "2048"})
    try:
        ensure_body_size_limit(settings, request)
    except HTTPException as exc:
        assert exc.status_code == 413
        detail = exc.detail if isinstance(exc.detail, dict) else {}
        assert detail.get("error_code") == "GATEWAY_BODY_TOO_LARGE"
    else:
        raise AssertionError("expected HTTPException 413 when request body is too large")


def test_gateway_rate_limit_by_session(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/15")
    monkeypatch.setenv("GATEWAY_RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("GATEWAY_RATE_LIMIT_IP_REQUESTS", "10")
    monkeypatch.setenv("GATEWAY_RATE_LIMIT_SESSION_REQUESTS", "1")
    monkeypatch.setenv("GATEWAY_RATE_LIMIT_WINDOW_SECONDS", "60")
    settings = get_gateway_security_settings()

    limiter = FixedWindowRateLimiter()
    request = _make_request(query_string="session_id=s-1")

    ensure_http_rate_limit(settings, request, limiter)

    try:
        ensure_http_rate_limit(settings, request, limiter)
    except HTTPException as exc:
        assert exc.status_code == 429
        detail = exc.detail if isinstance(exc.detail, dict) else {}
        assert detail.get("error_code") == "GATEWAY_RATE_LIMIT_SESSION"
    else:
        raise AssertionError("expected HTTPException 429 when session rate limit is exceeded")


def test_audit_context_respects_proxy_trust(monkeypatch):
    request = _make_request(headers={"X-Forwarded-For": "203.0.113.10"}, query_string="session_id=safe-sid")

    ctx = build_http_audit_context(request=request, auth_scheme="bearer", trust_proxy_headers=True)

    assert ctx.client_ip == "203.0.113.10"
    assert ctx.session_hash


def test_audit_context_ignores_proxy_when_not_trusted(monkeypatch):
    request = _make_request(headers={"X-Forwarded-For": "203.0.113.10"}, query_string="session_id=safe-sid")

    ctx = build_http_audit_context(request=request, auth_scheme="none", trust_proxy_headers=False)

    assert ctx.client_ip == "127.0.0.1"
