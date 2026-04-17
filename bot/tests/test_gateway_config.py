from app.core.config import (
    build_config_health_summary,
    get_embeddings_gateway_settings,
    get_redis_settings,
    get_rerank_gateway_settings,
    get_server_settings,
    normalize_openai_base_url,
)


def test_embeddings_base_fallback_to_openai_api_base(monkeypatch):
    monkeypatch.delenv("OPENAI_EMBEDDINGS_API_BASE", raising=False)
    monkeypatch.setenv("OPENAI_API_BASE", "http://litellm:4000/v1")

    cfg = get_embeddings_gateway_settings()

    assert cfg.base_url == "http://litellm:4000"


def test_rerank_config_parse_defaults_and_switch(monkeypatch):
    monkeypatch.setenv("OPENAI_API_BASE", "http://litellm:4000/")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("RERANK_DIRECT_UPSTREAM", "false")
    monkeypatch.setenv("RERANK_TIMEOUT", "not-a-number")
    monkeypatch.setenv("RERANK_RETRY_COUNT", "-5")
    monkeypatch.setenv("RERANK_TOP_N", "3")

    cfg = get_rerank_gateway_settings()

    assert cfg.direct_upstream is False
    assert cfg.base_url == "http://litellm:4000"
    assert cfg.api_key == "openai-key"
    assert cfg.timeout_seconds == 15.0
    assert cfg.retry_count == 0
    assert cfg.top_n == 3
    assert cfg.request_model == cfg.model


def test_normalize_openai_base_url_strip_v1_and_trailing_slash():
    assert normalize_openai_base_url("http://x:4000/v1/") == "http://x:4000"


def test_server_settings_invalid_port_fallback(monkeypatch):
    monkeypatch.setenv("API_PORT", "invalid")

    cfg = get_server_settings()

    assert cfg.port == 8000


def test_config_health_summary_warns_loopback_in_prod(monkeypatch):
    monkeypatch.setenv("ENV", "prod")
    monkeypatch.setenv("API_HOST", "127.0.0.1")
    monkeypatch.setenv("API_PORT", "8000")

    summary = build_config_health_summary()

    assert summary["ok"] is True
    assert any("回环地址" in item for item in summary["warnings"])


def test_redis_settings_build_url_from_parts(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setenv("REDIS_HOST", "redis.internal")
    monkeypatch.setenv("REDIS_PORT", "6380")
    monkeypatch.setenv("REDIS_DB", "2")
    monkeypatch.setenv("REDIS_PASSWORD", "secret")

    cfg = get_redis_settings()

    assert cfg.url == "redis://:secret@redis.internal:6380/2"


def test_redis_settings_prefers_explicit_url(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://custom-host:6379/5")

    cfg = get_redis_settings()

    assert cfg.url == "redis://custom-host:6379/5"
