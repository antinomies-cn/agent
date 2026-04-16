from app.core.config import (
    get_embeddings_gateway_settings,
    get_rerank_gateway_settings,
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
