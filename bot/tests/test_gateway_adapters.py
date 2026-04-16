import requests

from app.core.litellm_adapters import LiteLLMEmbeddings, rerank_texts_with_litellm


class _FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"http {self.status_code}", response=self)


def test_embeddings_dimension_fallback(monkeypatch):
    calls = []

    def _fake_post(url, headers=None, json=None, timeout=None):
        calls.append({"url": url, "json": dict(json or {})})
        if len(calls) == 1:
            return _FakeResponse(400, {"error": "unsupported dimensions"})
        return _FakeResponse(200, {"data": [{"embedding": [0.1, 0.2, 0.3]}]})

    monkeypatch.setattr("app.core.gateway_http.requests.post", _fake_post)

    client = LiteLLMEmbeddings(
        model="bge-m3",
        base_url="http://mock-gateway:4000",
        api_key="test-key",
        dimensions=1024,
        timeout=5,
        retry_count=0,
    )

    vector = client.embed_query("hello")

    assert vector == [0.1, 0.2, 0.3]
    assert len(calls) == 2
    assert calls[0]["json"].get("dimensions") == 1024
    assert "dimensions" not in calls[1]["json"]


def test_rerank_endpoint_fallback(monkeypatch):
    monkeypatch.setenv("RERANK_DIRECT_UPSTREAM", "true")
    monkeypatch.setenv("RERANK_API_BASE", "http://mock-rerank")
    monkeypatch.setenv("RERANK_API_KEY", "rerank-key")
    monkeypatch.setenv("RERANK_MODEL", "bge-reranker")
    monkeypatch.delenv("RERANK_UPSTREAM_MODEL", raising=False)

    called_urls = []

    def _fake_post(url, headers=None, json=None, timeout=None):
        called_urls.append(url)
        if url.endswith("/v1/rerank"):
            return _FakeResponse(
                200,
                {
                    "results": [
                        {"index": 1, "relevance_score": 0.4},
                        {"index": 0, "relevance_score": 0.9},
                    ]
                },
            )
        if url.endswith("/rerank"):
            return _FakeResponse(404, {"error": "not found"})
        return _FakeResponse(500, {"error": "unexpected endpoint"})

    monkeypatch.setattr("app.core.gateway_http.requests.post", _fake_post)

    results = rerank_texts_with_litellm("q", ["doc0", "doc1"])

    assert called_urls[0].endswith("/rerank")
    assert called_urls[1].endswith("/v1/rerank")
    assert results[0]["index"] == 0
    assert results[0]["relevance_score"] == 0.9


def test_rerank_missing_base_raises(monkeypatch):
    monkeypatch.setenv("RERANK_DIRECT_UPSTREAM", "true")
    monkeypatch.delenv("RERANK_API_BASE", raising=False)
    monkeypatch.setenv("RERANK_API_KEY", "rerank-key")

    try:
        rerank_texts_with_litellm("q", ["doc"])
    except RuntimeError as exc:
        assert "未配置" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when RERANK_API_BASE is missing")
