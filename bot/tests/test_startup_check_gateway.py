import requests

from app import startup_check


class _FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"http {self.status_code}", response=self)


def test_startup_check_non_strict_rerank_missing_config(monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_STARTUP_CHECK", "true")
    monkeypatch.setenv("OPENAI_API_BASE", "http://litellm:4000")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("EMBEDDINGS_MODEL", "bge-m3")
    monkeypatch.setenv("EMBEDDINGS_DIMENSION", "3")
    monkeypatch.setenv("RERANK_ENABLED", "true")
    monkeypatch.setenv("RERANK_DIRECT_UPSTREAM", "true")
    monkeypatch.setenv("RERANK_STARTUP_STRICT", "false")
    monkeypatch.setenv("RERANK_API_BASE", "")
    monkeypatch.setenv("RERANK_API_KEY", "")

    def _fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResponse(200, {"data": [{"embedding": [0.1, 0.2, 0.3]}]})

    monkeypatch.setattr(startup_check.requests, "post", _fake_post, raising=False)

    assert startup_check.main() == 0


def test_startup_check_strict_rerank_missing_config(monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_STARTUP_CHECK", "true")
    monkeypatch.setenv("OPENAI_API_BASE", "http://litellm:4000")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("EMBEDDINGS_MODEL", "bge-m3")
    monkeypatch.setenv("EMBEDDINGS_DIMENSION", "3")
    monkeypatch.setenv("RERANK_ENABLED", "true")
    monkeypatch.setenv("RERANK_DIRECT_UPSTREAM", "true")
    monkeypatch.setenv("RERANK_STARTUP_STRICT", "true")
    monkeypatch.setenv("RERANK_API_BASE", "")
    monkeypatch.setenv("RERANK_API_KEY", "")

    def _fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResponse(200, {"data": [{"embedding": [0.1, 0.2, 0.3]}]})

    monkeypatch.setattr(startup_check.requests, "post", _fake_post, raising=False)

    assert startup_check.main() == 1


def test_startup_check_embedding_dimension_mismatch(monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_STARTUP_CHECK", "true")
    monkeypatch.setenv("OPENAI_API_BASE", "http://litellm:4000")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("EMBEDDINGS_MODEL", "bge-m3")
    monkeypatch.setenv("EMBEDDINGS_DIMENSION", "5")
    monkeypatch.setenv("RERANK_ENABLED", "false")

    def _fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResponse(200, {"data": [{"embedding": [0.1, 0.2, 0.3]}]})

    monkeypatch.setattr(startup_check.requests, "post", _fake_post, raising=False)

    assert startup_check.main() == 1


def test_startup_check_sanitizes_legacy_embedding_model_path(monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_STARTUP_CHECK", "true")
    monkeypatch.setenv("OPENAI_API_BASE", "http://litellm:4000")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("EMBEDDINGS_MODEL", r"D:\legacy\models\bge-m3")
    monkeypatch.setenv("EMBEDDINGS_DIMENSION", "3")
    monkeypatch.setenv("RERANK_ENABLED", "false")

    seen_models = []

    def _fake_post(url, headers=None, json=None, timeout=None):
        seen_models.append((json or {}).get("model"))
        return _FakeResponse(200, {"data": [{"embedding": [0.1, 0.2, 0.3]}]})

    monkeypatch.setattr(startup_check.requests, "post", _fake_post, raising=False)

    assert startup_check.main() == 0
    assert seen_models and seen_models[0] == "bge-m3"
