import importlib
import sys


# 某些测试会向 sys.modules 注入 app.services.qdrant_service 的桩模块，
# 这里显式移除后再导入真实实现，避免跨测试污染。
sys.modules.pop("app.services.qdrant_service", None)
qdrant_service = importlib.import_module("app.services.qdrant_service")


class _DummyClient:
    def __init__(self, exists: bool):
        self._exists = exists
        self.deleted_calls = 0
        self.created_calls = 0

    def collection_exists(self, _collection_name):
        return self._exists

    def delete_collection(self, collection_name=None):
        self.deleted_calls += 1
        self._exists = False

    def create_collection(self, collection_name=None, vectors_config=None):
        self.created_calls += 1
        self._exists = True


def test_init_qdrant_force_recreate_invalidates_cache(monkeypatch):
    client = _DummyClient(exists=True)
    invalidation_calls = {"count": 0}

    monkeypatch.setattr(qdrant_service, "_get_qdrant_client", lambda: client)
    monkeypatch.setattr(qdrant_service, "_resolve_vector_size", lambda default_value=1024: 384)
    monkeypatch.setattr(qdrant_service, "_resolve_distance", lambda: qdrant_service.rest.Distance.COSINE)

    def _fake_invalidate():
        invalidation_calls["count"] += 1
        return True

    monkeypatch.setattr(qdrant_service, "_invalidate_vector_retriever_cache", _fake_invalidate)

    result = qdrant_service.init_qdrant_collection("demo", force_recreate=True)

    assert result["ok"] is True
    assert result["deleted"] is True
    assert result["created"] is True
    assert result["retriever_cache_reset_attempted"] is True
    assert result["retriever_cache_had_cache"] is True
    assert invalidation_calls["count"] == 1
    assert client.deleted_calls == 1
    assert client.created_calls == 1


def test_init_qdrant_without_recreate_does_not_invalidate_cache(monkeypatch):
    client = _DummyClient(exists=True)
    invalidation_calls = {"count": 0}

    monkeypatch.setattr(qdrant_service, "_get_qdrant_client", lambda: client)
    monkeypatch.setattr(qdrant_service, "_resolve_vector_size", lambda default_value=1024: 384)
    monkeypatch.setattr(qdrant_service, "_resolve_distance", lambda: qdrant_service.rest.Distance.COSINE)

    def _fake_invalidate():
        invalidation_calls["count"] += 1
        return True

    monkeypatch.setattr(qdrant_service, "_invalidate_vector_retriever_cache", _fake_invalidate)

    result = qdrant_service.init_qdrant_collection("demo", force_recreate=False)

    assert result["ok"] is True
    assert result["deleted"] is False
    assert result["created"] is False
    assert result["retriever_cache_reset_attempted"] is False
    assert result["retriever_cache_had_cache"] is False
    assert invalidation_calls["count"] == 0
    assert client.deleted_calls == 0
    assert client.created_calls == 0


def test_init_qdrant_force_recreate_on_missing_collection_invalidates_cache(monkeypatch):
    client = _DummyClient(exists=False)
    invalidation_calls = {"count": 0}

    monkeypatch.setattr(qdrant_service, "_get_qdrant_client", lambda: client)
    monkeypatch.setattr(qdrant_service, "_resolve_vector_size", lambda default_value=1024: 384)
    monkeypatch.setattr(qdrant_service, "_resolve_distance", lambda: qdrant_service.rest.Distance.COSINE)

    def _fake_invalidate():
        invalidation_calls["count"] += 1
        return True

    monkeypatch.setattr(qdrant_service, "_invalidate_vector_retriever_cache", _fake_invalidate)

    result = qdrant_service.recreate_qdrant_collection("demo")

    assert result["ok"] is True
    assert result["deleted"] is False
    assert result["created"] is True
    assert result["retriever_cache_reset_attempted"] is True
    assert result["retriever_cache_had_cache"] is True
    assert invalidation_calls["count"] == 1
    assert client.deleted_calls == 0
    assert client.created_calls == 1


def test_init_qdrant_create_without_force_still_attempts_reset(monkeypatch):
    client = _DummyClient(exists=False)
    invalidation_calls = {"count": 0}

    monkeypatch.setattr(qdrant_service, "_get_qdrant_client", lambda: client)
    monkeypatch.setattr(qdrant_service, "_resolve_vector_size", lambda default_value=1024: 384)
    monkeypatch.setattr(qdrant_service, "_resolve_distance", lambda: qdrant_service.rest.Distance.COSINE)

    def _fake_invalidate():
        invalidation_calls["count"] += 1
        return False

    monkeypatch.setattr(qdrant_service, "_invalidate_vector_retriever_cache", _fake_invalidate)

    result = qdrant_service.init_qdrant_collection("demo", force_recreate=False)

    assert result["ok"] is True
    assert result["deleted"] is False
    assert result["created"] is True
    assert result["retriever_cache_reset_attempted"] is True
    assert result["retriever_cache_had_cache"] is False
    assert invalidation_calls["count"] == 1
    assert client.deleted_calls == 0
    assert client.created_calls == 1
