import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _stub_module(module_name: str, **attrs):
    module = types.ModuleType(module_name)
    if module_name in {"langchain", "langchain_community"}:
        module.__path__ = []
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[module_name] = module
    return module


if "langchain.text_splitter" not in sys.modules:
    _stub_module("langchain")

    _stub_module("langchain.agents", tool=lambda fn: fn)

    class _DummySplitter:
        def __init__(self, *args, **kwargs):
            pass

        def split_documents(self, documents):
            return documents

    _stub_module("langchain.text_splitter", RecursiveCharacterTextSplitter=_DummySplitter)

if "langchain_community.document_loaders" not in sys.modules:
    _stub_module("langchain_community")

    class _DummyWebBaseLoader:
        def __init__(self, url, **kwargs):
            self.url = url
            self.kwargs = kwargs

        def load(self):
            return []

    _stub_module("langchain_community.document_loaders", WebBaseLoader=_DummyWebBaseLoader)

if "langchain_community.utilities" not in sys.modules:
    class _DummySerpAPIWrapper:
        def run(self, *args, **kwargs):
            return ""

    _stub_module("langchain_community.utilities", SerpAPIWrapper=_DummySerpAPIWrapper)

if "langchain_community.vectorstores" not in sys.modules:
    class _DummyQdrant:
        @staticmethod
        def from_documents(*args, **kwargs):
            return object()

    _stub_module("langchain_community.vectorstores", Qdrant=_DummyQdrant)

if "langchain_openai" not in sys.modules:
    _stub_module("langchain_openai", OpenAIEmbeddings=lambda *args, **kwargs: object())

if "qdrant_client" not in sys.modules:
    class _DummyDistance:
        COSINE = types.SimpleNamespace(name="COSINE")
        DOT = types.SimpleNamespace(name="DOT")
        EUCLID = types.SimpleNamespace(name="EUCLID")

    class _DummyVectorParams:
        def __init__(self, size, distance):
            self.size = size
            self.distance = distance

    class _DummyQdrantClient:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.collections = {}

        def collection_exists(self, name):
            return name in self.collections

        def create_collection(self, collection_name, vectors_config):
            self.collections[collection_name] = getattr(vectors_config, "size", 384)

        def delete_collection(self, collection_name):
            self.collections.pop(collection_name, None)

        def get_collections(self):
            class _Info:
                def __init__(self, names):
                    self.collections = [types.SimpleNamespace(name=n) for n in names]

            return _Info(self.collections.keys())

        def get_collection(self, collection_name):
            size = self.collections.get(collection_name, 384)

            class _CollectionInfo:
                def __init__(self, dim):
                    self.config = types.SimpleNamespace(
                        params=types.SimpleNamespace(vectors=types.SimpleNamespace(size=dim))
                    )

            return _CollectionInfo(size)

        def count(self, collection_name, exact=False):
            return types.SimpleNamespace(count=0)

    _stub_module("qdrant_client", QdrantClient=_DummyQdrantClient)
    _stub_module("qdrant_client.http", models=types.SimpleNamespace(Distance=_DummyDistance, VectorParams=_DummyVectorParams))

if "app.services.master_service" not in sys.modules:
    class _DummyMaster:
        def __init__(self):
            pass

        def check_llm_health(self):
            return True

        def get_memory_status(self, session_id):
            return {"session_id": session_id, "message_count": 0}

        def run(self, query, session_id):
            return {"output": "ok"}

        async def run_async(self, query, session_id):
            return {"output": "ok"}

    _stub_module("app.services.master_service", Master=_DummyMaster)

if "app.services.qdrant_service" not in sys.modules:
    _stub_module(
        "app.services.qdrant_service",
        init_qdrant_collection=lambda collection_name=None, force_recreate=False: {"ok": True},
        recreate_qdrant_collection=lambda collection_name=None: {"ok": True},
        qdrant_health=lambda: {"ok": True},
        qdrant_list_collections=lambda: {"ok": True, "collections": []},
        qdrant_repo_status=lambda: {"ok": True, "collections": []},
    )

from app import main


class _DummyQdrantClient:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.collections = {}

    def collection_exists(self, name):
        return name in self.collections

    def create_collection(self, collection_name, vectors_config):
        self.collections[collection_name] = getattr(vectors_config, "size", 384)

    def delete_collection(self, collection_name):
        self.collections.pop(collection_name, None)

    def get_collections(self):
        class _Info:
            def __init__(self, names):
                self.collections = [types.SimpleNamespace(name=n) for n in names]

        return _Info(self.collections.keys())

    def get_collection(self, collection_name):
        size = self.collections.get(collection_name, 384)

        class _CollectionInfo:
            def __init__(self, dim):
                self.config = types.SimpleNamespace(
                    params=types.SimpleNamespace(vectors=types.SimpleNamespace(size=dim))
                )

        return _CollectionInfo(size)

    def count(self, collection_name, exact=False):
        return types.SimpleNamespace(count=0)


class _FakeChunk:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


class _DummyQdrantStore:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def add_documents(self, documents):
        return documents


def _patch_external_deps(monkeypatch):
    monkeypatch.setattr(main, "QdrantClient", _DummyQdrantClient)
    monkeypatch.setattr(main, "_build_embeddings_client", lambda *args, **kwargs: object())
    monkeypatch.setattr(main, "Qdrant", _DummyQdrantStore)


def _enable_add_urls_write(monkeypatch):
    monkeypatch.setenv("ENV", "dev")
    monkeypatch.setenv("ADD_URLS_WRITE_ENABLED", "true")


def test_request_model_none_empty_missing_semantics():
    payload = main.AddUrlsRequest()
    assert payload.url is None
    assert payload.urls == []
    assert payload.separators is None

    payload2 = main.AddUrlsRequest(url=" ", urls=["", "  ", " https://example.com "])
    assert main._normalize_urls(payload2) == ["https://example.com"]

    cfg_none = main._build_chunking_config("custom", separators=None)
    cfg_empty = main._build_chunking_config("custom", separators=[])
    assert cfg_none["separators"] == cfg_empty["separators"]


def test_add_urls_response_contract(monkeypatch):
    _patch_external_deps(monkeypatch)
    _enable_add_urls_write(monkeypatch)
    monkeypatch.setattr(
        main,
        "_collect_chunks_from_urls",
        lambda clean_url_list, chunk_cfg, strategy: (
            [object(), object(), object()],
            [{"url": "https://bad.example", "error": "timeout"}],
        ),
    )

    client = TestClient(main.app)
    resp = client.post(
        "/add_urls",
        json={
            "urls": ["https://ok.example"],
            "chunk_strategy": "balanced",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["response"] == "URLs added!"
    assert body["mode"] in {"remote", "local"}
    assert isinstance(body["source_urls"], int)
    assert isinstance(body["chunks"], int)
    assert isinstance(body["failed_urls"], list)
    assert body["failed_urls"]
    assert "explanation" in body["failed_urls"][0]
    assert isinstance(body["failed_urls"][0]["explanation"], str)
    assert body["failed_urls"][0]["explanation"]
    assert isinstance(body["chunk_config"], dict)
    assert set(body["chunk_config"].keys()) == {"chunk_size", "chunk_overlap", "separators"}


def test_add_urls_dry_run_response_contract(monkeypatch):
    _patch_external_deps(monkeypatch)
    monkeypatch.setattr(
        main,
        "_collect_chunks_from_urls",
        lambda clean_url_list, chunk_cfg, strategy: (
            [
                _FakeChunk("first chunk content", {"source_url": "https://ok.example", "chunk_index": 0}),
                _FakeChunk("second chunk content", {"source_url": "https://ok.example", "chunk_index": 1}),
            ],
            [],
        ),
    )

    client = TestClient(main.app)
    resp = client.post(
        "/add_urls/dry_run",
        json={
            "url": "https://ok.example",
            "chunk_strategy": "faq",
            "preview_limit": 1,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["response"] == "Dry run completed"
    assert isinstance(body["failed_urls"], list)
    assert body["failed_urls"] == []
    assert isinstance(body["chunk_preview"], list)
    assert len(body["chunk_preview"]) == 1
    assert set(body["chunk_preview"][0].keys()) == {
        "source_url",
        "chunk_index",
        "content_length",
        "content_preview",
    }


def test_add_urls_accepts_query_params(monkeypatch):
    _patch_external_deps(monkeypatch)
    _enable_add_urls_write(monkeypatch)
    monkeypatch.setattr(
        main,
        "_collect_chunks_from_urls",
        lambda clean_url_list, chunk_cfg, strategy: ([object()], []),
    )

    client = TestClient(main.app)
    resp = client.post(
        "/add_urls",
        params={
            "url": "https://ok.example",
            "chunk_strategy": "balanced",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["source_urls"] == 1
    assert body["chunk_strategy"] == "balanced"


def test_add_urls_dry_run_accepts_query_params(monkeypatch):
    _patch_external_deps(monkeypatch)
    monkeypatch.setattr(
        main,
        "_collect_chunks_from_urls",
        lambda clean_url_list, chunk_cfg, strategy: (
            [_FakeChunk("first chunk content", {"source_url": "https://ok.example", "chunk_index": 0})],
            [],
        ),
    )

    client = TestClient(main.app)
    resp = client.post(
        "/add_urls/dry_run",
        params={
            "url": "https://ok.example",
            "chunk_strategy": "faq",
            "preview_limit": 1,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["source_urls"] == 1
    assert body["chunk_strategy"] == "faq"
    assert len(body["chunk_preview"]) == 1


def test_add_urls_retries_transient_loader_failure(monkeypatch):
    _patch_external_deps(monkeypatch)
    _enable_add_urls_write(monkeypatch)
    monkeypatch.setenv("ADD_URLS_FETCH_RETRY_COUNT", "2")
    monkeypatch.setenv("ADD_URLS_FETCH_TIMEOUT_SECONDS", "1")

    attempts = {"count": 0}

    class _FlakyWebBaseLoader:
        def __init__(self, url, **kwargs):
            self.url = url
            self.kwargs = kwargs

        def load(self):
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise RuntimeError("temporary fetch error")
            return [_FakeChunk("retry succeeded", {"source_url": self.url})]

    monkeypatch.setattr(main, "WebBaseLoader", _FlakyWebBaseLoader)

    chunks, failed_urls = main._collect_chunks_from_urls(
        ["https://ok.example"],
        main._build_chunking_config("balanced"),
        "balanced",
    )

    assert attempts["count"] == 3
    assert len(failed_urls) == 0
    assert len(chunks) == 1
    assert chunks[0].page_content == "retry succeeded"


def test_add_urls_truncates_long_content_before_chunking(monkeypatch):
    _patch_external_deps(monkeypatch)
    _enable_add_urls_write(monkeypatch)
    monkeypatch.setenv("ADD_URLS_MAX_CONTENT_CHARS", "10")
    monkeypatch.setenv("ADD_URLS_FETCH_RETRY_COUNT", "0")

    class _LongContentWebBaseLoader:
        def __init__(self, url, **kwargs):
            self.url = url
            self.kwargs = kwargs

        def load(self):
            return [_FakeChunk("0123456789abcdef", {"source_url": self.url})]

    monkeypatch.setattr(main, "WebBaseLoader", _LongContentWebBaseLoader)

    chunks, failed_urls = main._collect_chunks_from_urls(
        ["https://ok.example"],
        main._build_chunking_config("balanced"),
        "balanced",
    )

    assert len(failed_urls) == 0
    assert len(chunks) == 1
    assert chunks[0].page_content == "0123456789"
    assert chunks[0].metadata["content_truncated"] is True
    assert chunks[0].metadata["content_original_length"] == 16
    assert chunks[0].metadata["content_max_length"] == 10


def test_add_urls_blocks_private_loopback_and_link_local_urls(monkeypatch):
    _patch_external_deps(monkeypatch)
    _enable_add_urls_write(monkeypatch)

    client = TestClient(main.app)
    resp = client.post(
        "/add_urls",
        json={
            "urls": ["http://127.0.0.1:8000", "http://169.254.1.2", "http://192.168.1.8"],
            "chunk_strategy": "balanced",
        },
    )

    assert resp.status_code == 400
    body = resp.json()
    detail = body["detail"]
    assert detail["message"] == "URL全部被安全策略拦截"
    assert len(detail["failed_urls"]) == 3
    codes = {item["code"] for item in detail["failed_urls"]}
    assert "BLOCKED_LOOPBACK" in codes
    assert "BLOCKED_LINK_LOCAL" in codes
    assert "BLOCKED_PRIVATE_IP" in codes
    assert all(item.get("explanation") for item in detail["failed_urls"])


def test_add_urls_blocks_internal_hostname_with_machine_code(monkeypatch):
    _patch_external_deps(monkeypatch)
    _enable_add_urls_write(monkeypatch)

    client = TestClient(main.app)
    resp = client.post(
        "/add_urls",
        json={
            "url": "http://redis:6379",
            "chunk_strategy": "balanced",
        },
    )

    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert detail["failed_urls"][0]["code"] == "BLOCKED_INTERNAL_HOST"
    assert detail["failed_urls"][0]["explanation"]


def test_add_urls_blocks_unsupported_scheme_with_machine_code(monkeypatch):
    _patch_external_deps(monkeypatch)
    _enable_add_urls_write(monkeypatch)

    client = TestClient(main.app)
    resp = client.post(
        "/add_urls",
        json={
            "url": "ftp://example.com/resource.txt",
            "chunk_strategy": "balanced",
        },
    )

    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert detail["failed_urls"][0]["code"] == "UNSUPPORTED_SCHEME"
    assert detail["failed_urls"][0]["explanation"]


def test_add_urls_blocks_missing_host_with_machine_code(monkeypatch):
    _patch_external_deps(monkeypatch)
    _enable_add_urls_write(monkeypatch)

    client = TestClient(main.app)
    resp = client.post(
        "/add_urls",
        json={
            "url": "http:///path-only",
            "chunk_strategy": "balanced",
        },
    )

    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert detail["failed_urls"][0]["code"] == "MISSING_HOST"
    assert detail["failed_urls"][0]["explanation"]


def test_add_urls_prod_write_disabled_by_default(monkeypatch):
    _patch_external_deps(monkeypatch)
    monkeypatch.setenv("ENV", "prod")
    monkeypatch.delenv("ADD_URLS_WRITE_ENABLED", raising=False)

    client = TestClient(main.app)
    resp = client.post(
        "/add_urls",
        json={
            "url": "https://ok.example",
            "chunk_strategy": "balanced",
        },
    )

    assert resp.status_code == 404
    detail = resp.json()
    assert detail["detail"] == "Not Found"
    assert detail["error_code"] == "ROUTE_NOT_EXPOSED"
    assert detail["explanation"]


def test_add_urls_error_code_enum_contract():
    expected = {
        "FETCH_ERROR",
        "BLOCKED_URL",
        "INVALID_URL",
        "UNSUPPORTED_SCHEME",
        "MISSING_HOST",
        "INVALID_IP",
        "BLOCKED_PRIVATE_IP",
        "BLOCKED_LOOPBACK",
        "BLOCKED_LINK_LOCAL",
        "BLOCKED_INTERNAL_HOST",
    }
    actual = {item.value for item in main.AddUrlsErrorCode}
    assert actual == expected


def test_http_exception_string_detail_is_normalized(monkeypatch):
    monkeypatch.setenv("ENV", "dev")
    client = TestClient(main.app)

    resp = client.post("/add_urls", json={"chunk_strategy": "balanced"})
    assert resp.status_code == 400
    body = resp.json()
    assert body["detail"] == "请提供url或urls参数"
    assert body["error_code"] == "HTTP_400"
    assert body["explanation"]


def test_request_validation_error_is_normalized():
    client = TestClient(main.app)

    # 缺少 query/session_id，触发 FastAPI 的 422 校验异常。
    resp = client.post("/chat")
    assert resp.status_code == 422
    body = resp.json()
    assert body["error_code"] == "REQUEST_VALIDATION_ERROR"
    assert body["explanation"]
    assert isinstance(body["errors"], list)


def test_prod_only_exposes_chat_endpoint(monkeypatch):
    monkeypatch.setenv("ENV", "prod")

    client = TestClient(main.app)

    chat_resp = client.post(
        "/chat",
        params={
            "query": "hello",
            "session_id": "s1",
        },
    )
    assert chat_resp.status_code == 200

    health_resp = client.get("/health/live")
    assert health_resp.status_code == 404
    health_detail = health_resp.json()
    assert health_detail["error_code"] == "ROUTE_NOT_EXPOSED"
    assert health_detail["explanation"]

    add_urls_resp = client.post(
        "/add_urls",
        json={
            "url": "https://ok.example",
            "chunk_strategy": "balanced",
        },
    )
    assert add_urls_resp.status_code == 404
    add_urls_detail = add_urls_resp.json()
    assert add_urls_detail["error_code"] == "ROUTE_NOT_EXPOSED"
    assert add_urls_detail["explanation"]


def test_is_prod_runtime_uses_env_value(monkeypatch):
    monkeypatch.setenv("ENV", "PrOd")
    assert main._is_prod_runtime() is True

    monkeypatch.setenv("ENV", "dev")
    assert main._is_prod_runtime() is False


def test_ws_is_rejected_in_prod(monkeypatch):
    monkeypatch.setenv("ENV", "prod")
    client = TestClient(main.app)

    with pytest.raises(Exception):
        with client.websocket_connect("/ws?session_id=s1"):
            pass


def test_chat_returns_error_explanation_when_runtime_fails(monkeypatch):
    def _raise_error(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(main.master, "run", _raise_error)
    client = TestClient(main.app)
    resp = client.post("/chat", params={"query": "hi", "session_id": "s1"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 500
    assert body["error_code"] == "CHAT_RUNTIME_ERROR"
    assert body["explanation"]


def test_qdrant_init_returns_error_explanation_on_failure(monkeypatch):
    def _raise_error(*args, **kwargs):
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(main, "init_qdrant_collection", _raise_error)
    client = TestClient(main.app)
    resp = client.post("/qdrant/init")

    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 500
    assert body["error_code"] == "QDRANT_INIT_ERROR"
    assert body["explanation"]


def test_embedding_config_endpoint_returns_effective_config(monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_MODEL", "bge-m3")
    monkeypatch.setenv("EMBEDDINGS_DIMENSION", "1024")
    monkeypatch.setenv("QDRANT_COLLECTION", "web_collection")
    monkeypatch.setenv("QDRANT_DISTANCE", "cosine")

    client = TestClient(main.app)
    resp = client.get("/embedding/config")

    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "bge-m3"
    assert body["dimensions"] == 1024
    assert body["dimension_source"] == "env"
    assert body["collection"] == "web_collection"
    assert body["qdrant_distance"] == "cosine"


def test_rerank_config_endpoint_returns_direct_upstream_state(monkeypatch):
    monkeypatch.setenv("RERANK_ENABLED", "true")
    monkeypatch.setenv("RERANK_DIRECT_UPSTREAM", "true")
    monkeypatch.setenv("RERANK_MODEL", "bge-reranker")
    monkeypatch.setenv("RERANK_UPSTREAM_MODEL", "bge-reranker-v2-m3")
    monkeypatch.setenv("RERANK_API_BASE", "https://api.edgefn.net/v1")
    monkeypatch.setenv("RERANK_TIMEOUT", "15")
    monkeypatch.setenv("RERANK_TOP_N", "1")
    monkeypatch.setenv("RERANK_STARTUP_STRICT", "false")

    client = TestClient(main.app)
    resp = client.get("/rerank/config")

    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is True
    assert body["direct_upstream"] is True
    assert body["model"] == "bge-reranker"
    assert body["upstream_model"] == "bge-reranker-v2-m3"
    assert body["upstream_base"] == "https://api.edgefn.net/v1"
    assert body["timeout_seconds"] == 15.0
    assert body["top_n"] == 1
    assert body["startup_strict"] is False


def test_config_health_endpoint_returns_summary(monkeypatch):
    monkeypatch.setenv("ENV", "dev")
    monkeypatch.setenv("API_HOST", "127.0.0.1")
    monkeypatch.setenv("API_PORT", "8000")
    monkeypatch.setenv("QDRANT_COLLECTION", "web_collection")
    monkeypatch.setenv("QDRANT_DISTANCE", "cosine")

    client = TestClient(main.app)
    resp = client.get("/config/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 200
    assert "data" in body
    assert body["data"]["ok"] is True
    assert "highlights" in body["data"]
    assert body["data"]["highlights"]["env"] == "dev"
    assert body["data"]["highlights"]["qdrant_collection"] == "web_collection"


def test_cors_preflight_passes_when_gateway_auth_enabled(monkeypatch):
    monkeypatch.setenv("GATEWAY_AUTH_ENABLED", "true")
    monkeypatch.setenv("GATEWAY_AUTH_TOKENS", "token-a")
    monkeypatch.setattr(main, "gateway_security_settings", main.get_gateway_security_settings())

    client = TestClient(main.app)
    resp = client.options(
        "/chat",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert resp.status_code in {200, 204}
    assert "access-control-allow-origin" in {k.lower() for k in resp.headers.keys()}
