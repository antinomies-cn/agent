import sys
import types
from pathlib import Path

from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _stub_module(module_name: str, **attrs):
    module = types.ModuleType(module_name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[module_name] = module
    return module


if "langchain.text_splitter" not in sys.modules:
    _stub_module("langchain")

    class _DummySplitter:
        def __init__(self, *args, **kwargs):
            pass

        def split_documents(self, documents):
            return documents

    _stub_module("langchain.text_splitter", RecursiveCharacterTextSplitter=_DummySplitter)

if "langchain_community.document_loaders" not in sys.modules:
    _stub_module("langchain_community")

    class _DummyWebBaseLoader:
        def __init__(self, url):
            self.url = url

        def load(self):
            return []

    _stub_module("langchain_community.document_loaders", WebBaseLoader=_DummyWebBaseLoader)

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
    monkeypatch.setattr(main, "OpenAIEmbeddings", lambda *args, **kwargs: object())
    monkeypatch.setattr(main, "Qdrant", _DummyQdrantStore)


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
