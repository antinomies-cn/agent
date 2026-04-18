import types

from fastapi.testclient import TestClient

from app import main
from app.api.routers import ops as ops_router
from app.api.routers import tools as tools_router
from app.core.texts import USER_MESSAGES


class _Invoker:
    def __init__(self, result):
        self._result = result

    def invoke(self, payload):
        return self._result


class _TimeoutInvoker:
    def __init__(self, sleep_seconds):
        self._sleep_seconds = sleep_seconds

    def invoke(self, payload):
        import time

        time.sleep(self._sleep_seconds)
        return "late"


class _ErrorInvoker:
    def invoke(self, payload):
        raise RuntimeError("boom")


def test_tools_endpoint_wraps_structured_error_payload(monkeypatch):
    monkeypatch.setattr(
        tools_router,
        "test",
        _Invoker('{"ok": false, "code": "CONFIG_MISSING", "data": null, "error": "missing"}'),
    )

    client = TestClient(main.app)
    resp = client.post("/tools/test", json={"scope": "all"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["tool"] == "test"
    assert body["ok"] is False
    assert body["code"] == "CONFIG_MISSING"
    assert body["error"] == "missing"
    assert body["explanation"]


def test_tools_endpoint_wraps_plain_text_result(monkeypatch):
    monkeypatch.setattr(tools_router, "search", _Invoker("plain text result"))

    client = TestClient(main.app)
    resp = client.post("/tools/search", json={"query": "hello"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["tool"] == "search"
    assert body["ok"] is True
    assert body["code"] == "OK"
    assert body["data"] == "plain text result"
    assert body["error"] == ""
    assert body["explanation"] == ""


def test_tools_invoke_endpoint_wraps_plain_text_result(monkeypatch):
    monkeypatch.setattr(tools_router, "search", _Invoker("plain text result"))

    client = TestClient(main.app)
    resp = client.post("/tools/invoke/search", json={"query": "hello"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["tool"] == "search"
    assert body["ok"] is True
    assert body["code"] == "OK"
    assert body["data"] == "plain text result"


def test_tools_invoke_endpoint_returns_not_found_for_unknown_tool():
    client = TestClient(main.app)
    resp = client.post("/tools/invoke/not_exists", json={})

    assert resp.status_code == 404
    body = resp.json()
    assert body["error_code"] == "TOOL_NOT_FOUND"


def test_tools_invoke_endpoint_returns_validation_error_for_bad_payload():
    client = TestClient(main.app)
    resp = client.post("/tools/invoke/search", json={})

    assert resp.status_code == 422
    body = resp.json()
    assert body["error_code"] == "REQUEST_VALIDATION_ERROR"
    assert body["error"]["errors"]


def test_tools_invoke_endpoint_returns_tool_timeout(monkeypatch):
    monkeypatch.setenv("TOOL_SEARCH_TIMEOUT_SECONDS", "0.01")
    monkeypatch.setenv("TOOL_SEARCH_RETRY_COUNT", "0")
    monkeypatch.setattr(tools_router, "search", _TimeoutInvoker(0.05))

    client = TestClient(main.app)
    resp = client.post("/tools/invoke/search", json={"query": "hello"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["tool"] == "search"
    assert body["ok"] is False
    assert body["code"] == "TOOL_TIMEOUT"


def test_tools_invoke_endpoint_returns_tool_exec_error(monkeypatch):
    monkeypatch.setenv("TOOL_SEARCH_RETRY_COUNT", "0")
    monkeypatch.setattr(tools_router, "search", _ErrorInvoker())

    client = TestClient(main.app)
    resp = client.post("/tools/invoke/search", json={"query": "hello"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["tool"] == "search"
    assert body["ok"] is False
    assert body["code"] == "TOOL_EXEC_ERROR"


def test_tools_schema_endpoint_returns_args_schema():
    client = TestClient(main.app)
    resp = client.get("/tools/schema/search")

    assert resp.status_code == 200
    body = resp.json()
    assert body["tool"] == "search"
    assert isinstance(body["schema"], dict)


def test_tools_legacy_routes_marked_deprecated_in_openapi():
    client = TestClient(main.app)
    resp = client.get("/openapi.json")

    assert resp.status_code == 200
    body = resp.json()
    paths = body["paths"]

    assert paths["/tools/test"]["post"]["deprecated"] is True
    assert paths["/tools/invoke/{tool_name}"]["post"].get("deprecated") is not True


def test_debug_ui_page_is_served():
    client = TestClient(main.app)
    resp = client.get("/debug/ui")

    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "工具调试台" in resp.text


def test_ops_qdrant_init_returns_error_when_dependency_missing(monkeypatch):
    monkeypatch.delattr(main, "init_qdrant_collection", raising=False)

    client = TestClient(main.app)
    resp = client.post("/qdrant/init")

    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 500
    assert body["error_code"] == "QDRANT_INIT_ERROR"
    assert body["explanation"]


def test_ops_health_live_returns_current_env(monkeypatch):
    monkeypatch.setenv("ENV", "dev")

    client = TestClient(main.app)
    resp = client.get("/health/live")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["env"] == "development"


def test_conversation_ws_happy_path(monkeypatch):
    monkeypatch.setattr(main, "ensure_websocket_auth", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "ensure_ws_rate_limit", lambda *args, **kwargs: None)

    async def _ok_run_async(text, session_id):
        return {"output": f"echo:{text}:{session_id}"}

    monkeypatch.setattr(main.master, "run_async", _ok_run_async)

    client = TestClient(main.app)
    with client.websocket_connect("/ws?session_id=s1") as ws:
        ws.send_text("ping")
        got = ws.receive_text()

    assert got == "echo:ping:s1"


def test_conversation_ws_uses_default_output_when_missing(monkeypatch):
    monkeypatch.setattr(main, "ensure_websocket_auth", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "ensure_ws_rate_limit", lambda *args, **kwargs: None)

    async def _empty_run_async(text, session_id):
        return {}

    monkeypatch.setattr(main.master, "run_async", _empty_run_async)

    client = TestClient(main.app)
    with client.websocket_connect("/ws?session_id=s1") as ws:
        ws.send_text("ping")
        got = ws.receive_text()

    assert got == USER_MESSAGES["ws_default"]


def test_conversation_ws_service_unavailable_branch(monkeypatch):
    monkeypatch.setattr(main, "ensure_websocket_auth", None)
    monkeypatch.setattr(main, "ensure_ws_rate_limit", None)

    client = TestClient(main.app)
    try:
        with client.websocket_connect("/ws?session_id=s1"):
            assert False, "expected websocket connection to be rejected"
    except Exception:
        assert True


def test_ops_health_unhealthy_when_master_missing(monkeypatch):
    monkeypatch.setattr(main, "master", None)

    client = TestClient(main.app)
    resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "unhealthy"
    assert body["error_code"] == "HEALTH_CHECK_ERROR"


def test_ops_health_handles_runtime_exception(monkeypatch):
    class _Master:
        def check_llm_health(self):
            raise RuntimeError("llm down")

    monkeypatch.setattr(main, "master", _Master())

    client = TestClient(main.app)
    resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "unhealthy"
    assert body["error_code"] == "HEALTH_CHECK_ERROR"
    assert body["explanation"]


def test_ops_memory_status_returns_error_when_master_missing(monkeypatch):
    monkeypatch.setattr(main, "master", None)

    client = TestClient(main.app)
    resp = client.get("/memory/status", params={"session_id": "s1"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 500
    assert body["error_code"] == "MEMORY_STATUS_ERROR"
    assert body["explanation"]
