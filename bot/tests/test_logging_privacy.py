import json
import logging

from app.core import logger_setup


def test_log_event_sanitizes_sensitive_fields_in_prod(monkeypatch, caplog):
    monkeypatch.setenv("ENV", "prod")
    caplog.set_level(logging.INFO)

    logger_setup.log_event(
        logging.INFO,
        "privacy.test",
        session_id="user-session-123456",
        query="hello secret world",
        input="user input payload",
        response="assistant response payload",
        error="something bad happened with secret",
    )

    payload = json.loads(caplog.records[-1].message)

    assert payload["session_id"].startswith("sid:")
    assert payload["query"].startswith("len=")
    assert payload["input"].startswith("len=")
    assert payload["response"].startswith("len=")
    assert payload["error"].startswith("len=")
    assert "hello secret world" not in caplog.records[-1].message
    assert "user-session-123456" not in caplog.records[-1].message


def test_mask_session_id_and_text_preview(monkeypatch):
    monkeypatch.setenv("ENV", "dev")

    masked = logger_setup.mask_session_id("abc123456789")
    summary = logger_setup.summarize_text_for_log("line1\nline2", preview_chars=8)

    assert masked == "abc1…6789"
    assert "len=" in summary
    assert "sha256=" in summary
    assert "line1" in summary
    assert "\n" not in summary
