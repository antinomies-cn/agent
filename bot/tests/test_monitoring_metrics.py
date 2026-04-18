from app.llm.custom_llm import CustomProxyLLM
from app.core import monitoring


def test_extract_usage_handles_missing_usage():
    usage = CustomProxyLLM._extract_usage({"choices": []})
    assert usage["prompt_tokens"] == 0
    assert usage["completion_tokens"] == 0
    assert usage["total_tokens"] == 0


def test_extract_usage_handles_invalid_values():
    usage = CustomProxyLLM._extract_usage(
        {
            "usage": {
                "prompt_tokens": "not-int",
                "completion_tokens": -2,
                "total_tokens": None,
            }
        }
    )
    assert usage["prompt_tokens"] == 0
    assert usage["completion_tokens"] == 0
    assert usage["total_tokens"] == 0


def test_extract_usage_falls_back_total_tokens_sum():
    usage = CustomProxyLLM._extract_usage(
        {
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 0,
            }
        }
    )
    assert usage["prompt_tokens"] == 11
    assert usage["completion_tokens"] == 7
    assert usage["total_tokens"] == 18


def test_monitoring_alert_on_llm_timeout_streak(monkeypatch):
    monitoring.reset_monitoring_state()
    monkeypatch.setenv("MONITOR_LLM_TIMEOUT_STREAK_THRESHOLD", "2")
    monkeypatch.setenv("MONITOR_ALERT_COOLDOWN_SECONDS", "0")

    monitoring.record_llm_call(model="m1", path="/chat", status="timeout", elapsed_ms=10)
    monitoring.record_llm_call(model="m1", path="/chat", status="timeout", elapsed_ms=12)

    snap = monitoring.get_metrics_snapshot()
    alerts = [x for x in snap["alerts_total"] if x["alert_type"] == "llm_timeout_streak"]
    assert alerts
    assert alerts[0]["value"] >= 1


def test_monitoring_alert_on_llm_error_rate(monkeypatch):
    monitoring.reset_monitoring_state()
    monkeypatch.setenv("MONITOR_LLM_ERROR_RATE_WINDOW_SECONDS", "60")
    monkeypatch.setenv("MONITOR_LLM_ERROR_RATE_MIN_SAMPLES", "4")
    monkeypatch.setenv("MONITOR_LLM_ERROR_RATE_THRESHOLD", "0.5")
    monkeypatch.setenv("MONITOR_ALERT_COOLDOWN_SECONDS", "0")

    monitoring.record_llm_call(model="m2", path="/chat", status="success", elapsed_ms=5)
    monitoring.record_llm_call(model="m2", path="/chat", status="timeout", elapsed_ms=5)
    monitoring.record_llm_call(model="m2", path="/chat", status="http_error", elapsed_ms=5)
    monitoring.record_llm_call(model="m2", path="/chat", status="error", elapsed_ms=5)

    snap = monitoring.get_metrics_snapshot()
    alerts = [x for x in snap["alerts_total"] if x["alert_type"] == "llm_error_rate_high"]
    assert alerts
    assert alerts[0]["value"] >= 1