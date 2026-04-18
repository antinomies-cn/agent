import threading
import time
from collections import defaultdict
from collections import deque
from typing import Dict, Tuple

from app.core.config import get_env_float
from app.core.config import get_env_int
from app.core.logger_setup import log_event
import logging


class _MonitoringStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._http_requests_total: Dict[Tuple[str, str, str], int] = defaultdict(int)
        self._http_request_latency_ms_sum: Dict[Tuple[str, str], int] = defaultdict(int)

        self._llm_calls_total: Dict[Tuple[str, str, str], int] = defaultdict(int)
        self._llm_latency_ms_sum: Dict[Tuple[str, str], int] = defaultdict(int)

        self._llm_tokens_total: Dict[Tuple[str, str, str], int] = defaultdict(int)
        self._alerts_total: Dict[Tuple[str, str, str], int] = defaultdict(int)

        self._last_alert_ts: Dict[Tuple[str, str], float] = {}
        self._llm_timeout_streak: Dict[Tuple[str, str], int] = defaultdict(int)
        self._llm_outcomes: Dict[Tuple[str, str], deque] = defaultdict(deque)
        self._http_outcomes: Dict[Tuple[str, str], deque] = defaultdict(deque)

    @staticmethod
    def _is_llm_failure_status(status: str) -> bool:
        return status in {
            "http_error",
            "timeout",
            "circuit_open",
            "connection_error",
            "parse_error",
            "error",
        }

    @staticmethod
    def _is_http_server_error(status_code: str) -> bool:
        try:
            code = int(status_code)
        except (TypeError, ValueError):
            return False
        return code >= 500

    @staticmethod
    def _prune_events(window: deque, now_ts: float, window_seconds: float) -> None:
        min_ts = now_ts - max(float(window_seconds), 1.0)
        while window and window[0][0] < min_ts:
            window.popleft()

    def _evaluate_llm_timeout_streak(self, model: str, path: str) -> None:
        threshold = get_env_int("MONITOR_LLM_TIMEOUT_STREAK_THRESHOLD", default=3, min_value=1)
        streak = self._llm_timeout_streak[(model, path)]
        if streak < threshold:
            return

        self.maybe_emit_alert(
            alert_type="llm_timeout_streak",
            severity="critical",
            source=f"llm.{model}",
            message="LLM 连续超时达到阈值",
            cooldown_seconds=get_env_float("MONITOR_ALERT_COOLDOWN_SECONDS", default=120.0, min_value=0.0),
            extra_fields={
                "model": model,
                "path": path,
                "streak": streak,
                "threshold": threshold,
            },
        )

    def _evaluate_llm_error_rate(self, model: str, path: str, now_ts: float) -> None:
        window_seconds = get_env_float("MONITOR_LLM_ERROR_RATE_WINDOW_SECONDS", default=300.0, min_value=1.0)
        min_samples = get_env_int("MONITOR_LLM_ERROR_RATE_MIN_SAMPLES", default=20, min_value=1)
        threshold = get_env_float("MONITOR_LLM_ERROR_RATE_THRESHOLD", default=0.4, min_value=0.0)

        key = (model, path)
        bucket = self._llm_outcomes[key]
        self._prune_events(bucket, now_ts, window_seconds)
        total = len(bucket)
        if total < min_samples:
            return

        failed = sum(1 for _, status in bucket if self._is_llm_failure_status(status))
        ratio = failed / total if total > 0 else 0.0
        if ratio < threshold:
            return

        self.maybe_emit_alert(
            alert_type="llm_error_rate_high",
            severity="critical",
            source=f"llm.{model}",
            message="LLM 错误率超过阈值",
            cooldown_seconds=get_env_float("MONITOR_ALERT_COOLDOWN_SECONDS", default=120.0, min_value=0.0),
            extra_fields={
                "model": model,
                "path": path,
                "window_seconds": window_seconds,
                "sample_count": total,
                "failed_count": failed,
                "error_rate": round(ratio, 4),
                "threshold": threshold,
            },
        )

    def _evaluate_http_5xx_rate(self, path: str, method: str, now_ts: float) -> None:
        window_seconds = get_env_float("MONITOR_HTTP_5XX_RATE_WINDOW_SECONDS", default=300.0, min_value=1.0)
        min_samples = get_env_int("MONITOR_HTTP_5XX_RATE_MIN_SAMPLES", default=30, min_value=1)
        threshold = get_env_float("MONITOR_HTTP_5XX_RATE_THRESHOLD", default=0.3, min_value=0.0)

        key = (path, method)
        bucket = self._http_outcomes[key]
        self._prune_events(bucket, now_ts, window_seconds)
        total = len(bucket)
        if total < min_samples:
            return

        failed = sum(1 for _, status_code in bucket if self._is_http_server_error(status_code))
        ratio = failed / total if total > 0 else 0.0
        if ratio < threshold:
            return

        self.maybe_emit_alert(
            alert_type="http_5xx_rate_high",
            severity="critical",
            source=f"http.{method}",
            message="HTTP 5xx 比例超过阈值",
            cooldown_seconds=get_env_float("MONITOR_ALERT_COOLDOWN_SECONDS", default=120.0, min_value=0.0),
            extra_fields={
                "path": path,
                "method": method,
                "window_seconds": window_seconds,
                "sample_count": total,
                "failed_count": failed,
                "error_rate": round(ratio, 4),
                "threshold": threshold,
            },
        )

    def record_http_request(self, path: str, method: str, status_code: int, elapsed_ms: int) -> None:
        clean_path = (path or "unknown").strip() or "unknown"
        clean_method = (method or "GET").upper()
        status = str(int(status_code))
        elapsed = max(int(elapsed_ms), 0)
        now_ts = time.time()
        with self._lock:
            self._http_requests_total[(clean_path, clean_method, status)] += 1
            self._http_request_latency_ms_sum[(clean_path, clean_method)] += elapsed
            bucket = self._http_outcomes[(clean_path, clean_method)]
            bucket.append((now_ts, status))
            self._prune_events(
                bucket,
                now_ts,
                get_env_float("MONITOR_HTTP_5XX_RATE_WINDOW_SECONDS", default=300.0, min_value=1.0),
            )

            self._evaluate_http_5xx_rate(clean_path, clean_method, now_ts)

    def record_llm_call(self, model: str, path: str, status: str, elapsed_ms: int) -> None:
        clean_model = (model or "unknown").strip() or "unknown"
        clean_path = (path or "unknown").strip() or "unknown"
        clean_status = (status or "unknown").strip() or "unknown"
        elapsed = max(int(elapsed_ms), 0)
        now_ts = time.time()
        with self._lock:
            self._llm_calls_total[(clean_model, clean_path, clean_status)] += 1
            self._llm_latency_ms_sum[(clean_model, clean_path)] += elapsed

            if clean_status == "timeout":
                self._llm_timeout_streak[(clean_model, clean_path)] += 1
            else:
                self._llm_timeout_streak[(clean_model, clean_path)] = 0

            llm_bucket = self._llm_outcomes[(clean_model, clean_path)]
            llm_bucket.append((now_ts, clean_status))
            self._prune_events(
                llm_bucket,
                now_ts,
                get_env_float("MONITOR_LLM_ERROR_RATE_WINDOW_SECONDS", default=300.0, min_value=1.0),
            )

            self._evaluate_llm_timeout_streak(clean_model, clean_path)
            self._evaluate_llm_error_rate(clean_model, clean_path, now_ts)

    def record_llm_usage(self, model: str, path: str, prompt_tokens: int, completion_tokens: int, total_tokens: int) -> None:
        clean_model = (model or "unknown").strip() or "unknown"
        clean_path = (path or "unknown").strip() or "unknown"
        pt = max(int(prompt_tokens), 0)
        ct = max(int(completion_tokens), 0)
        tt = max(int(total_tokens), 0)
        with self._lock:
            self._llm_tokens_total[("prompt", clean_model, clean_path)] += pt
            self._llm_tokens_total[("completion", clean_model, clean_path)] += ct
            self._llm_tokens_total[("total", clean_model, clean_path)] += tt

    def maybe_emit_alert(
        self,
        *,
        alert_type: str,
        severity: str,
        source: str,
        message: str,
        cooldown_seconds: float,
        extra_fields: dict,
    ) -> bool:
        clean_type = (alert_type or "unknown").strip() or "unknown"
        clean_severity = (severity or "warning").strip() or "warning"
        clean_source = (source or "unknown").strip() or "unknown"

        cooldown = max(float(cooldown_seconds), 0.0)
        now = time.time()
        key = (clean_type, clean_source)

        with self._lock:
            last_ts = self._last_alert_ts.get(key, 0.0)
            if cooldown > 0 and (now - last_ts) < cooldown:
                return False

            self._last_alert_ts[key] = now
            self._alerts_total[(clean_type, clean_severity, clean_source)] += 1

        payload = {
            "alert_type": clean_type,
            "severity": clean_severity,
            "source": clean_source,
            "message": message,
            "cooldown_seconds": cooldown,
        }
        payload.update(extra_fields)
        log_event(logging.ERROR, "alert.triggered", **payload)
        return True

    def snapshot(self) -> dict:
        with self._lock:
            http_total = dict(self._http_requests_total)
            http_latency = dict(self._http_request_latency_ms_sum)
            llm_calls = dict(self._llm_calls_total)
            llm_latency = dict(self._llm_latency_ms_sum)
            llm_tokens = dict(self._llm_tokens_total)
            alerts = dict(self._alerts_total)
            timeout_streak = dict(self._llm_timeout_streak)

        return {
            "http_requests_total": [
                {"path": k[0], "method": k[1], "status_code": k[2], "value": v}
                for k, v in sorted(http_total.items())
            ],
            "http_request_latency_ms_sum": [
                {"path": k[0], "method": k[1], "value": v}
                for k, v in sorted(http_latency.items())
            ],
            "llm_calls_total": [
                {"model": k[0], "path": k[1], "status": k[2], "value": v}
                for k, v in sorted(llm_calls.items())
            ],
            "llm_latency_ms_sum": [
                {"model": k[0], "path": k[1], "value": v}
                for k, v in sorted(llm_latency.items())
            ],
            "llm_tokens_total": [
                {"token_type": k[0], "model": k[1], "path": k[2], "value": v}
                for k, v in sorted(llm_tokens.items())
            ],
            "alerts_total": [
                {"alert_type": k[0], "severity": k[1], "source": k[2], "value": v}
                for k, v in sorted(alerts.items())
            ],
            "llm_timeout_streak": [
                {"model": k[0], "path": k[1], "value": v}
                for k, v in sorted(timeout_streak.items())
                if v > 0
            ],
        }

    def prometheus_text(self) -> str:
        snap = self.snapshot()
        lines = []

        lines.append("# HELP app_http_requests_total Total HTTP requests")
        lines.append("# TYPE app_http_requests_total counter")
        for item in snap["http_requests_total"]:
            lines.append(
                f'app_http_requests_total{{path="{item["path"]}",method="{item["method"]}",status_code="{item["status_code"]}"}} {item["value"]}'
            )

        lines.append("# HELP app_http_request_latency_ms_sum Sum of HTTP request latency in ms")
        lines.append("# TYPE app_http_request_latency_ms_sum counter")
        for item in snap["http_request_latency_ms_sum"]:
            lines.append(
                f'app_http_request_latency_ms_sum{{path="{item["path"]}",method="{item["method"]}"}} {item["value"]}'
            )

        lines.append("# HELP app_llm_calls_total Total LLM calls")
        lines.append("# TYPE app_llm_calls_total counter")
        for item in snap["llm_calls_total"]:
            lines.append(
                f'app_llm_calls_total{{model="{item["model"]}",path="{item["path"]}",status="{item["status"]}"}} {item["value"]}'
            )

        lines.append("# HELP app_llm_latency_ms_sum Sum of LLM call latency in ms")
        lines.append("# TYPE app_llm_latency_ms_sum counter")
        for item in snap["llm_latency_ms_sum"]:
            lines.append(
                f'app_llm_latency_ms_sum{{model="{item["model"]}",path="{item["path"]}"}} {item["value"]}'
            )

        lines.append("# HELP app_llm_tokens_total Total LLM token usage")
        lines.append("# TYPE app_llm_tokens_total counter")
        for item in snap["llm_tokens_total"]:
            lines.append(
                f'app_llm_tokens_total{{token_type="{item["token_type"]}",model="{item["model"]}",path="{item["path"]}"}} {item["value"]}'
            )

        lines.append("# HELP app_alerts_total Total alert events")
        lines.append("# TYPE app_alerts_total counter")
        for item in snap["alerts_total"]:
            lines.append(
                f'app_alerts_total{{alert_type="{item["alert_type"]}",severity="{item["severity"]}",source="{item["source"]}"}} {item["value"]}'
            )

        lines.append("")
        return "\n".join(lines)


_store = _MonitoringStore()


def record_http_request(path: str, method: str, status_code: int, elapsed_ms: int) -> None:
    _store.record_http_request(path=path, method=method, status_code=status_code, elapsed_ms=elapsed_ms)


def record_llm_call(model: str, path: str, status: str, elapsed_ms: int) -> None:
    _store.record_llm_call(model=model, path=path, status=status, elapsed_ms=elapsed_ms)


def record_llm_usage(model: str, path: str, prompt_tokens: int, completion_tokens: int, total_tokens: int) -> None:
    _store.record_llm_usage(
        model=model,
        path=path,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def emit_alert(
    *,
    alert_type: str,
    severity: str,
    source: str,
    message: str,
    cooldown_seconds: float | None = None,
    **extra_fields,
) -> bool:
    cooldown = cooldown_seconds
    if cooldown is None:
        cooldown = get_env_float("MONITOR_ALERT_COOLDOWN_SECONDS", default=120.0, min_value=0.0)

    return _store.maybe_emit_alert(
        alert_type=alert_type,
        severity=severity,
        source=source,
        message=message,
        cooldown_seconds=cooldown,
        extra_fields=extra_fields,
    )


def get_metrics_snapshot() -> dict:
    return _store.snapshot()


def render_prometheus_metrics() -> str:
    return _store.prometheus_text()


def reset_monitoring_state() -> None:
    global _store
    _store = _MonitoringStore()