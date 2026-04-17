import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from app.core.config import get_env_bool, get_env_float, get_env_int
from app.core.logger_setup import log_event
import logging


class CircuitOpenError(RuntimeError):
    def __init__(self, component: str, operation: str, retry_after_seconds: float) -> None:
        super().__init__(
            f"circuit open: component={component} operation={operation} retry_after_seconds={retry_after_seconds:.2f}"
        )
        self.component = component
        self.operation = operation
        self.retry_after_seconds = max(float(retry_after_seconds), 0.0)


@dataclass(frozen=True)
class CircuitBreakerSettings:
    enabled: bool
    failure_threshold: int
    open_seconds: float


def _resolve_component_prefix(component: str) -> str:
    clean = (component or "").strip().upper()
    if clean == "LLM":
        return "LLM"
    if clean == "QDRANT":
        return "QDRANT"
    return "EXTERNAL_API"


def _resolve_settings(component: str) -> CircuitBreakerSettings:
    enabled = get_env_bool("GATEWAY_CIRCUIT_BREAKER_ENABLED", default=True)
    prefix = _resolve_component_prefix(component)

    failure_threshold = get_env_int(f"{prefix}_CB_FAILURE_THRESHOLD", default=3, min_value=1)
    open_seconds = get_env_float(f"{prefix}_CB_OPEN_SECONDS", default=30.0, min_value=1.0)

    return CircuitBreakerSettings(
        enabled=enabled,
        failure_threshold=failure_threshold,
        open_seconds=open_seconds,
    )


class _CircuitBreaker:
    def __init__(self, component: str) -> None:
        self.component = component
        self._lock = threading.Lock()
        self._state = "closed"
        self._failure_count = 0
        self._opened_at = 0.0

    def _check_can_execute(self, operation: str, settings: CircuitBreakerSettings) -> None:
        if not settings.enabled:
            return

        with self._lock:
            if self._state != "open":
                return

            elapsed = time.time() - self._opened_at
            if elapsed >= settings.open_seconds:
                # Move to half-open by allowing a single probe call.
                self._state = "half_open"
                return

            retry_after = settings.open_seconds - elapsed
            raise CircuitOpenError(self.component, operation, retry_after)

    def _on_success(self) -> None:
        with self._lock:
            self._state = "closed"
            self._failure_count = 0
            self._opened_at = 0.0

    def _on_failure(self, operation: str, settings: CircuitBreakerSettings, error: Exception) -> None:
        if not settings.enabled:
            return

        with self._lock:
            self._failure_count += 1
            if self._failure_count >= settings.failure_threshold:
                self._state = "open"
                self._opened_at = time.time()
                log_event(
                    logging.WARNING,
                    "gateway.circuit.open",
                    component=self.component,
                    operation=operation,
                    failure_count=self._failure_count,
                    threshold=settings.failure_threshold,
                    open_seconds=settings.open_seconds,
                    error=str(error)[:180],
                )

    def execute(self, operation: str, func: Callable[[], Any], fallback: Optional[Callable[[Exception], Any]] = None) -> Any:
        settings = _resolve_settings(self.component)

        try:
            self._check_can_execute(operation=operation, settings=settings)
        except Exception as open_error:
            log_event(
                logging.WARNING,
                "gateway.circuit.block",
                component=self.component,
                operation=operation,
                error=str(open_error)[:180],
            )
            if fallback is not None:
                return fallback(open_error)
            raise

        start = time.perf_counter()
        try:
            result = func()
            self._on_success()
            log_event(
                logging.INFO,
                "gateway.circuit.success",
                component=self.component,
                operation=operation,
                elapsed_ms=int((time.perf_counter() - start) * 1000),
            )
            return result
        except Exception as run_error:
            self._on_failure(operation=operation, settings=settings, error=run_error)
            log_event(
                logging.WARNING,
                "gateway.circuit.failure",
                component=self.component,
                operation=operation,
                elapsed_ms=int((time.perf_counter() - start) * 1000),
                error=str(run_error)[:180],
            )
            if fallback is not None:
                return fallback(run_error)
            raise

    def snapshot(self) -> Dict[str, Any]:
        settings = _resolve_settings(self.component)
        with self._lock:
            state = self._state
            failure_count = self._failure_count
            opened_at = self._opened_at

        retry_after_seconds = 0.0
        if settings.enabled and state == "open":
            elapsed = time.time() - opened_at
            retry_after_seconds = max(settings.open_seconds - elapsed, 0.0)

        return {
            "component": self.component,
            "enabled": settings.enabled,
            "state": state,
            "failure_count": failure_count,
            "failure_threshold": settings.failure_threshold,
            "open_seconds": settings.open_seconds,
            "retry_after_seconds": round(retry_after_seconds, 3),
        }


class _CircuitRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: Dict[str, _CircuitBreaker] = {}

    def get(self, component: str) -> _CircuitBreaker:
        key = (component or "external_api").strip().lower()
        with self._lock:
            breaker = self._items.get(key)
            if breaker is None:
                breaker = _CircuitBreaker(component=key)
                self._items[key] = breaker
            return breaker

    def snapshot(self) -> list[Dict[str, Any]]:
        with self._lock:
            items = list(self._items.items())
        return [breaker.snapshot() for _, breaker in sorted(items, key=lambda item: item[0])]


_registry = _CircuitRegistry()


def resilience_execute(
    component: str,
    operation: str,
    func: Callable[[], Any],
    fallback: Optional[Callable[[Exception], Any]] = None,
) -> Any:
    breaker = _registry.get(component)
    return breaker.execute(operation=operation, func=func, fallback=fallback)


def get_resilience_snapshot() -> list[Dict[str, Any]]:
    return _registry.snapshot()
