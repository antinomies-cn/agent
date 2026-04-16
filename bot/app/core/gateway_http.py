import time
import logging
from typing import Any, Dict, Iterable, Optional

import requests

from app.core.logger_setup import log_event

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _safe_status_code(error: Exception) -> int:
    if isinstance(error, requests.exceptions.HTTPError) and error.response is not None:
        return int(error.response.status_code)
    return 0


def _is_retryable_exception(error: Exception) -> bool:
    if isinstance(error, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return True
    if isinstance(error, requests.exceptions.HTTPError):
        return _safe_status_code(error) in _RETRYABLE_STATUS_CODES
    return False


def post_json_with_retry(
    url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
    timeout_seconds: float,
    retry_count: int,
    component: str,
    operation: str,
    accepted_error_statuses: Optional[Iterable[int]] = None,
) -> requests.Response:
    """POST JSON with bounded retries; returns response for optional accepted error statuses."""
    accepted = set(accepted_error_statuses or [])
    max_retries = max(int(retry_count or 0), 0)
    timeout_seconds = max(float(timeout_seconds or 0), 3.0)

    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        start = time.perf_counter()
        attempt_no = attempt + 1
        try:
            response = requests.post(url, headers=headers, json=body, timeout=timeout_seconds)
            if response.status_code >= 400:
                if response.status_code in accepted:
                    log_event(
                        logging.INFO,
                        f"{component}.{operation}.accepted_error",
                        endpoint=url,
                        attempt=attempt_no,
                        max_attempts=max_retries + 1,
                        status_code=response.status_code,
                        elapsed_ms=int((time.perf_counter() - start) * 1000),
                    )
                    return response
                response.raise_for_status()

            log_event(
                logging.INFO,
                f"{component}.{operation}.success",
                endpoint=url,
                attempt=attempt_no,
                max_attempts=max_retries + 1,
                status_code=response.status_code,
                elapsed_ms=int((time.perf_counter() - start) * 1000),
            )
            return response
        except Exception as error:  # pragma: no cover - behavior validated via public callers
            last_error = error
            should_retry = attempt < max_retries and _is_retryable_exception(error)
            status_code = _safe_status_code(error)
            elapsed_ms = int((time.perf_counter() - start) * 1000)

            if not should_retry:
                log_event(
                    logging.ERROR,
                    f"{component}.{operation}.error",
                    endpoint=url,
                    attempt=attempt_no,
                    max_attempts=max_retries + 1,
                    status_code=status_code,
                    elapsed_ms=elapsed_ms,
                    error=str(error)[:220],
                )
                raise

            sleep_seconds = min(1.5, 0.3 * (attempt + 1))
            log_event(
                logging.WARNING,
                f"{component}.{operation}.retry",
                endpoint=url,
                attempt=attempt_no,
                max_attempts=max_retries + 1,
                status_code=status_code,
                backoff_seconds=sleep_seconds,
                elapsed_ms=elapsed_ms,
            )
            time.sleep(sleep_seconds)

    if last_error is not None:
        raise last_error
    raise RuntimeError("gateway request failed without error detail")
