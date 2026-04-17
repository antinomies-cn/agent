import json
import os
import logging
import logging.handlers
import hashlib
import uuid
from contextvars import ContextVar
from app.core.config import get_env_int, get_env_str, is_prod_runtime


_TRACE_ID_CTX: ContextVar[str] = ContextVar("trace_id", default="")


def _is_prod_runtime() -> bool:
    return is_prod_runtime()


def _sha256_short(text: str) -> str:
    clean = (text or "").strip()
    if not clean:
        return ""
    return hashlib.sha256(clean.encode("utf-8")).hexdigest()[:12]


def mask_session_id(session_id: str) -> str:
    clean = (session_id or "").strip()
    if not clean:
        return ""
    if _is_prod_runtime():
        return f"sid:{_sha256_short(clean)}"
    if len(clean) <= 8:
        return clean
    return f"{clean[:4]}…{clean[-4:]}"


def summarize_text_for_log(text: str, preview_chars: int = 32) -> str:
    clean = (text or "").strip()
    if not clean:
        return ""

    length = len(clean)
    digest = _sha256_short(clean)
    preview = clean[: max(0, preview_chars)].replace("\r", " ").replace("\n", " ")
    if _is_prod_runtime():
        return f"len={length},sha256={digest}"
    if len(clean) > preview_chars:
        preview = f"{preview}…"
    return f"len={length},sha256={digest},preview={preview}"


def summarize_error_for_log(error: str, preview_chars: int = 80) -> str:
    clean = (error or "").strip()
    if not clean:
        return ""
    if _is_prod_runtime():
        return f"len={len(clean)},sha256={_sha256_short(clean)}"
    if len(clean) > preview_chars:
        return f"{clean[:preview_chars]}…"
    return clean


def _sanitize_field(key: str, value):
    lowered = (key or "").strip().lower()
    if lowered in {"session_id", "session", "sid"}:
        return mask_session_id(str(value))
    if lowered in {"query", "input", "output", "response", "content", "message", "prompt", "tool_input", "preview"}:
        return summarize_text_for_log(str(value))
    if lowered in {"error", "err", "detail"}:
        return summarize_error_for_log(str(value))
    return value


def setup_logger():
    """配置规范化的日志系统。"""
    log_level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    log_level = get_env_str("LOG_LEVEL", "info").lower() or "info"
    log_dir = get_env_str("LOG_DIR", "./logs") or "./logs"
    log_max_size = get_env_int("LOG_MAX_SIZE", default=10 * 1024 * 1024, min_value=1)
    log_backup_count = get_env_int("LOG_BACKUP_COUNT", default=5, min_value=1)

    os.makedirs(log_dir, exist_ok=True)

    log_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(module)s | %(funcName)s | %(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level_map.get(log_level, logging.INFO))
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    root_logger.addHandler(console_handler)

    file_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, "divination_master.log"),
        maxBytes=log_max_size,
        backupCount=log_backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(log_format)
    root_logger.addHandler(file_handler)

    return root_logger


logger = setup_logger()


def get_trace_id() -> str:
    trace_id = _TRACE_ID_CTX.get("")
    if trace_id:
        return trace_id

    fallback = get_env_str("TRACE_ID", "")
    if fallback:
        return fallback
    return ""


def set_trace_id(trace_id: str = "") -> str:
    value = (trace_id or "").strip() or uuid.uuid4().hex[:16]
    _TRACE_ID_CTX.set(value)
    return value


def clear_trace_id() -> None:
    _TRACE_ID_CTX.set("")


def log_event(level: int, event: str, **fields) -> None:
    payload = {"event": event}
    if "component" not in fields:
        payload["component"] = (event or "unknown").split(".")[0]

    trace_id = fields.get("trace_id") or get_trace_id()
    if trace_id:
        payload["trace_id"] = trace_id

    for key, value in fields.items():
        payload[key] = _sanitize_field(key, value)
    logger.log(level, json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
