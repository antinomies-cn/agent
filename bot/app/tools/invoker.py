from __future__ import annotations

import concurrent.futures
import inspect
import json
import logging
import re
import time
from typing import Any

from app.core.config import get_env_float, get_env_int
from app.core.logger_setup import log_event
from app.tools.registry import get_tool_metadata


class ToolPayloadValidationError(ValueError):
    def __init__(self, message: str, errors: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.errors = errors or []


def _validate_with_schema(schema: Any, payload: dict[str, Any]) -> dict[str, Any]:
    if hasattr(schema, "model_validate"):
        validated = schema.model_validate(payload)
        if hasattr(validated, "model_dump"):
            return validated.model_dump()
        return payload

    if hasattr(schema, "parse_obj"):
        validated = schema.parse_obj(payload)
        if hasattr(validated, "dict"):
            return validated.dict()
        return payload

    return payload


def validate_tool_payload(tool_obj: Any, payload: dict[str, Any]) -> dict[str, Any]:
    args_schema = getattr(tool_obj, "args_schema", None)
    if args_schema is None:
        return _validate_with_signature(tool_obj, payload)

    try:
        return _validate_with_schema(args_schema, payload)
    except Exception as exc:
        errors = exc.errors() if hasattr(exc, "errors") else [{"msg": str(exc)}]
        raise ToolPayloadValidationError("工具参数校验失败", errors=errors) from exc


def _validate_with_signature(tool_obj: Any, payload: dict[str, Any]) -> dict[str, Any]:
    if not callable(tool_obj):
        return payload

    try:
        signature = inspect.signature(tool_obj)
    except Exception:
        return payload

    missing_required: list[dict[str, Any]] = []
    normalized_payload = dict(payload)
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if param.default is inspect.Parameter.empty and name not in normalized_payload:
            missing_required.append(
                {
                    "loc": ["body", name],
                    "msg": "field required",
                    "type": "value_error.missing",
                }
            )

    if missing_required:
        raise ToolPayloadValidationError("工具参数校验失败", errors=missing_required)

    return normalized_payload


def get_tool_args_schema_json(tool_obj: Any) -> dict[str, Any]:
    args_schema = getattr(tool_obj, "args_schema", None)
    if args_schema is None:
        return {}

    if hasattr(args_schema, "model_json_schema"):
        return args_schema.model_json_schema()

    if hasattr(args_schema, "schema"):
        return args_schema.schema()

    if callable(tool_obj):
        return _build_schema_from_signature(tool_obj)

    return {}


def _build_schema_from_signature(tool_obj: Any) -> dict[str, Any]:
    try:
        signature = inspect.signature(tool_obj)
    except Exception:
        return {}

    required: list[str] = []
    properties: dict[str, dict[str, Any]] = {}
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        dict: "object",
        list: "array",
    }

    for name, param in signature.parameters.items():
        if name == "self":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        annotation = param.annotation if param.annotation is not inspect.Parameter.empty else None
        json_type = type_map.get(annotation, "string")
        properties[name] = {"title": name, "type": json_type}

        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def tool_error_explanation(code: str, error: str) -> str:
    if not error:
        return ""
    if code == "CONFIG_MISSING":
        return "工具配置缺失，请检查相关环境变量。"
    if code in {"TIMEOUT", "TOOL_TIMEOUT"}:
        return "工具调用超时，请稍后重试或上调超时参数。"
    if code.startswith("HTTP_"):
        return "上游服务返回HTTP错误，请检查上游可用性和鉴权。"
    if code == "TOOL_EXEC_ERROR":
        return "工具执行异常，请检查依赖服务与输入参数。"
    return "工具执行失败，请检查输入参数与依赖服务状态。"


def _to_env_suffix(tool_name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", (tool_name or "").strip()).upper().strip("_")
    return normalized or "TOOL"


def _resolve_invoke_policy(tool_name: str) -> tuple[float, int]:
    meta = get_tool_metadata(tool_name)
    metadata_timeout = meta.timeout_seconds if meta is not None else 8.0
    metadata_retry = meta.retry_count if meta is not None else 0

    default_timeout = get_env_float("TOOL_DEFAULT_TIMEOUT_SECONDS", default=metadata_timeout, min_value=0.01)
    default_retry = get_env_int("TOOL_DEFAULT_RETRY_COUNT", default=metadata_retry, min_value=0)

    suffix = _to_env_suffix(tool_name)
    timeout = get_env_float(f"TOOL_{suffix}_TIMEOUT_SECONDS", default=default_timeout, min_value=0.01)
    retry_count = get_env_int(f"TOOL_{suffix}_RETRY_COUNT", default=default_retry, min_value=0)
    return timeout, retry_count


def _execute_once(tool_obj: Any, payload: dict[str, Any], timeout_seconds: float) -> Any:
    if hasattr(tool_obj, "invoke"):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(tool_obj.invoke, payload)
            return future.result(timeout=timeout_seconds)

    if callable(tool_obj):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(tool_obj, **payload)
            return future.result(timeout=timeout_seconds)

    raise TypeError("tool object is not invokable")


def wrap_tool_result(tool_name: str, raw_result: Any) -> dict[str, Any]:
    parsed = None
    if isinstance(raw_result, dict):
        parsed = raw_result
    elif isinstance(raw_result, str):
        try:
            parsed = json.loads(raw_result)
        except Exception:
            parsed = None

    if isinstance(parsed, dict) and {"ok", "code", "data", "error"}.issubset(parsed.keys()):
        wrapped = {"tool": tool_name, **parsed}
        wrapped["explanation"] = tool_error_explanation(str(wrapped.get("code", "")), str(wrapped.get("error", "")))
        return wrapped

    wrapped = {
        "tool": tool_name,
        "ok": True,
        "code": "OK",
        "data": parsed if parsed is not None else raw_result,
        "error": "",
    }
    wrapped["explanation"] = ""
    return wrapped


def invoke_tool(tool_obj: Any, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    validated_payload = validate_tool_payload(tool_obj=tool_obj, payload=payload)
    timeout_seconds, retry_count = _resolve_invoke_policy(tool_name)

    last_error = ""
    for attempt in range(retry_count + 1):
        start = time.perf_counter()
        try:
            result = _execute_once(tool_obj=tool_obj, payload=validated_payload, timeout_seconds=timeout_seconds)
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            log_event(
                logging.INFO,
                "tool.invoke",
                tool_name=tool_name,
                success=True,
                attempt=attempt + 1,
                retry_count=retry_count,
                timeout_seconds=timeout_seconds,
                elapsed_ms=elapsed_ms,
            )
            return wrap_tool_result(tool_name, result)
        except concurrent.futures.TimeoutError:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            last_error = f"工具调用超时（timeout={timeout_seconds}s）"
            log_event(
                logging.WARNING,
                "tool.invoke",
                tool_name=tool_name,
                success=False,
                attempt=attempt + 1,
                retry_count=retry_count,
                timeout_seconds=timeout_seconds,
                elapsed_ms=elapsed_ms,
                error_code="TOOL_TIMEOUT",
                error=last_error,
            )
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            last_error = str(exc)[:200]
            log_event(
                logging.ERROR,
                "tool.invoke",
                tool_name=tool_name,
                success=False,
                attempt=attempt + 1,
                retry_count=retry_count,
                timeout_seconds=timeout_seconds,
                elapsed_ms=elapsed_ms,
                error_code="TOOL_EXEC_ERROR",
                error=last_error,
            )

    if "超时" in last_error or "timeout" in last_error.lower():
        return wrap_tool_result(
            tool_name,
            {
                "ok": False,
                "code": "TOOL_TIMEOUT",
                "data": None,
                "error": last_error or "工具调用超时",
            },
        )

    return wrap_tool_result(
        tool_name,
        {
            "ok": False,
            "code": "TOOL_EXEC_ERROR",
            "data": None,
            "error": last_error or "工具执行异常",
        },
    )
