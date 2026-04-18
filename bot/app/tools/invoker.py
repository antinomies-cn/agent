from __future__ import annotations

import inspect
import json
from typing import Any


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
    if code == "TIMEOUT":
        return "工具调用超时，请稍后重试或上调超时参数。"
    if code.startswith("HTTP_"):
        return "上游服务返回HTTP错误，请检查上游可用性和鉴权。"
    return "工具执行失败，请检查输入参数与依赖服务状态。"


def wrap_tool_result(tool_name: str, raw_result: Any) -> dict[str, Any]:
    parsed = None
    if isinstance(raw_result, str):
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
    if hasattr(tool_obj, "invoke"):
        result = tool_obj.invoke(validated_payload)
    elif callable(tool_obj):
        result = tool_obj(**validated_payload)
    else:
        raise TypeError(f"tool {tool_name} is not invokable")
    return wrap_tool_result(tool_name, result)
