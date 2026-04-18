from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.core.config import is_prod_runtime
from app.tools.invoker import ToolPayloadValidationError, get_tool_args_schema_json, get_tool_invoke_policy, invoke_tool
from app.tools.registry import TOOL_REGISTRY, get_tool, get_tool_metadata
from app.tools.mytools import (
    astro_current_chart,
    astro_day_scope,
    astro_month_scope,
    astro_my_sign,
    astro_natal_chart,
    astro_transit_chart,
    astro_week_scope,
    search,
    test,
    vector_search,
    xingpan,
)

router = APIRouter()
_DEBUG_UI_PATH = Path(__file__).resolve().parents[1] / "static" / "debug_ui.html"
_DEBUG_UI_HTML = _DEBUG_UI_PATH.read_text(encoding="utf-8")


class ToolTestRequest(BaseModel):
    scope: Optional[str] = Field(default="all", description="all|astro|vector|search")


class ToolSearchRequest(BaseModel):
    query: str = Field(description="搜索关键词")


class ToolVectorSearchRequest(BaseModel):
    query: str = Field(description="向量检索关键词")


class ToolXingpanRequest(BaseModel):
    name: str = Field(description="姓名")
    birth_dt: str = Field(description="出生时间 YYYY-MM-DD HH:MM:SS")
    longitude: float = Field(description="经度")
    latitude: float = Field(description="纬度")


class ToolAstroChartRequest(BaseModel):
    birth_dt: str = Field(description="出生时间 YYYY-MM-DD HH:MM:SS")
    longitude: float = Field(description="经度")
    latitude: float = Field(description="纬度")


def _ensure_debug_tools_enabled() -> None:
    if is_prod_runtime():
        raise HTTPException(
            status_code=404,
            detail={
                "message": "Not Found",
                "error_code": "DEBUG_TOOL_DISABLED",
                "explanation": "调试接口仅在开发环境开放，请在 dev 环境调用。",
            },
        )


def _resolve_debug_tool(tool_name: str):
    # 优先使用模块级对象，兼容测试中的 monkeypatch。
    patched = globals().get(tool_name)
    if patched is not None and hasattr(patched, "invoke"):
        return patched
    return get_tool(tool_name)


def _invoke_debug_tool(tool_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    tool_obj = _resolve_debug_tool(tool_name)
    if tool_obj is None:
        raise HTTPException(
            status_code=404,
            detail={
                "message": "工具不存在",
                "error_code": "TOOL_NOT_FOUND",
                "explanation": f"未找到工具: {tool_name}",
            },
        )

    try:
        return invoke_tool(tool_obj=tool_obj, tool_name=tool_name, payload=payload)
    except ToolPayloadValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors) from exc


def _register_legacy_tool_route(
    path: str,
    tool_name: str,
    summary: str,
    description: str,
    payload_builder,
    request_model=None,
    endpoint_name: str | None = None,
) -> None:
    def endpoint(payload: Optional[Dict[str, Any]] = Body(default=None)):
        _ensure_debug_tools_enabled()

        if request_model is not None:
            try:
                if hasattr(request_model, "model_validate"):
                    validated = request_model.model_validate(payload or {})
                else:
                    validated = request_model.parse_obj(payload or {})
            except Exception as exc:
                errors = exc.errors() if hasattr(exc, "errors") else [{"msg": str(exc)}]
                raise HTTPException(status_code=422, detail=errors) from exc
            built_payload = payload_builder(validated)
        else:
            built_payload = payload_builder(payload or {})

        return _invoke_debug_tool(tool_name, built_payload)

    endpoint.__name__ = endpoint_name or f"debug_tool_{tool_name.replace('/', '_').replace('-', '_')}"
    endpoint.__qualname__ = endpoint.__name__

    router.add_api_route(
        path,
        endpoint,
        methods=["POST"],
        summary=f"[Legacy] {summary}",
        description=description,
        deprecated=True,
    )


_register_legacy_tool_route(
    "/tools/test",
    "test",
    "工具调试：系统自检",
    "调用 test 工具，支持 scope=all|astro|vector|search 用于快速排查配置。",
    lambda payload: {"scope": payload.scope or "all"},
    ToolTestRequest,
)
_register_legacy_tool_route(
    "/tools/search",
    "search",
    "工具调试：联网搜索",
    "调用 search 工具，使用 SERPAPI 查询外部信息。",
    lambda payload: {"query": payload.query},
    ToolSearchRequest,
)
_register_legacy_tool_route(
    "/tools/vector_search",
    "vector_search",
    "工具调试：向量检索",
    "调用 vector_search 工具，从已入库文档中检索相似内容。",
    lambda payload: {"query": payload.query},
    ToolVectorSearchRequest,
)
_register_legacy_tool_route(
    "/tools/xingpan",
    "xingpan",
    "工具调试：星盘",
    "调用 xingpan 工具，依据姓名、出生时间、经纬度查询星盘。",
    lambda payload: {
        "name": payload.name,
        "birth_dt": payload.birth_dt,
        "longitude": payload.longitude,
        "latitude": payload.latitude,
    },
    ToolXingpanRequest,
)
_register_legacy_tool_route(
    "/tools/astro/my_sign",
    "astro_my_sign",
    "工具调试：星座信息",
    "调用 astro_my_sign 工具，读取 ASTRO_UID 并查询星座信息。",
    lambda payload: {
        "birth_dt": payload.birth_dt,
        "longitude": payload.longitude,
        "latitude": payload.latitude,
    },
    ToolAstroChartRequest,
)
_register_legacy_tool_route(
    "/tools/astro/day",
    "astro_day_scope",
    "工具调试：日运势",
    "调用 astro_day_scope 工具，读取 ASTRO_UID 并查询日运势。",
    lambda payload: {},
)
_register_legacy_tool_route(
    "/tools/astro/week",
    "astro_week_scope",
    "工具调试：周运势",
    "调用 astro_week_scope 工具，读取 ASTRO_UID 并查询周运势。",
    lambda payload: {},
)
_register_legacy_tool_route(
    "/tools/astro/month",
    "astro_month_scope",
    "工具调试：月运势",
    "调用 astro_month_scope 工具，读取 ASTRO_UID 并查询月运势。",
    lambda payload: {},
)
_register_legacy_tool_route(
    "/tools/astro/natal_chart",
    "astro_natal_chart",
    "工具调试：本命盘",
    "调用 astro_natal_chart 工具，依据出生时间与经纬度查询本命盘。",
    lambda payload: {
        "birth_dt": payload.birth_dt,
        "longitude": payload.longitude,
        "latitude": payload.latitude,
    },
    ToolAstroChartRequest,
)
_register_legacy_tool_route(
    "/tools/astro/current_chart",
    "astro_current_chart",
    "工具调试：当前天象盘",
    "调用 astro_current_chart 工具，查询当前天象盘。",
    lambda payload: {},
)
_register_legacy_tool_route(
    "/tools/astro/transit_chart",
    "astro_transit_chart",
    "工具调试：行运盘",
    "调用 astro_transit_chart 工具，依据出生时间与经纬度查询行运盘。",
    lambda payload: {
        "birth_dt": payload.birth_dt,
        "longitude": payload.longitude,
        "latitude": payload.latitude,
    },
    ToolAstroChartRequest,
)


@router.post("/tools/invoke/{tool_name}", summary="工具调试：统一调用入口", description="统一按工具名调用工具，payload 透传到工具参数。")
def debug_tool_invoke(tool_name: str, payload: Optional[Dict[str, Any]] = Body(default=None)):
    _ensure_debug_tools_enabled()
    return _invoke_debug_tool(tool_name, payload or {})


@router.get("/tools/schema/{tool_name}", summary="工具调试：参数模型", description="查看指定工具的参数Schema（仅开发环境）。")
def debug_tool_schema(tool_name: str):
    _ensure_debug_tools_enabled()
    tool_obj = _resolve_debug_tool(tool_name)
    if tool_obj is None:
        raise HTTPException(
            status_code=404,
            detail={
                "message": "工具不存在",
                "error_code": "TOOL_NOT_FOUND",
                "explanation": f"未找到工具: {tool_name}",
            },
        )

    metadata = get_tool_metadata(tool_name)
    return {
        "tool": tool_name,
        "schema": get_tool_args_schema_json(tool_obj),
        "metadata": {
            "owner": metadata.owner,
            "version": metadata.version,
            "risk_level": metadata.risk_level,
            "idempotent": metadata.idempotent,
            "requires_env": list(metadata.requires_env),
        }
        if metadata is not None
        else None,
        "policy": get_tool_invoke_policy(tool_name),
    }


@router.get("/tools/catalog", summary="工具调试：工具目录", description="查看全部工具的参数Schema、元数据与生效策略（仅开发环境）。")
def debug_tool_catalog():
    _ensure_debug_tools_enabled()

    items = []
    for tool_name, tool_obj in TOOL_REGISTRY.items():
        metadata = get_tool_metadata(tool_name)
        items.append(
            {
                "tool": tool_name,
                "schema": get_tool_args_schema_json(tool_obj),
                "metadata": {
                    "owner": metadata.owner,
                    "version": metadata.version,
                    "risk_level": metadata.risk_level,
                    "idempotent": metadata.idempotent,
                    "requires_env": list(metadata.requires_env),
                }
                if metadata is not None
                else None,
                "policy": get_tool_invoke_policy(tool_name),
            }
        )

    return {
        "count": len(items),
        "items": sorted(items, key=lambda one: one.get("tool", "")),
    }


@router.get("/debug/ui", response_class=HTMLResponse, summary="调试界面", description="简易工具调试界面，仅非生产可用。")
def debug_ui():
    _ensure_debug_tools_enabled()
    return HTMLResponse(content=_DEBUG_UI_HTML)
