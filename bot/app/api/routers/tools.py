import json
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.core.config import is_prod_runtime
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


def _tool_error_explanation(code: str, error: str) -> str:
    if not error:
        return ""
    if code == "CONFIG_MISSING":
        return "工具配置缺失，请检查相关环境变量。"
    if code == "TIMEOUT":
        return "工具调用超时，请稍后重试或上调超时参数。"
    if code.startswith("HTTP_"):
        return "上游服务返回HTTP错误，请检查上游可用性和鉴权。"
    return "工具执行失败，请检查输入参数与依赖服务状态。"


def _wrap_tool_result(tool_name: str, raw_result: Any) -> Dict[str, Any]:
    parsed = None
    if isinstance(raw_result, str):
        try:
            parsed = json.loads(raw_result)
        except Exception:
            parsed = None

    if isinstance(parsed, dict) and {"ok", "code", "data", "error"}.issubset(parsed.keys()):
        wrapped = {"tool": tool_name, **parsed}
        wrapped["explanation"] = _tool_error_explanation(str(wrapped.get("code", "")), str(wrapped.get("error", "")))
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


@router.post("/tools/test", summary="工具调试：系统自检", description="调用 test 工具，支持 scope=all|astro|vector|search 用于快速排查配置。")
def debug_tool_test(payload: ToolTestRequest = Body(...)):
    _ensure_debug_tools_enabled()
    result = test.invoke({"scope": payload.scope or "all"})
    return _wrap_tool_result("test", result)


@router.post("/tools/search", summary="工具调试：联网搜索", description="调用 search 工具，使用 SERPAPI 查询外部信息。")
def debug_tool_search(payload: ToolSearchRequest = Body(...)):
    _ensure_debug_tools_enabled()
    result = search.invoke({"query": payload.query})
    return _wrap_tool_result("search", result)


@router.post("/tools/vector_search", summary="工具调试：向量检索", description="调用 vector_search 工具，从已入库文档中检索相似内容。")
def debug_tool_vector_search(payload: ToolVectorSearchRequest = Body(...)):
    _ensure_debug_tools_enabled()
    result = vector_search.invoke({"query": payload.query})
    return _wrap_tool_result("vector_search", result)


@router.post("/tools/xingpan", summary="工具调试：星盘", description="调用 xingpan 工具，依据姓名、出生时间、经纬度查询星盘。")
def debug_tool_xingpan(payload: ToolXingpanRequest = Body(...)):
    _ensure_debug_tools_enabled()
    result = xingpan.invoke(
        {
            "name": payload.name,
            "birth_dt": payload.birth_dt,
            "longitude": payload.longitude,
            "latitude": payload.latitude,
        }
    )
    return _wrap_tool_result("xingpan", result)


@router.post("/tools/astro/my_sign", summary="工具调试：星座信息", description="调用 astro_my_sign 工具，读取 ASTRO_UID 并查询星座信息。")
def debug_tool_astro_my_sign(payload: ToolAstroChartRequest = Body(...)):
    _ensure_debug_tools_enabled()
    result = astro_my_sign.invoke(
        {
            "birth_dt": payload.birth_dt,
            "longitude": payload.longitude,
            "latitude": payload.latitude,
        }
    )
    return _wrap_tool_result("astro_my_sign", result)


@router.post("/tools/astro/day", summary="工具调试：日运势", description="调用 astro_day_scope 工具，读取 ASTRO_UID 并查询日运势。")
def debug_tool_astro_day():
    _ensure_debug_tools_enabled()
    result = astro_day_scope.invoke({})
    return _wrap_tool_result("astro_day_scope", result)


@router.post("/tools/astro/week", summary="工具调试：周运势", description="调用 astro_week_scope 工具，读取 ASTRO_UID 并查询周运势。")
def debug_tool_astro_week():
    _ensure_debug_tools_enabled()
    result = astro_week_scope.invoke({})
    return _wrap_tool_result("astro_week_scope", result)


@router.post("/tools/astro/month", summary="工具调试：月运势", description="调用 astro_month_scope 工具，读取 ASTRO_UID 并查询月运势。")
def debug_tool_astro_month():
    _ensure_debug_tools_enabled()
    result = astro_month_scope.invoke({})
    return _wrap_tool_result("astro_month_scope", result)


@router.post("/tools/astro/natal_chart", summary="工具调试：本命盘", description="调用 astro_natal_chart 工具，依据出生时间与经纬度查询本命盘。")
def debug_tool_astro_natal(payload: ToolAstroChartRequest = Body(...)):
    _ensure_debug_tools_enabled()
    result = astro_natal_chart.invoke(
        {
            "birth_dt": payload.birth_dt,
            "longitude": payload.longitude,
            "latitude": payload.latitude,
        }
    )
    return _wrap_tool_result("astro_natal_chart", result)


@router.post("/tools/astro/current_chart", summary="工具调试：当前天象盘", description="调用 astro_current_chart 工具，查询当前天象盘。")
def debug_tool_astro_current():
    _ensure_debug_tools_enabled()
    result = astro_current_chart.invoke({})
    return _wrap_tool_result("astro_current_chart", result)


@router.post("/tools/astro/transit_chart", summary="工具调试：行运盘", description="调用 astro_transit_chart 工具，依据出生时间与经纬度查询行运盘。")
def debug_tool_astro_transit(payload: ToolAstroChartRequest = Body(...)):
    _ensure_debug_tools_enabled()
    result = astro_transit_chart.invoke(
        {
            "birth_dt": payload.birth_dt,
            "longitude": payload.longitude,
            "latitude": payload.latitude,
        }
    )
    return _wrap_tool_result("astro_transit_chart", result)


@router.get("/debug/ui", response_class=HTMLResponse, summary="调试界面", description="简易工具调试界面，仅非生产可用。")
def debug_ui():
    _ensure_debug_tools_enabled()
    return HTMLResponse(content=_DEBUG_UI_HTML)
