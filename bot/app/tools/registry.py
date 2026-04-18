from __future__ import annotations

from typing import Any

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


TOOL_REGISTRY: dict[str, Any] = {
    "search": search,
    "test": test,
    "vector_search": vector_search,
    "xingpan": xingpan,
    "astro_my_sign": astro_my_sign,
    "astro_natal_chart": astro_natal_chart,
    "astro_current_chart": astro_current_chart,
    "astro_transit_chart": astro_transit_chart,
    "astro_day_scope": astro_day_scope,
    "astro_week_scope": astro_week_scope,
    "astro_month_scope": astro_month_scope,
}

INTENT_TOOL_NAMES: dict[str, list[str]] = {
    "astro_my_sign": ["astro_my_sign", "test", "search"],
    "astro_natal_chart": ["astro_natal_chart", "xingpan", "test", "search"],
    "astro_current_chart": ["astro_current_chart", "test", "search"],
    "astro_transit_chart": ["astro_transit_chart", "xingpan", "test", "search"],
    "xingpan": ["xingpan", "astro_natal_chart", "astro_transit_chart", "test", "search"],
    "vector_search": ["vector_search", "search", "test"],
    "search": ["search", "vector_search", "test"],
}

ASTRO_TOOL_NAMES = {
    "astro_natal_chart",
    "astro_current_chart",
    "astro_transit_chart",
    "astro_my_sign",
    "xingpan",
}


def get_tool(tool_name: str):
    return TOOL_REGISTRY.get((tool_name or "").strip())


def get_tools_by_names(tool_names: list[str]) -> list[Any]:
    tools: list[Any] = []
    for name in tool_names:
        tool = get_tool(name)
        if tool is not None:
            tools.append(tool)
    return tools


def get_all_tools() -> list[Any]:
    return list(TOOL_REGISTRY.values())


def get_tools_for_intent(intent: str) -> list[Any] | None:
    names = INTENT_TOOL_NAMES.get((intent or "").strip())
    if names is None:
        return None
    return get_tools_by_names(names)
