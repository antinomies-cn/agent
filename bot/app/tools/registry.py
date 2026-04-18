from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any

from app.core.config import get_env_bool, get_env_str
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

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class ToolMeta:
    owner: str
    version: str
    risk_level: str
    idempotent: bool
    timeout_seconds: float
    retry_count: int
    debug_tier: str = "public"
    requires_env: tuple[str, ...] = ()


TOOL_METADATA: dict[str, ToolMeta] = {
    "search": ToolMeta(
        owner="platform",
        version="1.0",
        risk_level="medium",
        idempotent=True,
        timeout_seconds=8.0,
        retry_count=1,
        debug_tier="public",
        requires_env=("SERPAPI_API_KEY",),
    ),
    "test": ToolMeta(
        owner="platform",
        version="1.0",
        risk_level="low",
        idempotent=True,
        timeout_seconds=6.0,
        retry_count=0,
        debug_tier="public",
    ),
    "vector_search": ToolMeta(
        owner="platform",
        version="1.0",
        risk_level="medium",
        idempotent=True,
        timeout_seconds=10.0,
        retry_count=1,
        debug_tier="public",
    ),
    "xingpan": ToolMeta(
        owner="astro",
        version="1.0",
        risk_level="high",
        idempotent=True,
        timeout_seconds=15.0,
        retry_count=1,
        debug_tier="protected",
        requires_env=("XINGPAN_APP_ID", "XINGPAN_APP_KEY"),
    ),
    "astro_my_sign": ToolMeta(
        owner="astro",
        version="1.0",
        risk_level="high",
        idempotent=True,
        timeout_seconds=15.0,
        retry_count=1,
        debug_tier="protected",
        requires_env=("XINGPAN_APP_ID", "XINGPAN_APP_KEY", "ASTRO_UID"),
    ),
    "astro_natal_chart": ToolMeta(
        owner="astro",
        version="1.0",
        risk_level="high",
        idempotent=True,
        timeout_seconds=15.0,
        retry_count=1,
        debug_tier="protected",
        requires_env=("XINGPAN_APP_ID", "XINGPAN_APP_KEY"),
    ),
    "astro_current_chart": ToolMeta(
        owner="astro",
        version="1.0",
        risk_level="high",
        idempotent=True,
        timeout_seconds=15.0,
        retry_count=1,
        debug_tier="protected",
        requires_env=("XINGPAN_APP_ID", "XINGPAN_APP_KEY"),
    ),
    "astro_transit_chart": ToolMeta(
        owner="astro",
        version="1.0",
        risk_level="high",
        idempotent=True,
        timeout_seconds=15.0,
        retry_count=1,
        debug_tier="protected",
        requires_env=("XINGPAN_APP_ID", "XINGPAN_APP_KEY"),
    ),
    "astro_day_scope": ToolMeta(
        owner="astro",
        version="1.0",
        risk_level="medium",
        idempotent=True,
        timeout_seconds=15.0,
        retry_count=1,
        debug_tier="protected",
        requires_env=("XINGPAN_APP_ID", "XINGPAN_APP_KEY", "ASTRO_UID"),
    ),
    "astro_week_scope": ToolMeta(
        owner="astro",
        version="1.0",
        risk_level="medium",
        idempotent=True,
        timeout_seconds=15.0,
        retry_count=1,
        debug_tier="protected",
        requires_env=("XINGPAN_APP_ID", "XINGPAN_APP_KEY", "ASTRO_UID"),
    ),
    "astro_month_scope": ToolMeta(
        owner="astro",
        version="1.0",
        risk_level="medium",
        idempotent=True,
        timeout_seconds=15.0,
        retry_count=1,
        debug_tier="protected",
        requires_env=("XINGPAN_APP_ID", "XINGPAN_APP_KEY", "ASTRO_UID"),
    ),
}


def get_tool(tool_name: str):
    return TOOL_REGISTRY.get((tool_name or "").strip())


def get_tool_metadata(tool_name: str) -> ToolMeta | None:
    return TOOL_METADATA.get((tool_name or "").strip())


def get_tool_debug_access(tool_name: str) -> dict[str, Any]:
    metadata = get_tool_metadata(tool_name)
    tier = metadata.debug_tier if metadata is not None else "public"
    allow_protected = get_env_bool("TOOL_DEBUG_ALLOW_PROTECTED", default=True)
    allow_internal = get_env_bool("TOOL_DEBUG_ALLOW_INTERNAL", default=False)

    if tier == "public":
        allowed = True
    elif tier == "protected":
        allowed = allow_protected
    else:
        allowed = allow_internal

    return {
        "tier": tier,
        "allowed": allowed,
    }


def is_tool_debug_allowed(tool_name: str) -> bool:
    return bool(get_tool_debug_access(tool_name).get("allowed", False))


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
    names = get_effective_intent_tool_names().get((intent or "").strip())
    if names is None:
        return None
    return get_tools_by_names(names)


def get_effective_intent_tool_names() -> dict[str, list[str]]:
    raw = get_env_str("INTENT_TOOL_MAPPING_JSON", "").strip()
    if not raw:
        return INTENT_TOOL_NAMES

    try:
        data = json.loads(raw)
    except Exception as exc:
        logger.warning("INTENT_TOOL_MAPPING_JSON 解析失败，回退默认映射 | err: %s", str(exc)[:160])
        return INTENT_TOOL_NAMES

    if not isinstance(data, dict):
        return INTENT_TOOL_NAMES

    merged = dict(INTENT_TOOL_NAMES)
    for intent, names in data.items():
        if not isinstance(intent, str) or not isinstance(names, list):
            continue
        valid_names = [name for name in names if isinstance(name, str) and name in TOOL_REGISTRY]
        if valid_names:
            merged[intent.strip()] = valid_names
    return merged
