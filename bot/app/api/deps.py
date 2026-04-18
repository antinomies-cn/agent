from typing import Any, Optional, List

from fastapi import Body, Query

from app.schemas.add_urls import AddUrlsRequest


def resolve_runtime_dependency(name: str, default: Any) -> Any:
    try:
        from app import main as app_main
    except Exception:
        return default
    return getattr(app_main, name, default)


def build_success_response(
    data: Any = None,
    *,
    message: str = "请求成功",
    code: Any = 200,
    **legacy_fields: Any,
) -> dict:
    """构造统一成功响应，并保留历史字段兼容。"""
    payload = {
        "ok": True,
        "code": code,
        "message": message,
        "data": data,
        "error": None,
    }
    payload.update(legacy_fields)
    return payload


def build_error_response(
    *,
    error_code: str,
    message: str,
    error: str = "",
    data: Any = None,
    code: Any = 500,
    **legacy_fields: Any,
) -> dict:
    """构造统一失败响应，并保留历史字段兼容。"""
    payload = {
        "ok": False,
        "code": code,
        "message": message,
        "data": data,
        "error": error,
        "error_code": error_code,
    }
    payload.update(legacy_fields)
    return payload


def resolve_add_urls_payload(
    payload: Optional[AddUrlsRequest] = Body(default=None),
    url: Optional[str] = Query(default=None),
    urls: Optional[List[str]] = Query(default=None),
    chunk_strategy: Optional[str] = Query(default=None),
    chunk_size: Optional[int] = Query(default=None, ge=100, le=4000),
    chunk_overlap: Optional[int] = Query(default=None, ge=0, le=1000),
    separators: Optional[List[str]] = Query(default=None),
    preview_limit: Optional[int] = Query(default=None, ge=1, le=20),
) -> AddUrlsRequest:
    data = payload.model_dump() if payload is not None else {}

    existing_url = data.get("url")
    if url is not None and (existing_url is None or (isinstance(existing_url, str) and not existing_url.strip())):
        data["url"] = url
    if urls is not None and ("urls" not in data or not data.get("urls")):
        data["urls"] = urls
    if chunk_strategy is not None and data.get("chunk_strategy") is None:
        data["chunk_strategy"] = chunk_strategy
    if chunk_size is not None and data.get("chunk_size") is None:
        data["chunk_size"] = chunk_size
    if chunk_overlap is not None and data.get("chunk_overlap") is None:
        data["chunk_overlap"] = chunk_overlap
    if separators is not None and data.get("separators") is None:
        data["separators"] = separators
    if preview_limit is not None and data.get("preview_limit") is None:
        data["preview_limit"] = preview_limit

    return AddUrlsRequest(**data)