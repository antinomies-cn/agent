import ipaddress
import logging
import socket
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from fastapi import HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from app.core.config import get_env_float, get_env_int, get_qdrant_settings, get_runtime_settings
from app.core.embedding_config import resolve_embedding_config
from app.core.litellm_adapters import build_litellm_embeddings_client
from app.core.logger_setup import logger
from app.schemas.add_urls import ADD_URLS_ERROR_EXPLANATIONS, AddUrlsErrorCode


def _resolve_runtime_dependency(name: str, default: Any) -> Any:
    try:
        from app import main as main_module
    except Exception:
        return default

    return getattr(main_module, name, default)


def _build_chunking_config(
    strategy: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    separators: Optional[List[str]] = None,
) -> Dict[str, Any]:
    strategy_defaults = {
        "balanced": {
            "chunk_size": 900,
            "chunk_overlap": 120,
            "separators": ["\n\n", "\n", "。", "！", "？", "；", ". ", "! ", "? ", "，", ",", " ", ""],
        },
        "faq": {
            "chunk_size": 480,
            "chunk_overlap": 80,
            "separators": ["\n\n", "\n", "。", "？", "！", "；", ". ", "? ", "! ", "，", ",", " ", ""],
        },
        "article": {
            "chunk_size": 1400,
            "chunk_overlap": 180,
            "separators": ["\n\n", "\n", "###", "##", "#", "。", "；", ". ", ",", " ", ""],
        },
        "custom": {
            "chunk_size": 800,
            "chunk_overlap": 100,
            "separators": ["\n\n", "\n", "。", ". ", " ", ""],
        },
    }

    base = strategy_defaults.get(strategy, strategy_defaults["balanced"]).copy()
    if chunk_size is not None:
        base["chunk_size"] = chunk_size
    if chunk_overlap is not None:
        base["chunk_overlap"] = chunk_overlap
    if separators:
        clean_separators = [s for s in separators if isinstance(s, str)]
        if clean_separators:
            base["separators"] = clean_separators

    if base["chunk_overlap"] >= base["chunk_size"]:
        base["chunk_overlap"] = max(0, base["chunk_size"] // 5)
    return base


def _chunk_documents(documents: List[Any], source_url: str, cfg: Dict[str, Any], strategy: str) -> List[Any]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        separators=cfg["separators"],
    )
    chunks = splitter.split_documents(documents)
    total = len(chunks)
    for idx, chunk in enumerate(chunks):
        chunk.metadata = {
            **(chunk.metadata or {}),
            "source_url": source_url,
            "chunk_strategy": strategy,
            "chunk_index": idx,
            "chunk_total": total,
        }
    return chunks


def _normalize_urls(payload) -> List[str]:
    clean_url_list = []
    if payload.url and payload.url.strip():
        clean_url_list.append(payload.url.strip())
    if payload.urls:
        clean_url_list.extend([u.strip() for u in payload.urls if isinstance(u, str) and u.strip()])
    return clean_url_list


def _get_add_urls_fetch_timeout_seconds() -> float:
    return get_env_float("ADD_URLS_FETCH_TIMEOUT_SECONDS", default=10.0, min_value=1.0)


def _get_add_urls_fetch_retry_count() -> int:
    return get_env_int("ADD_URLS_FETCH_RETRY_COUNT", default=2, min_value=0)


def _get_add_urls_fetch_backoff_seconds() -> float:
    return get_env_float("ADD_URLS_FETCH_BACKOFF_SECONDS", default=1.0, min_value=0.0)


def _get_add_urls_max_content_chars() -> int:
    return get_env_int("ADD_URLS_MAX_CONTENT_CHARS", default=20000, min_value=1)


def _get_add_urls_error_explanation(code: Optional[str]) -> str:
    clean_code = (code or "").strip()
    if clean_code in ADD_URLS_ERROR_EXPLANATIONS:
        return ADD_URLS_ERROR_EXPLANATIONS[clean_code]
    return "未知错误类型，请查看 error 字段并联系管理员排查。"


def _normalize_failed_urls(items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for item in items or []:
        url = str((item or {}).get("url", "") or "")
        code = str((item or {}).get("code", AddUrlsErrorCode.FETCH_ERROR.value) or AddUrlsErrorCode.FETCH_ERROR.value)
        error = str((item or {}).get("error", "") or "")
        explanation = str((item or {}).get("explanation", "") or "")
        normalized.append(
            {
                "url": url,
                "code": code,
                "error": error,
                "explanation": explanation or _get_add_urls_error_explanation(code),
            }
        )
    return normalized


def _is_public_http_url(raw_url: str) -> tuple[bool, Optional[AddUrlsErrorCode], str]:
    try:
        parsed = urlparse(raw_url)
    except Exception:
        return False, AddUrlsErrorCode.INVALID_URL, "URL解析失败"

    scheme = (parsed.scheme or "").lower()
    if scheme not in {"http", "https"}:
        return False, AddUrlsErrorCode.UNSUPPORTED_SCHEME, "仅支持 http/https URL"

    hostname = (parsed.hostname or "").strip()
    if not hostname:
        return False, AddUrlsErrorCode.MISSING_HOST, "URL缺少主机名"

    normalized_host = hostname.lower().rstrip(".")
    if normalized_host in {"localhost", "localhost.localdomain"}:
        return False, AddUrlsErrorCode.BLOCKED_LOOPBACK, "禁止访问环回地址"

    ip_candidates = []
    host_is_ip_literal = False
    try:
        ip_candidates.append(ipaddress.ip_address(hostname))
        host_is_ip_literal = True
    except ValueError:
        if "." not in normalized_host:
            return False, AddUrlsErrorCode.BLOCKED_INTERNAL_HOST, "禁止访问疑似内网主机名"
        try:
            infos = socket.getaddrinfo(hostname, None)
            for info in infos:
                sockaddr = info[4]
                if not sockaddr:
                    continue
                candidate_ip = sockaddr[0]
                try:
                    ip_candidates.append(ipaddress.ip_address(candidate_ip))
                except ValueError:
                    continue
        except socket.gaierror:
            return True, None, ""
        except Exception:
            return True, None, ""

    if not ip_candidates and host_is_ip_literal:
        return False, AddUrlsErrorCode.INVALID_IP, "IP地址无效"

    for ip_obj in ip_candidates:
        if ip_obj.is_loopback:
            return False, AddUrlsErrorCode.BLOCKED_LOOPBACK, "禁止访问环回地址"
        if ip_obj.is_link_local:
            return False, AddUrlsErrorCode.BLOCKED_LINK_LOCAL, "禁止访问链路本地地址"
        if ip_obj.is_private:
            return False, AddUrlsErrorCode.BLOCKED_PRIVATE_IP, "禁止访问私网地址"

    return True, None, ""


def _partition_safe_urls(clean_url_list: List[str]) -> tuple[List[str], List[Dict[str, str]]]:
    allowed_urls: List[str] = []
    blocked_urls: List[Dict[str, str]] = []
    for one_url in clean_url_list:
        ok, code, reason = _is_public_http_url(one_url)
        if ok:
            allowed_urls.append(one_url)
        else:
            blocked_urls.append(
                {
                    "url": one_url,
                    "code": (code or AddUrlsErrorCode.BLOCKED_URL).value,
                    "error": reason[:120],
                    "explanation": _get_add_urls_error_explanation((code or AddUrlsErrorCode.BLOCKED_URL).value),
                }
            )
    return allowed_urls, blocked_urls


def _ensure_add_urls_write_enabled() -> None:
    runtime_settings = get_runtime_settings()
    if runtime_settings.is_prod and not runtime_settings.add_urls_write_enabled:
        raise HTTPException(
            status_code=403,
            detail={
                "message": "生产环境默认禁用 /add_urls 入库，请显式开启开关",
                "required_env": "ADD_URLS_WRITE_ENABLED=true",
            },
        )


def _collect_chunks_from_urls(clean_url_list: List[str], chunk_cfg: Dict[str, Any], strategy: str):
    all_chunks = []
    failed_urls = []
    verify_ssl = get_runtime_settings().web_loader_verify_ssl
    fetch_timeout_seconds = _get_add_urls_fetch_timeout_seconds()
    fetch_retry_count = _get_add_urls_fetch_retry_count()
    fetch_backoff_seconds = _get_add_urls_fetch_backoff_seconds()
    max_content_chars = _get_add_urls_max_content_chars()
    web_loader_cls = _resolve_runtime_dependency("WebBaseLoader", WebBaseLoader)
    if not verify_ssl:
        logger.warning("WebBaseLoader SSL校验已关闭（WEB_LOADER_VERIFY_SSL=false），仅建议临时排障使用")

    for one_url in clean_url_list:
        last_error: Optional[Exception] = None
        documents = None
        for attempt in range(fetch_retry_count + 1):
            try:
                web_loader = web_loader_cls(
                    one_url,
                    verify_ssl=verify_ssl,
                    continue_on_failure=False,
                    raise_for_status=True,
                    requests_kwargs={"timeout": fetch_timeout_seconds},
                )
                documents = web_loader.load()
                break
            except Exception as exc:
                last_error = exc
                if attempt < fetch_retry_count:
                    sleep_seconds = fetch_backoff_seconds * (2 ** attempt)
                    logger.warning(
                        "URL抓取失败，准备重试 | url: %s | attempt: %s/%s | sleep_seconds: %.2f | err: %s",
                        one_url,
                        attempt + 1,
                        fetch_retry_count + 1,
                        sleep_seconds,
                        str(exc)[:200],
                    )
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)
                    continue

        if documents is None:
            failed_urls.append(
                {
                    "url": one_url,
                    "code": AddUrlsErrorCode.FETCH_ERROR.value,
                    "error": str(last_error)[:400] if last_error else "抓取失败",
                    "explanation": _get_add_urls_error_explanation(AddUrlsErrorCode.FETCH_ERROR.value),
                }
            )
            continue

        truncated_docs = 0
        normalized_documents = []
        for doc in documents:
            page_content = getattr(doc, "page_content", "")
            if not isinstance(page_content, str):
                page_content = str(page_content or "")
            original_length = len(page_content)
            if original_length > max_content_chars:
                page_content = page_content[:max_content_chars]
                truncated_docs += 1
            doc.page_content = page_content
            if original_length > max_content_chars:
                metadata = dict(getattr(doc, "metadata", {}) or {})
                metadata["content_truncated"] = True
                metadata["content_original_length"] = original_length
                metadata["content_max_length"] = max_content_chars
                doc.metadata = metadata
            normalized_documents.append(doc)

        if truncated_docs:
            logger.info(
                "URL内容已截断以控制最大长度 | url: %s | truncated_docs: %s | max_content_chars: %s",
                one_url,
                truncated_docs,
                max_content_chars,
            )

        all_chunks.extend(_chunk_documents(normalized_documents, one_url, chunk_cfg, strategy))
    return all_chunks, failed_urls


def _compute_chunk_quality_report(chunks: List[Any], failed_urls: List[dict], min_len: int = 120) -> Dict[str, Any]:
    total = len(chunks)
    if total == 0:
        return {
            "score": 0,
            "label": "poor",
            "stats": {
                "chunks": 0,
                "failed_urls": len(failed_urls),
            },
            "signals": ["no_chunks"],
        }

    lengths = []
    empty_chunks = 0
    short_chunks = 0
    bad_end_chunks = 0
    symbols = {",", "，", ".", "。", "!", "！", "?", "？", ";", "；", ":", "：", "…"}

    for chunk in chunks:
        content = (getattr(chunk, "page_content", "") or "").strip()
        if not content:
            empty_chunks += 1
            continue
        length = len(content)
        lengths.append(length)
        if length < min_len:
            short_chunks += 1
        if content[-1] not in symbols:
            bad_end_chunks += 1

    chunks_count = len(chunks)
    empty_ratio = empty_chunks / chunks_count
    short_ratio = short_chunks / chunks_count
    bad_end_ratio = bad_end_chunks / chunks_count

    avg_len = int(sum(lengths) / len(lengths)) if lengths else 0
    min_len_val = min(lengths) if lengths else 0
    max_len_val = max(lengths) if lengths else 0

    score = 100
    score -= int(empty_ratio * 100) * 3
    score -= int(short_ratio * 100) * 2
    score -= int(bad_end_ratio * 100) * 2
    score = max(0, min(100, score))

    if score >= 85:
        label = "good"
    elif score >= 70:
        label = "fair"
    else:
        label = "poor"

    signals = []
    if empty_ratio > 0.01:
        signals.append("empty_chunks")
    if short_ratio > 0.15:
        signals.append("too_many_short")
    if bad_end_ratio > 0.3:
        signals.append("mid_sentence_cut")
    if failed_urls:
        signals.append("fetch_failed")

    suggestions = []
    if empty_ratio > 0.01:
        suggestions.append("检查抓取来源，过滤无正文页面或广告脚本")
    if short_ratio > 0.15:
        suggestions.append("尝试提高 chunk_size 或降低 chunk_overlap")
    if bad_end_ratio > 0.3:
        suggestions.append("断句偏多，调整 separators，使切分优先落在句号/问号/换行")
    if 0.2 < bad_end_ratio <= 0.3:
        suggestions.append(
            f"bad_end_ratio={bad_end_ratio:.3f}；可将 chunk_size 上调 15%-25%，并把强分隔符(\\n\\n, \\n, 。, ！, ？)放在 separators 前部"
        )
    if min_len_val and min_len_val < 60 and short_ratio > 0.01:
        suggestions.append(f"min_len={min_len_val}；建议过滤 <80 字碎片，或适度上调 chunk_size")
    if avg_len and avg_len < 300:
        suggestions.append(f"avg_len={avg_len}；可增加 chunk_size 或做正文抽取(去导航/页脚)")
    if failed_urls:
        suggestions.append("检查失败 URL 的可访问性与 SSL 配置")
    if score < 70 and not suggestions:
        suggestions.append("整体质量偏低，建议微调 chunk_size 与 separators 再评估")

    return {
        "score": score,
        "label": label,
        "stats": {
            "chunks": chunks_count,
            "failed_urls": len(failed_urls),
            "avg_len": avg_len,
            "min_len": min_len_val,
            "max_len": max_len_val,
            "empty_ratio": round(empty_ratio, 3),
            "short_ratio": round(short_ratio, 3),
            "bad_end_ratio": round(bad_end_ratio, 3),
        },
        "signals": signals,
        "suggestions": suggestions,
    }


def _resolve_embedding_output_dim(embeddings_client: Any, default_size: int) -> int:
    dimensions = getattr(embeddings_client, "dimensions", None)
    if isinstance(dimensions, int) and dimensions > 0:
        return dimensions

    try:
        probe_vector = embeddings_client.embed_query("dimension_probe")
        if isinstance(probe_vector, (list, tuple)) and len(probe_vector) > 0:
            return len(probe_vector)
    except Exception as e:
        logger.warning("Embedding维度探测失败，使用默认值: %s | err: %s", default_size, str(e)[:120])

    return default_size


class VectorSizeMismatchError(RuntimeError):
    def __init__(self, collection_name: str, existing_size: int, expected_size: int) -> None:
        message = (
            "Qdrant collection vector size mismatch: "
            f"collection={collection_name} existing={existing_size} expected={expected_size}"
        )
        super().__init__(message)
        self.collection_name = collection_name
        self.existing_size = existing_size
        self.expected_size = expected_size


def _extract_collection_vector_size(collection_info: Any) -> Optional[int]:
    config = getattr(collection_info, "config", None)
    params = getattr(config, "params", None)
    vectors = getattr(params, "vectors", None)
    if vectors is None:
        return None

    direct_size = getattr(vectors, "size", None)
    if isinstance(direct_size, int) and direct_size > 0:
        return direct_size

    if isinstance(vectors, dict):
        for vector_cfg in vectors.values():
            size = getattr(vector_cfg, "size", None)
            if isinstance(size, int) and size > 0:
                return size

    return None


def _fetch_collection_vector_size_via_http(collection_name: str) -> Optional[int]:
    qdrant_settings = get_qdrant_settings()
    qdrant_url = qdrant_settings.url
    if not qdrant_url:
        return None

    api_key = qdrant_settings.api_key
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["api-key"] = api_key

    url = f"{qdrant_url.rstrip('/')}/collections/{collection_name}"
    try:
        resp = requests.get(url, headers=headers, timeout=6)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        logger.warning("Qdrant HTTP元数据读取失败 | collection: %s | err: %s", collection_name, str(e)[:160])
        return None

    result = payload.get("result", payload)
    config = result.get("config") if isinstance(result, dict) else None
    params = config.get("params") if isinstance(config, dict) else None
    vectors = params.get("vectors") if isinstance(params, dict) else None
    if vectors is None:
        return None

    if isinstance(vectors, dict):
        size = vectors.get("size")
        if isinstance(size, int) and size > 0:
            return size
        for vector_cfg in vectors.values():
            if isinstance(vector_cfg, dict):
                vsize = vector_cfg.get("size")
                if isinstance(vsize, int) and vsize > 0:
                    return vsize
    return None


def _recreate_collection_with_dim(client: QdrantClient, collection_name: str, vector_size: int, old_dim: Optional[int] = None) -> None:
    distance = _resolve_qdrant_distance()
    try:
        client.delete_collection(collection_name=collection_name)
    except TypeError:
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=rest.VectorParams(size=vector_size, distance=distance),
    )
    logger.warning(
        "Qdrant集合已重建 | collection: %s | old_dim: %s | new_dim: %s",
        collection_name,
        old_dim,
        vector_size,
    )


def _is_vector_dim_mismatch_error(error: Exception) -> bool:
    msg = str(error)
    return "Vector dimension error" in msg or "expected dim" in msg and "got" in msg


def _resolve_qdrant_distance() -> Any:
    distance_name = get_qdrant_settings().distance
    distance_map = {
        "cosine": rest.Distance.COSINE,
        "dot": rest.Distance.DOT,
        "euclid": rest.Distance.EUCLID,
    }
    return distance_map.get(distance_name, rest.Distance.COSINE)


def _ensure_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int) -> Optional[int]:
    try:
        exists = client.collection_exists(collection_name)
    except Exception:
        info = client.get_collections()
        collections = [c.name for c in getattr(info, "collections", []) or []]
        exists = collection_name in collections

    distance = _resolve_qdrant_distance()

    if exists:
        existing_size: Optional[int] = None
        try:
            info = client.get_collection(collection_name=collection_name)
            existing_size = _extract_collection_vector_size(info)
        except TypeError:
            info = client.get_collection(collection_name)
            existing_size = _extract_collection_vector_size(info)
        except Exception as e:
            logger.warning(
                "读取Qdrant集合元数据失败，尝试HTTP兜底 | collection: %s | err: %s",
                collection_name,
                str(e)[:160],
            )
            existing_size = _fetch_collection_vector_size_via_http(collection_name)
            if existing_size is None:
                logger.warning(
                    "HTTP兜底仍失败，跳过预检并在写入阶段兜底 | collection: %s",
                    collection_name,
                )
                return None

        if existing_size and existing_size != vector_size:
            raise VectorSizeMismatchError(
                collection_name=collection_name,
                existing_size=existing_size,
                expected_size=vector_size,
            )
        return existing_size

    client.create_collection(
        collection_name=collection_name,
        vectors_config=rest.VectorParams(size=vector_size, distance=distance),
    )
    logger.info(
        "Qdrant collection自动创建成功 | collection: %s | vector_size: %s | distance: %s",
        collection_name,
        vector_size,
        distance.name,
    )
    return vector_size


def _build_embeddings_client(vector_size: int):
    embedding_cfg = resolve_embedding_config(default_dimension=vector_size)
    return build_litellm_embeddings_client(default_dimensions=embedding_cfg.dimensions)
