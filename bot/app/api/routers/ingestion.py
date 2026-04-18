import logging
import time
from typing import Any, Callable, List, Literal, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

from app.api.deps import resolve_add_urls_payload, resolve_runtime_dependency
from app.core.config import get_qdrant_settings
from app.core.embedding_config import resolve_embedding_config
from app.core.gateway_resilience import CircuitOpenError, resilience_execute
from app.core.logger_setup import logger, log_event
from app.schemas.add_urls import (
    AddUrlsDryRunResponse,
    AddUrlsRequest,
    AddUrlsResponse,
    ChunkConfigModel,
    ChunkPreviewItem,
    FailedUrlItem,
)
from app.services import add_urls_service

router = APIRouter()


def _dep(name: str) -> Callable[..., Any]:
    mapping = {
        "_ensure_add_urls_write_enabled": add_urls_service._ensure_add_urls_write_enabled,
        "_normalize_urls": add_urls_service._normalize_urls,
        "_partition_safe_urls": add_urls_service._partition_safe_urls,
        "_normalize_failed_urls": add_urls_service._normalize_failed_urls,
        "_build_chunking_config": add_urls_service._build_chunking_config,
        "_collect_chunks_from_urls": add_urls_service._collect_chunks_from_urls,
        "_compute_chunk_quality_report": add_urls_service._compute_chunk_quality_report,
        "_build_embeddings_client": add_urls_service._build_embeddings_client,
        "_resolve_embedding_output_dim": add_urls_service._resolve_embedding_output_dim,
        "_ensure_qdrant_collection": add_urls_service._ensure_qdrant_collection,
        "_is_vector_dim_mismatch_error": add_urls_service._is_vector_dim_mismatch_error,
    }
    if name == "VectorSizeMismatchError":
        return resolve_runtime_dependency(name, add_urls_service.VectorSizeMismatchError)
    if name == "QdrantClient":
        return resolve_runtime_dependency(name, QdrantClient)
    if name == "Qdrant":
        return resolve_runtime_dependency(name, Qdrant)
    return resolve_runtime_dependency(name, mapping[name])


@router.post(
    "/add_urls",
    response_model=AddUrlsResponse,
    summary="URL学习入库",
    description="抓取URL内容、切块并写入向量库。支持 JSON 与 Query 参数。",
)
def add_urls(
    payload: AddUrlsRequest = Depends(resolve_add_urls_payload),
):
    ensure_write_enabled = _dep("_ensure_add_urls_write_enabled")
    normalize_urls = _dep("_normalize_urls")
    partition_safe_urls = _dep("_partition_safe_urls")
    normalize_failed_urls = _dep("_normalize_failed_urls")
    build_chunking_config = _dep("_build_chunking_config")
    collect_chunks_from_urls = _dep("_collect_chunks_from_urls")
    compute_chunk_quality_report = _dep("_compute_chunk_quality_report")
    build_embeddings_client = _dep("_build_embeddings_client")
    resolve_embedding_output_dim = _dep("_resolve_embedding_output_dim")
    ensure_qdrant_collection = _dep("_ensure_qdrant_collection")
    is_vector_dim_mismatch_error = _dep("_is_vector_dim_mismatch_error")
    VectorSizeMismatchError = _dep("VectorSizeMismatchError")
    qdrant_client_cls = _dep("QdrantClient")
    qdrant_store_cls = _dep("Qdrant")

    ensure_write_enabled()
    overall_start = time.perf_counter()

    clean_url_list = normalize_urls(payload)
    if not clean_url_list:
        raise HTTPException(status_code=400, detail="请提供url或urls参数")

    clean_url_list, blocked_urls = partition_safe_urls(clean_url_list)
    blocked_urls = normalize_failed_urls(blocked_urls)
    if not clean_url_list:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "URL全部被安全策略拦截",
                "failed_urls": blocked_urls,
            },
        )

    chunk_cfg = build_chunking_config(
        strategy=payload.chunk_strategy,
        chunk_size=payload.chunk_size,
        chunk_overlap=payload.chunk_overlap,
        separators=payload.separators,
    )
    all_chunks, failed_urls = collect_chunks_from_urls(clean_url_list, chunk_cfg, payload.chunk_strategy)
    failed_urls = normalize_failed_urls([*blocked_urls, *failed_urls])
    quality_report = compute_chunk_quality_report(all_chunks, failed_urls)

    if not all_chunks:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "URL内容加载失败，未生成可写入的文本块",
                "failed_urls": failed_urls,
            },
        )

    qdrant_settings = get_qdrant_settings()
    qdrant_url = qdrant_settings.url
    qdrant_api_key = qdrant_settings.api_key or None
    qdrant_path = qdrant_settings.path
    collection_name = qdrant_settings.collection
    embedding_cfg = resolve_embedding_config()
    vector_size = embedding_cfg.dimensions

    if qdrant_url:
        client = qdrant_client_cls(url=qdrant_url, api_key=qdrant_api_key)
        mode = "remote"
    else:
        client = qdrant_client_cls(path=qdrant_path)
        mode = "local"

    try:
        embeddings_client = build_embeddings_client(vector_size=vector_size)
        effective_vector_size = resolve_embedding_output_dim(embeddings_client, default_size=vector_size)

        try:
            resilience_execute(
                component="qdrant",
                operation="ensure_collection",
                func=lambda: ensure_qdrant_collection(
                    client=client,
                    collection_name=collection_name,
                    vector_size=effective_vector_size,
                ),
            )
        except VectorSizeMismatchError as mismatch_err:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "Qdrant collection vector size mismatch",
                    "collection": mismatch_err.collection_name,
                    "existing_size": mismatch_err.existing_size,
                    "expected_size": mismatch_err.expected_size,
                    "hint": "请调整 EMBEDDINGS_DIMENSION 与集合一致，或调用 /qdrant/recreate 重建集合。",
                },
            )

        vector_store = qdrant_store_cls(client, collection_name, embeddings_client)
        try:
            resilience_execute(
                component="qdrant",
                operation="add_documents",
                func=lambda: vector_store.add_documents(all_chunks),
            )
        except Exception as write_err:
            if is_vector_dim_mismatch_error(write_err):
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": "Qdrant collection vector size mismatch",
                        "collection": collection_name,
                        "expected_size": effective_vector_size,
                        "hint": "请调整 EMBEDDINGS_DIMENSION 与集合一致，或调用 /qdrant/recreate 重建集合。",
                    },
                )
            if isinstance(write_err, CircuitOpenError):
                raise HTTPException(
                    status_code=503,
                    detail={
                        "message": "Qdrant 熔断保护中，请稍后重试",
                        "error_code": "QDRANT_CIRCUIT_OPEN",
                        "retry_after_seconds": write_err.retry_after_seconds,
                    },
                )
            raise
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "向量写入Qdrant失败",
                "error": str(e)[:400],
                "mode": mode,
                "collection": collection_name,
            },
        )

    log_event(
        logging.INFO,
        "add_urls.done",
        mode=mode,
        collection=collection_name,
        url_count=len(clean_url_list),
        chunks=len(all_chunks),
        failed=len(failed_urls),
        elapsed_ms=int((time.perf_counter() - overall_start) * 1000),
    )
    return AddUrlsResponse(
        response="URLs added!",
        collection=collection_name,
        mode=mode,
        source_urls=len(clean_url_list),
        chunks=len(all_chunks),
        failed_urls=[FailedUrlItem(**item) for item in failed_urls],
        chunk_strategy=payload.chunk_strategy,
        chunk_config=ChunkConfigModel(
            chunk_size=chunk_cfg["chunk_size"],
            chunk_overlap=chunk_cfg["chunk_overlap"],
            separators=chunk_cfg["separators"],
        ),
        quality_report=quality_report,
    )


@router.post(
    "/add_urls/dry_run",
    response_model=AddUrlsDryRunResponse,
    summary="URL切块预览",
    description="仅抓取与切块预览，不写入向量库。支持 JSON 与 Query 参数。",
)
def add_urls_dry_run(
    payload: AddUrlsRequest = Depends(resolve_add_urls_payload),
):
    normalize_urls = _dep("_normalize_urls")
    partition_safe_urls = _dep("_partition_safe_urls")
    normalize_failed_urls = _dep("_normalize_failed_urls")
    build_chunking_config = _dep("_build_chunking_config")
    collect_chunks_from_urls = _dep("_collect_chunks_from_urls")
    compute_chunk_quality_report = _dep("_compute_chunk_quality_report")

    overall_start = time.perf_counter()
    clean_url_list = normalize_urls(payload)
    if not clean_url_list:
        raise HTTPException(status_code=400, detail="请提供url或urls参数")

    clean_url_list, blocked_urls = partition_safe_urls(clean_url_list)
    blocked_urls = normalize_failed_urls(blocked_urls)
    if not clean_url_list:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "URL全部被安全策略拦截",
                "failed_urls": blocked_urls,
            },
        )

    chunk_cfg = build_chunking_config(
        strategy=payload.chunk_strategy,
        chunk_size=payload.chunk_size,
        chunk_overlap=payload.chunk_overlap,
        separators=payload.separators,
    )
    all_chunks, failed_urls = collect_chunks_from_urls(clean_url_list, chunk_cfg, payload.chunk_strategy)
    failed_urls = normalize_failed_urls([*blocked_urls, *failed_urls])
    quality_report = compute_chunk_quality_report(all_chunks, failed_urls)

    if not all_chunks:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "URL内容加载失败，未生成可预览的文本块",
                "failed_urls": failed_urls,
            },
        )

    preview_limit = min(payload.preview_limit, len(all_chunks))
    chunk_preview = []
    for chunk in all_chunks[:preview_limit]:
        content = (chunk.page_content or "").strip()
        chunk_preview.append(
            {
                "source_url": (chunk.metadata or {}).get("source_url", ""),
                "chunk_index": (chunk.metadata or {}).get("chunk_index", 0),
                "content_length": len(content),
                "content_preview": content[:200],
            }
        )

    log_event(
        logging.INFO,
        "add_urls.dry_run.done",
        url_count=len(clean_url_list),
        chunks=len(all_chunks),
        failed=len(failed_urls),
        elapsed_ms=int((time.perf_counter() - overall_start) * 1000),
    )
    return AddUrlsDryRunResponse(
        response="Dry run completed",
        source_urls=len(clean_url_list),
        chunks=len(all_chunks),
        failed_urls=[FailedUrlItem(**item) for item in failed_urls],
        chunk_strategy=payload.chunk_strategy,
        chunk_config=ChunkConfigModel(
            chunk_size=chunk_cfg["chunk_size"],
            chunk_overlap=chunk_cfg["chunk_overlap"],
            separators=chunk_cfg["separators"],
        ),
        chunk_preview=[ChunkPreviewItem(**item) for item in chunk_preview],
        quality_report=quality_report,
    )


@router.post("/add_pdfs", summary="PDF入库占位", description="占位接口：未来支持PDF解析与入库。")
def add_pdfs():
    logger.info("调用add_pdfs接口")
    return {"response": "PDFs added!"}


@router.post("/add_texts", summary="文本入库占位", description="占位接口：未来支持直接文本入库。")
def add_texts():
    logger.info("调用add_texts接口")
    return {"response": "Texts added!"}
