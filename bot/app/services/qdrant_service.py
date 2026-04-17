import os
import time
import logging
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from app.core.config import get_qdrant_settings
from app.core.embedding_config import resolve_embedding_config
from app.core.logger_setup import logger, log_event


def _invalidate_vector_retriever_cache() -> bool:
    """尽力失效进程内向量检索器缓存，返回是否检测到已有缓存。"""
    try:
        # 延迟导入，避免服务层与工具层形成模块初始化循环依赖。
        from app.tools.mytools import reset_vector_retriever_cache
    except Exception as e:
        logger.warning("向量检索器缓存失效导入失败 | err: %s", str(e)[:160])
        return False

    try:
        return bool(reset_vector_retriever_cache())
    except Exception as e:
        logger.warning("向量检索器缓存失效执行失败 | err: %s", str(e)[:160], exc_info=True)
        return False


def _get_qdrant_client() -> QdrantClient:
    """根据环境变量创建Qdrant客户端。"""
    settings = get_qdrant_settings()
    qdrant_url = settings.url
    qdrant_api_key = settings.api_key or None
    qdrant_path = settings.path

    if qdrant_url:
        return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    return QdrantClient(path=qdrant_path)


def _resolve_vector_size(default_value: int = 1024) -> int:
    """建库维度统一走 EmbeddingConfig 策略。"""
    return resolve_embedding_config(default_dimension=default_value).dimensions


def _resolve_distance() -> rest.Distance:
    distance_name = get_qdrant_settings().distance
    distance_map = {
        "cosine": rest.Distance.COSINE,
        "dot": rest.Distance.DOT,
        "euclid": rest.Distance.EUCLID,
    }
    return distance_map.get(distance_name, rest.Distance.COSINE)


def init_qdrant_collection(collection_name: Optional[str] = None, force_recreate: bool = False) -> dict:
    """初始化Qdrant collection；可按需删除旧集合并重建。"""
    start_time = time.perf_counter()
    target_collection = (collection_name or get_qdrant_settings().collection).strip()
    if not target_collection:
        target_collection = "divination_master_collection"

    vector_size = _resolve_vector_size()
    distance = _resolve_distance()
    client = _get_qdrant_client()

    try:
        try:
            exists = client.collection_exists(target_collection)
        except Exception:
            info = client.get_collections()
            exists = target_collection in [c.name for c in getattr(info, "collections", []) or []]

        deleted = False
        if exists and force_recreate:
            try:
                client.delete_collection(collection_name=target_collection)
            except TypeError:
                client.delete_collection(target_collection)
            deleted = True
            exists = False

        retriever_cache_reset_attempted = False
        retriever_cache_had_cache = False

        if exists:
            result = {
                "ok": True,
                "created": False,
                "deleted": False,
                "collection": target_collection,
                "vector_size": vector_size,
                "distance": distance.name,
                "retriever_cache_reset_attempted": False,
                "retriever_cache_had_cache": False,
            }
        else:
            client.create_collection(
                collection_name=target_collection,
                vectors_config=rest.VectorParams(size=vector_size, distance=distance),
            )
            retriever_cache_reset_attempted = True
            retriever_cache_had_cache = _invalidate_vector_retriever_cache()
            result = {
                "ok": True,
                "created": True,
                "deleted": deleted,
                "collection": target_collection,
                "vector_size": vector_size,
                "distance": distance.name,
                "retriever_cache_reset_attempted": retriever_cache_reset_attempted,
                "retriever_cache_had_cache": retriever_cache_had_cache,
            }

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        log_event(
            logging.INFO,
            "qdrant.init",
            collection=target_collection,
            created=result["created"],
            deleted=result["deleted"],
            vector_size=vector_size,
            distance=distance.name,
            retriever_cache_reset_attempted=result["retriever_cache_reset_attempted"],
            retriever_cache_had_cache=result["retriever_cache_had_cache"],
            elapsed_ms=elapsed_ms,
        )
        return result
    except Exception as e:
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        logger.error("Qdrant初始化异常 | collection: %s | err: %s", target_collection, str(e)[:160], exc_info=True)
        log_event(
            logging.ERROR,
            "qdrant.init.error",
            collection=target_collection,
            vector_size=vector_size,
            distance=distance.name,
            elapsed_ms=elapsed_ms,
            error=str(e)[:200],
        )
        raise


def recreate_qdrant_collection(collection_name: Optional[str] = None) -> dict:
    """删除并重建指定 collection。"""
    return init_qdrant_collection(collection_name=collection_name, force_recreate=True)


def qdrant_health() -> dict:
    """检查Qdrant连通性。"""
    start_time = time.perf_counter()
    client = _get_qdrant_client()
    info = client.get_collections()
    result = {
        "ok": True,
        "collections_count": len(getattr(info, "collections", []) or []),
    }
    elapsed_ms = int((time.perf_counter() - start_time) * 1000)
    log_event(
        logging.INFO,
        "qdrant.health",
        collections_count=result["collections_count"],
        elapsed_ms=elapsed_ms,
    )
    return result


def qdrant_list_collections() -> dict:
    """列出Qdrant现有collections。"""
    start_time = time.perf_counter()
    client = _get_qdrant_client()
    info = client.get_collections()
    collections = [c.name for c in getattr(info, "collections", []) or []]
    elapsed_ms = int((time.perf_counter() - start_time) * 1000)
    log_event(
        logging.INFO,
        "qdrant.collections",
        collections_count=len(collections),
        elapsed_ms=elapsed_ms,
    )
    return {"ok": True, "collections": collections}


def qdrant_repo_status() -> dict:
    """简单监视Qdrant仓库状态，不改变任何配置。"""
    start_time = time.perf_counter()
    qdrant_settings = get_qdrant_settings()
    qdrant_url = qdrant_settings.url
    qdrant_path = qdrant_settings.path
    mode = "remote" if qdrant_url else "local"

    client = _get_qdrant_client()
    info = client.get_collections()
    collections = [c.name for c in getattr(info, "collections", []) or []]

    status = {
        "ok": True,
        "mode": mode,
        "collections_count": len(collections),
        "collections": collections,
    }

    if mode == "remote":
        status["qdrant_url"] = qdrant_url
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        log_event(
            logging.INFO,
            "qdrant.status",
            mode=mode,
            collections_count=status["collections_count"],
            elapsed_ms=elapsed_ms,
        )
        return status

    db_exists = os.path.exists(qdrant_path)
    db_size = os.path.getsize(qdrant_path) if db_exists else 0
    status.update(
        {
            "qdrant_path": qdrant_path,
            "db_exists": db_exists,
            "db_size_bytes": db_size,
        }
    )
    elapsed_ms = int((time.perf_counter() - start_time) * 1000)
    log_event(
        logging.INFO,
        "qdrant.status",
        mode=mode,
        collections_count=status["collections_count"],
        db_exists=db_exists,
        db_size_bytes=db_size,
        elapsed_ms=elapsed_ms,
    )
    return status
