import os
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest


def _get_qdrant_client() -> QdrantClient:
    """根据环境变量创建Qdrant客户端。"""
    qdrant_url = os.getenv("QDRANT_URL", "").strip()
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip() or None
    qdrant_path = os.getenv("QDRANT_DB_PATH", "")

    if qdrant_url:
        return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    return QdrantClient(path=qdrant_path)


def init_qdrant_collection() -> dict:
    """初始化Qdrant collection（如不存在则创建）。"""
    collection_name = os.getenv("QDRANT_COLLECTION", "divination_master_collection")
    try:
        vector_size = int(os.getenv("QDRANT_VECTOR_SIZE", "384"))
    except ValueError:
        vector_size = 384

    distance_name = os.getenv("QDRANT_DISTANCE", "Cosine").strip().lower()
    distance_map = {
        "cosine": rest.Distance.COSINE,
        "dot": rest.Distance.DOT,
        "euclid": rest.Distance.EUCLID,
    }
    distance = distance_map.get(distance_name, rest.Distance.COSINE)

    client = _get_qdrant_client()

    if client.collection_exists(collection_name):
        return {
            "ok": True,
            "created": False,
            "collection": collection_name,
            "vector_size": vector_size,
            "distance": distance.name,
        }

    client.create_collection(
        collection_name=collection_name,
        vectors_config=rest.VectorParams(size=vector_size, distance=distance),
    )

    return {
        "ok": True,
        "created": True,
        "collection": collection_name,
        "vector_size": vector_size,
        "distance": distance.name,
    }


def qdrant_health() -> dict:
    """检查Qdrant连通性。"""
    client = _get_qdrant_client()
    info = client.get_collections()
    return {
        "ok": True,
        "collections_count": len(getattr(info, "collections", []) or []),
    }


def qdrant_list_collections() -> dict:
    """列出Qdrant现有collections。"""
    client = _get_qdrant_client()
    info = client.get_collections()
    collections = [c.name for c in getattr(info, "collections", []) or []]
    return {"ok": True, "collections": collections}
