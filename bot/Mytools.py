import os
import json
import logging
import threading
import requests
from dotenv import load_dotenv
from langchain.agents import tool
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings

load_dotenv()

logger = logging.getLogger(__name__)

# ===================== retriever初始化 =====================
_vector_retriever = None
_vector_retriever_lock = threading.Lock()


def _astro_base_url() -> str:
    """获取星盘服务基础地址。"""
    return os.getenv("XINGPAN_API_URL", "https://cloud.apiworks.com/open/astro").rstrip("/")


def _request_astro_api(path: str, method: str = "POST", payload: dict | None = None) -> str:
    """统一封装星盘接口调用。"""
    api_url = f"{_astro_base_url()}/{path.lstrip('/')}"
    app_id = os.getenv("XINGPAN_APP_ID", "")
    app_key = os.getenv("XINGPAN_APP_KEY", "")
    try:
        timeout_seconds = float(os.getenv("XINGPAN_TIMEOUT", "15"))
    except ValueError:
        timeout_seconds = 15.0

    if not app_id or not app_key:
        logger.error("星盘工具配置缺失 | XINGPAN_APP_ID/XINGPAN_APP_KEY 未设置")
        return "星盘服务暂未配置，请联系管理员设置 XINGPAN_APP_ID 和 XINGPAN_APP_KEY。"

    req_body = payload or {}
    headers = {
        "X-App-Id": app_id,
        "X-App-Key": app_key,
        "Content-Type": "application/json",
    }

    try:
        m = (method or "POST").upper()
        if m == "GET":
            resp = requests.get(api_url, headers=headers, timeout=max(1.0, timeout_seconds))
        else:
            resp = requests.post(api_url, headers=headers, json=req_body, timeout=max(1.0, timeout_seconds))
        resp.raise_for_status()

        data = resp.json()
        logger.info("执行星盘接口成功 | method: %s | path: %s", m, path)
        return json.dumps(data, ensure_ascii=False)
    except requests.exceptions.Timeout:
        logger.error("星盘接口超时 | method: %s | path: %s", method, path)
        return "星盘服务响应超时，请稍后再试。"
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else 0
        body = ""
        try:
            body = (e.response.text or "")[:300] if e.response is not None else ""
        except Exception:
            body = ""
        logger.error(
            "星盘接口HTTP错误 | status: %s | method: %s | path: %s | body: %s",
            status_code,
            method,
            path,
            body,
        )
        return f"星盘服务调用失败（HTTP {status_code}），请稍后重试。"
    except Exception as e:
        logger.error(
            "星盘接口异常 | method: %s | path: %s | error: %s",
            method,
            path,
            str(e)[:200],
            exc_info=True,
        )
        return "星盘服务暂时不可用，请稍后再试。"


def _call_xingpan_api(name: str, birth_dt: str, lng: float, lat: float) -> str:
    """兼容旧工具：按本命盘接口调用。"""
    payload = {
        "birth_dt": birth_dt,
        "longitude": lng,
        "latitude": lat,
    }
    result = _request_astro_api("chart/natal", method="POST", payload=payload)
    return (
        f"姓名：{name}\n"
        f"出生时间：{birth_dt}\n"
        f"经纬度：({lng}, {lat})\n"
        f"星盘原始结果：{result}"
    )


def _get_vector_retriever():
    """惰性初始化向量检索器，避免每次请求重复创建客户端和Embedding。"""
    global _vector_retriever
    if _vector_retriever is not None:
        return _vector_retriever

    with _vector_retriever_lock:
        if _vector_retriever is not None:
            return _vector_retriever

        db_path = os.getenv("QDRANT_DB_PATH", "./qdrant_data/qdrant.db")
        collection_name = os.getenv("QDRANT_COLLECTION", "divination_master_collection")
        try:
            top_k = int(os.getenv("VECTOR_SEARCH_TOP_K", "4"))
        except ValueError:
            top_k = 4

        vector_store = Qdrant(
            QdrantClient(path=db_path),
            collection_name,
            OpenAIEmbeddings(),
        )
        _vector_retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": max(1, top_k)},
        )
        logger.info(
            "向量检索器初始化完成 | db_path: %s | collection: %s | top_k: %s",
            db_path,
            collection_name,
            max(1, top_k),
        )
        return _vector_retriever

# ===================== 工具定义 =====================
@tool
def test(scope: str = "all") -> str:
    """系统自检工具。

    参数:
    - scope: all|astro|vector|search
    """
    clean_scope = (scope or "all").strip().lower()
    if clean_scope not in {"all", "astro", "vector", "search"}:
        return "scope 参数仅支持 all|astro|vector|search。"

    report = {
        "scope": clean_scope,
        "ok": True,
        "checks": {},
    }

    if clean_scope in {"all", "astro"}:
        astro_ok = True
        missing = []
        if not os.getenv("XINGPAN_APP_ID", "").strip():
            missing.append("XINGPAN_APP_ID")
            astro_ok = False
        if not os.getenv("XINGPAN_APP_KEY", "").strip():
            missing.append("XINGPAN_APP_KEY")
            astro_ok = False

        report["checks"]["astro"] = {
            "ok": astro_ok,
            "base_url": _astro_base_url(),
            "missing_env": missing,
        }
        report["ok"] = report["ok"] and astro_ok

    if clean_scope in {"all", "vector"}:
        vector_ok = True
        vector_error = ""
        try:
            _ = _get_vector_retriever()
        except Exception as e:
            vector_ok = False
            vector_error = str(e)[:200]

        report["checks"]["vector"] = {
            "ok": vector_ok,
            "collection": os.getenv("QDRANT_COLLECTION", "divination_master_collection"),
            "db_path": os.getenv("QDRANT_DB_PATH", "./qdrant_data/qdrant.db"),
            "error": vector_error,
        }
        report["ok"] = report["ok"] and vector_ok

    if clean_scope in {"all", "search"}:
        search_ok = bool(os.getenv("SERPAPI_API_KEY", "").strip())
        report["checks"]["search"] = {
            "ok": search_ok,
            "missing_env": [] if search_ok else ["SERPAPI_API_KEY"],
        }
        report["ok"] = report["ok"] and search_ok

    logger.info("执行测试工具 | scope: %s | ok: %s", clean_scope, report["ok"])
    return json.dumps(report, ensure_ascii=False)

@tool
def search(query: str) -> str:
    """搜索工具:只有当你需要获取最新信息或者查询事实时才使用这个工具。"""
    serp = SerpAPIWrapper()
    result = serp.run(query)
    logger.info(f"执行搜索工具 | 查询: {query[:50]} | 结果: {result[:100]}")
    return result

@tool
def vector_search(query: str) -> str:
    """向量搜索工具:只有当你需要从已知文档中查询信息时才使用这个工具。"""
    clean_query = (query or "").strip()
    if not clean_query:
        return "请输入需要检索的问题。"

    try:
        retriever = _get_vector_retriever()
        docs = retriever.invoke(clean_query)
        result = "\n\n".join(
            [doc.page_content for doc in docs if getattr(doc, "page_content", "")]
        ).strip()

        if not result:
            logger.info(f"执行向量搜索工具 | 查询: {clean_query[:50]} | 命中: 0")
            return "未检索到相关内容。"

        logger.info(f"执行向量搜索工具 | 查询: {clean_query[:50]} | 结果长度: {len(result)}")
        return result
    except Exception as e:
        logger.error(f"向量搜索工具异常 | 查询: {clean_query[:50]} | 错误: {str(e)[:100]}", exc_info=True)
        return "向量检索暂时不可用，请稍后重试。"

@tool
def xingpan(name: str, birth_dt: str, longitude: float, latitude: float) -> str:
    """星盘工具，只有做星盘相关的占卜时才使用。

    需要用户姓名、出生时间（YYYY-MM-DD HH:MM:SS）、出生地经纬度。
    """

    clean_name = (name or "").strip()
    clean_birth_dt = (birth_dt or "").strip()

    if not clean_name:
        return "请先提供姓名。"
    if not clean_birth_dt:
        return "请先提供出生时间，格式示例：1999-10-17 21:00:00。"

    try:
        lng = float(longitude)
        lat = float(latitude)
    except (TypeError, ValueError):
        return "经纬度格式有误，请提供数字类型的 longitude 和 latitude。"

    return _call_xingpan_api(clean_name, clean_birth_dt, lng, lat)


@tool
def astro_my_sign() -> str:
    """星座信息工具：当用户查询星座信息时使用。uid 从 .env 读取。"""
    uid = os.getenv("ASTRO_UID", "") or os.getenv("UID", "")
    uid = uid.strip()
    if not uid:
        return "请在 .env 中配置 ASTRO_UID（或 UID）后再查询星座信息。"
    return _request_astro_api(f"mySign/{uid}", method="GET")


@tool
def astro_natal_chart(birth_dt: str, longitude: float, latitude: float) -> str:
    """本命盘查询工具：当用户要查本命盘时使用。"""
    clean_birth_dt = (birth_dt or "").strip()
    if not clean_birth_dt:
        return "请先提供出生时间，格式示例：1999-10-17 21:00:00。"
    try:
        lng = float(longitude)
        lat = float(latitude)
    except (TypeError, ValueError):
        return "经纬度格式有误，请提供数字类型的 longitude 和 latitude。"

    payload = {
        "birth_dt": clean_birth_dt,
        "longitude": lng,
        "latitude": lat,
    }
    return _request_astro_api("chart/natal", method="POST", payload=payload)


@tool
def astro_current_chart() -> str:
    """天象盘查询工具：当用户要查当前天象盘时使用。"""
    return _request_astro_api("chart/current", method="POST", payload={})


@tool
def astro_transit_chart(birth_dt: str, longitude: float, latitude: float) -> str:
    """行运盘查询工具：当用户要查行运盘时使用。"""
    clean_birth_dt = (birth_dt or "").strip()
    if not clean_birth_dt:
        return "请先提供出生时间，格式示例：1999-10-17 21:00:00。"
    try:
        lng = float(longitude)
        lat = float(latitude)
    except (TypeError, ValueError):
        return "经纬度格式有误，请提供数字类型的 longitude 和 latitude。"

    payload = {
        "birth_dt": clean_birth_dt,
        "longitude": lng,
        "latitude": lat,
    }
    return _request_astro_api("chart/transit", method="POST", payload=payload)


