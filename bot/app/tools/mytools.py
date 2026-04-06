import os
import json
import time
import importlib.util
import logging
import threading
import requests
from datetime import datetime
from langchain.agents import tool
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from app.core import config as _app_config
from app.core.logger_setup import log_event

logger = logging.getLogger(__name__)

# ===================== retriever初始化 =====================
_vector_retriever = None
_vector_retriever_lock = threading.Lock()


def _resolve_embeddings_dimension(default_value: int) -> int:
    raw = os.getenv("EMBEDDINGS_DIMENSION", "").strip()
    if not raw:
        return default_value
    try:
        value = int(raw)
        if value > 0:
            return value
    except ValueError:
        pass
    logger.warning("EMBEDDINGS_DIMENSION 非法，已回退默认值: %s", default_value)
    return default_value


def _resolve_local_embedding_model(model_name: str, cache_dir: str) -> str:
    """解析本地Embedding模型路径，兼容仓库名与本地目录。"""
    clean_model = (model_name or "").strip()
    if not clean_model:
        return clean_model

    if os.path.isdir(clean_model):
        return clean_model

    if cache_dir and "/" in clean_model:
        candidate = os.path.join(cache_dir, *clean_model.split("/"))
        if os.path.isdir(candidate):
            return candidate

    return clean_model


def _normalize_openai_base_url(raw_base_url: str) -> str:
    """规范化 OPENAI_API_BASE，确保配置不包含 /v1。"""
    base = (raw_base_url or "").strip().rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base.rstrip("/")


def _build_embeddings_client(default_dimensions: int = 384):
    """按 EMBEDDINGS_API 构造 Embeddings 客户端。"""
    provider = os.getenv("EMBEDDINGS_API", "openai").strip().lower()
    embedding_model = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small").strip() or "text-embedding-3-small"

    if provider == "local":
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except Exception as e:
            raise RuntimeError("EMBEDDINGS_API=local 需要安装 sentence-transformers") from e
        hf_endpoint = os.getenv("EMBEDDINGS_HF_ENDPOINT", "").strip()
        local_files_only = os.getenv("EMBEDDINGS_LOCAL_FILES_ONLY", "false").strip().lower() in {"1", "true", "yes", "on"}
        cache_dir = os.getenv("EMBEDDINGS_CACHE_DIR", "").strip()
        resolved_model = _resolve_local_embedding_model(embedding_model, cache_dir)

        # 仅在 Embedding 初始化路径下设置 HF 运行时环境，避免影响其他模块。
        if hf_endpoint and not local_files_only:
            os.environ["HF_ENDPOINT"] = hf_endpoint.rstrip("/")
        if local_files_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if cache_dir:
            os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_dir

        model_kwargs = {"local_files_only": local_files_only}

        try:
            client = HuggingFaceEmbeddings(
                model_name=resolved_model,
                cache_folder=(cache_dir or None),
                model_kwargs=model_kwargs,
                encode_kwargs={"normalize_embeddings": True},
            )
            log_event(
                logging.INFO,
                "embeddings.init",
                provider="local",
                model=resolved_model,
            )
            return client
        except Exception as e:
            expected_path = os.path.join(cache_dir, "BAAI", "bge-small-zh-v1.5") if cache_dir else ""
            hint = (
                "本地模型不可用。请先预下载模型到缓存目录，"
                "或设置 EMBEDDINGS_HF_ENDPOINT 到可访问镜像站，"
                "或将 EMBEDDINGS_API 切回 openai。"
            )
            raise RuntimeError(
                f"初始化本地Embedding失败: {str(e)[:240]} | model={resolved_model} | expected_dir={expected_path} | {hint}"
            ) from e

    api_base = _normalize_openai_base_url(os.getenv("OPENAI_API_BASE", ""))
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    kwargs = {
        "model": embedding_model,
        "dimensions": _resolve_embeddings_dimension(default_dimensions),
        # 某些代理模型不在 tiktoken 映射表中，固定编码可避免无意义告警。
        "tiktoken_model_name": "cl100k_base",
    }
    if api_base:
        kwargs["openai_api_base"] = f"{api_base}/v1"

    if api_key:
        kwargs["openai_api_key"] = api_key

    client = OpenAIEmbeddings(**kwargs)
    log_event(
        logging.INFO,
        "embeddings.init",
        provider="openai",
        model=embedding_model,
        dimensions=kwargs.get("dimensions"),
    )
    return client


def _tool_result(ok: bool, data=None, error: str = "", code: str = "OK") -> str:
    """统一工具返回结构，便于上层稳定解析。"""
    return json.dumps(
        {
            "ok": ok,
            "code": code,
            "data": data,
            "error": error,
        },
        ensure_ascii=False,
    )


def _normalize_birth_dt(raw: str) -> str:
    """将出生时间统一归一化为 %Y-%m-%d %H:%M:%S。"""
    text = (raw or "").strip()
    if not text:
        return ""

    # 兼容 ISO8601 的 Z 时区表达
    text = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass

    # 已是目标格式时直接返回
    try:
        dt = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return text


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
        return _tool_result(
            ok=False,
            code="CONFIG_MISSING",
            error="星盘服务暂未配置，请联系管理员设置 XINGPAN_APP_ID 和 XINGPAN_APP_KEY。",
        )

    req_body = payload or {}
    headers = {
        "X-App-Id": app_id,
        "X-App-Key": app_key,
        "Content-Type": "application/json",
    }

    start_time = time.perf_counter()
    try:
        m = (method or "POST").upper()
        if m == "GET":
            resp = requests.get(api_url, headers=headers, timeout=max(1.0, timeout_seconds))
        else:
            resp = requests.post(api_url, headers=headers, json=req_body, timeout=max(1.0, timeout_seconds))
        resp.raise_for_status()

        data = resp.json()
        logger.info("执行星盘接口成功 | method: %s | path: %s", m, path)
        log_event(
            logging.INFO,
            "astro.request",
            method=m,
            path=path,
            status_code=resp.status_code,
            elapsed_ms=int((time.perf_counter() - start_time) * 1000),
        )
        return _tool_result(ok=True, code="OK", data=data)
    except requests.exceptions.Timeout:
        logger.error("星盘接口超时 | method: %s | path: %s", method, path)
        log_event(
            logging.ERROR,
            "astro.request.timeout",
            method=method,
            path=path,
            elapsed_ms=int((time.perf_counter() - start_time) * 1000),
        )
        return _tool_result(ok=False, code="TIMEOUT", error="星盘服务响应超时，请稍后再试。")
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
        log_event(
            logging.ERROR,
            "astro.request.http_error",
            method=method,
            path=path,
            status_code=status_code,
            elapsed_ms=int((time.perf_counter() - start_time) * 1000),
        )
        return _tool_result(
            ok=False,
            code=f"HTTP_{status_code}",
            error=f"星盘服务调用失败（HTTP {status_code}），请稍后重试。",
            data={"status_code": status_code, "body": body},
        )
    except Exception as e:
        logger.error(
            "星盘接口异常 | method: %s | path: %s | error: %s",
            method,
            path,
            str(e)[:200],
            exc_info=True,
        )
        log_event(
            logging.ERROR,
            "astro.request.error",
            method=method,
            path=path,
            error=str(e)[:200],
            elapsed_ms=int((time.perf_counter() - start_time) * 1000),
        )
        return _tool_result(ok=False, code="UNKNOWN_ERROR", error="星盘服务暂时不可用，请稍后再试。")


def _call_xingpan_api(name: str, birth_dt: str, lng: float, lat: float) -> str:
    """兼容旧工具：按本命盘接口调用。"""
    normalized_birth_dt = _normalize_birth_dt(birth_dt)
    payload = {
        "birth_dt": normalized_birth_dt,
        "longitude": lng,
        "latitude": lat,
    }
    result = _request_astro_api("chart/natal", method="POST", payload=payload)
    return (
        f"姓名：{name}\n"
        f"出生时间：{normalized_birth_dt}\n"
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

        qdrant_url = os.getenv("QDRANT_URL", "").strip()
        qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip()
        db_path = os.getenv("QDRANT_DB_PATH", "./qdrant_data/qdrant.db")
        collection_name = os.getenv("QDRANT_COLLECTION", "divination_master_collection")
        try:
            top_k = int(os.getenv("VECTOR_SEARCH_TOP_K", "4"))
        except ValueError:
            top_k = 4

        init_start = time.perf_counter()
        if qdrant_url:
            client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key or None)
            mode = "remote"
        else:
            client = QdrantClient(path=db_path)
            mode = "local"

        vector_store = Qdrant(
            client,
            collection_name,
            _build_embeddings_client(default_dimensions=384),
        )
        _vector_retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": max(1, top_k)},
        )
        logger.info(
            "向量检索器初始化完成 | mode: %s | db_path: %s | url: %s | collection: %s | top_k: %s",
            mode,
            db_path,
            qdrant_url or "",
            collection_name,
            max(1, top_k),
        )
        log_event(
            logging.INFO,
            "vector_retriever.init",
            mode=mode,
            collection=collection_name,
            top_k=max(1, top_k),
            elapsed_ms=int((time.perf_counter() - init_start) * 1000),
        )
        return _vector_retriever

# ===================== 工具定义 =====================
@tool
def test(scope: str = "all") -> str:
    """系统自检工具。

    参数:
    - scope: all|astro|vector|search
    """
    start = time.perf_counter()
    clean_scope = (scope or "all").strip().lower()
    if clean_scope not in {"all", "astro", "vector", "search"}:
        return "scope 参数仅支持 all|astro|vector|search。"

    report = {
        "scope": clean_scope,
        "ok": True,
        "checks": {},
        "meta": {
            "tool": "test",
            "version": "1.1",
        },
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

        uid = (os.getenv("ASTRO_UID", "") or os.getenv("UID", "")).strip()
        timeout_text = os.getenv("XINGPAN_TIMEOUT", "15")
        try:
            timeout_seconds = float(timeout_text)
        except ValueError:
            timeout_seconds = 15.0

        probe_ok = False
        probe_msg = ""
        if astro_ok and uid:
            try:
                probe_raw = _request_astro_api(f"mySign/{uid}", method="GET")
                probe_data = json.loads(probe_raw)
                probe_ok = bool(probe_data.get("ok"))
                probe_msg = probe_data.get("code", "") if isinstance(probe_data, dict) else ""
                astro_ok = astro_ok and probe_ok
            except Exception as e:
                probe_msg = str(e)[:120]
                astro_ok = False
        elif astro_ok and not uid:
            probe_msg = "缺少 ASTRO_UID/UID，跳过接口探测"

        report["checks"]["astro"] = {
            "ok": astro_ok,
            "base_url": _astro_base_url(),
            "missing_env": missing,
            "uid_present": bool(uid),
            "timeout_seconds": timeout_seconds,
            "api_probe_ok": probe_ok,
            "api_probe_msg": probe_msg,
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
        has_key = bool(os.getenv("SERPAPI_API_KEY", "").strip())
        serpapi_installed = importlib.util.find_spec("serpapi") is not None
        search_ok = has_key and serpapi_installed
        report["checks"]["search"] = {
            "ok": search_ok,
            "serpapi_installed": serpapi_installed,
            "missing_env": [] if has_key else ["SERPAPI_API_KEY"],
        }
        report["ok"] = report["ok"] and search_ok

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    report["meta"]["elapsed_ms"] = elapsed_ms
    logger.info("执行测试工具 | scope: %s | ok: %s", clean_scope, report["ok"])
    return json.dumps(report, ensure_ascii=False)

@tool
def search(query: str) -> str:
    """搜索工具:只有当你需要获取最新信息或者查询事实时才使用这个工具。"""
    clean_query = (query or "").strip()
    if not clean_query:
        return "请输入需要搜索的问题。"

    try:
        start_time = time.perf_counter()
        serp = SerpAPIWrapper()
        result = serp.run(clean_query)
        logger.info(f"执行搜索工具 | 查询: {clean_query[:50]} | 结果: {str(result)[:100]}")
        log_event(
            logging.INFO,
            "tool.search",
            query_len=len(clean_query),
            elapsed_ms=int((time.perf_counter() - start_time) * 1000),
        )
        return str(result)
    except Exception as e:
        logger.error(f"搜索工具异常 | 查询: {clean_query[:50]} | 错误: {str(e)[:100]}", exc_info=True)
        log_event(
            logging.ERROR,
            "tool.search.error",
            query_len=len(clean_query),
            error=str(e)[:160],
        )
        return "搜索服务暂时不可用，请稍后重试。"

@tool
def vector_search(query: str) -> str:
    """向量搜索工具:只有当你需要从已知文档中查询信息时才使用这个工具。"""
    clean_query = (query or "").strip()
    if not clean_query:
        return "请输入需要检索的问题。"

    try:
        start_time = time.perf_counter()
        retriever = _get_vector_retriever()
        docs = retriever.invoke(clean_query)
        result = "\n\n".join(
            [doc.page_content for doc in docs if getattr(doc, "page_content", "")]
        ).strip()

        if not result:
            logger.info(f"执行向量搜索工具 | 查询: {clean_query[:50]} | 命中: 0")
            log_event(
                logging.INFO,
                "tool.vector_search",
                query_len=len(clean_query),
                hit_count=0,
                elapsed_ms=int((time.perf_counter() - start_time) * 1000),
            )
            return "未检索到相关内容。"

        logger.info(f"执行向量搜索工具 | 查询: {clean_query[:50]} | 结果长度: {len(result)}")
        log_event(
            logging.INFO,
            "tool.vector_search",
            query_len=len(clean_query),
            hit_count=len(docs) if docs is not None else 0,
            elapsed_ms=int((time.perf_counter() - start_time) * 1000),
        )
        return result
    except Exception as e:
        logger.error(f"向量搜索工具异常 | 查询: {clean_query[:50]} | 错误: {str(e)[:100]}", exc_info=True)
        log_event(
            logging.ERROR,
            "tool.vector_search.error",
            query_len=len(clean_query),
            error=str(e)[:160],
        )
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
    clean_birth_dt = _normalize_birth_dt((birth_dt or "").strip())
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
    clean_birth_dt = _normalize_birth_dt((birth_dt or "").strip())
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


