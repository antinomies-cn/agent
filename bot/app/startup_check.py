import os
import sys
import logging
import requests

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency in local shell
    load_dotenv = None

# 兼容容器/直跑：确保项目根目录在模块搜索路径中。
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from app.core.logger_setup import logger, log_event


def _load_env_files() -> None:
    """本地直跑时加载 .env；容器环境变量已注入时不会被覆盖。"""
    if load_dotenv is None:
        return

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_file = os.path.join(base_dir, ".env")
    env_example_file = os.path.join(base_dir, ".env.docker.example")

    if os.path.isfile(env_file):
        load_dotenv(dotenv_path=env_file, override=False)
    elif os.path.isfile(env_example_file):
        load_dotenv(dotenv_path=env_example_file, override=False)


def _is_true(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _warn(message: str) -> None:
    logger.warning("startup-check warn: %s", message)
    log_event(logging.WARNING, "startup_check.warn", message=message)


def _normalize_openai_base_url(raw_base_url: str) -> str:
    base = (raw_base_url or "").strip().rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base.rstrip("/")


def _fail(message: str) -> int:
    logger.error("startup-check error: %s", message)
    log_event(logging.ERROR, "startup_check.error", message=message)
    return 1


def main() -> int:
    _load_env_files()

    enabled = _is_true(os.getenv("EMBEDDINGS_STARTUP_CHECK", "true"), default=True)
    if not enabled:
        logger.info("startup-check skip: EMBEDDINGS_STARTUP_CHECK=false")
        log_event(logging.INFO, "startup_check.skip", reason="EMBEDDINGS_STARTUP_CHECK=false")
        return 0

    api_base = _normalize_openai_base_url(
        os.getenv("OPENAI_EMBEDDINGS_API_BASE", "").strip() or os.getenv("OPENAI_API_BASE", "").strip()
    )
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    embedding_model = os.getenv("EMBEDDINGS_MODEL", "bge-m3").strip() or "bge-m3"

    if not api_base:
        return _fail("OPENAI_API_BASE 未配置")
    if not api_key:
        return _fail("OPENAI_API_KEY 未配置")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    expected_dim = None
    expected_dim_text = os.getenv("EMBEDDINGS_DIMENSION", "").strip()
    if expected_dim_text:
        try:
            parsed = int(expected_dim_text)
            if parsed > 0:
                expected_dim = parsed
        except ValueError:
            logger.warning("startup-check warn: EMBEDDINGS_DIMENSION invalid: %s", expected_dim_text)
            log_event(logging.WARNING, "startup_check.warn", reason="EMBEDDINGS_DIMENSION invalid")

    embed_body = {
        "model": embedding_model,
        "input": ["startup_check_probe"],
    }
    if expected_dim:
        embed_body["dimensions"] = expected_dim

    try:
        resp = requests.post(
            f"{api_base}/v1/embeddings",
            headers=headers,
            json=embed_body,
            timeout=15,
        )
        if resp.status_code >= 400 and expected_dim is not None:
            fallback_body = {
                "model": embedding_model,
                "input": ["startup_check_probe"],
            }
            fallback_resp = requests.post(
                f"{api_base}/v1/embeddings",
                headers=headers,
                json=fallback_body,
                timeout=15,
            )
            fallback_resp.raise_for_status()
            payload = fallback_resp.json()
        else:
            resp.raise_for_status()
            payload = resp.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or not data:
            return _fail("LiteLLM embeddings 响应缺少 data")
        first = data[0] if isinstance(data[0], dict) else {}
        vector = first.get("embedding") if isinstance(first, dict) else None
        if not isinstance(vector, list) or not vector:
            return _fail("LiteLLM embeddings 响应缺少 embedding")
        actual_dim = len(vector)
    except Exception as e:
        return _fail(f"LiteLLM embeddings 探测失败: {str(e)[:280]}")

    if expected_dim and actual_dim != expected_dim:
        return _fail(f"向量维度不匹配: actual={actual_dim}, EMBEDDINGS_DIMENSION={expected_dim}")

    rerank_enabled = _is_true(os.getenv("RERANK_ENABLED", "true"), default=True)
    rerank_model = os.getenv("RERANK_MODEL", "bge-reranker").strip() or "bge-reranker"
    rerank_ok = True
    rerank_note = "skipped"

    if rerank_enabled:
        rerank_strict = _is_true(os.getenv("RERANK_STARTUP_STRICT", "false"), default=False)
        direct_upstream = _is_true(os.getenv("RERANK_DIRECT_UPSTREAM", "true"), default=True)
        rerank_base = _normalize_openai_base_url(os.getenv("RERANK_API_BASE", "").strip()) if direct_upstream else api_base
        rerank_api_key = os.getenv("RERANK_API_KEY", "").strip() if direct_upstream else api_key
        rerank_body = {
            "model": (os.getenv("RERANK_UPSTREAM_MODEL", "").strip() or ("bge-reranker-v2-m3" if rerank_model == "bge-reranker" else rerank_model)),
            "query": "startup_check_probe",
            "documents": ["alpha", "beta"],
            "top_n": 1,
        }

        if not rerank_base or not rerank_api_key:
            message = "RERANK_API_BASE 或 RERANK_API_KEY 未配置，跳过 rerank 探测"
            if rerank_strict:
                return _fail(message)
            _warn(message)
            rerank_ok = False
            rerank_note = "skipped_missing_config"
            logger.info(
                "startup-check ok: provider=litellm embedding_model=%s dim=%s rerank_enabled=%s rerank_model=%s rerank_endpoint=%s",
                embedding_model,
                actual_dim,
                rerank_enabled,
                rerank_model,
                rerank_note,
            )
            log_event(
                logging.INFO,
                "startup_check.ok",
                provider="litellm",
                embedding_model=embedding_model,
                dim=actual_dim,
                rerank_enabled=rerank_enabled,
                rerank_ok=rerank_ok,
                rerank_model=rerank_model,
                rerank_endpoint=rerank_note,
            )
            return 0

        rerank_headers = {
            "Authorization": f"Bearer {rerank_api_key}",
            "Content-Type": "application/json",
        }

        last_error = None
        endpoints = ("/rerank", "/v1/rerank") if direct_upstream else ("/v1/rerank", "/rerank")
        for endpoint in endpoints:
            try:
                rerank_resp = requests.post(
                    f"{rerank_base}{endpoint}",
                    headers=rerank_headers,
                    json=rerank_body,
                    timeout=15,
                )
                if rerank_resp.status_code == 404:
                    continue
                rerank_resp.raise_for_status()
                rerank_payload = rerank_resp.json()
                results = rerank_payload.get("results") if isinstance(rerank_payload, dict) else None
                if not isinstance(results, list):
                    results = rerank_payload.get("data") if isinstance(rerank_payload, dict) else None
                if not isinstance(results, list) or not results:
                    raise RuntimeError("rerank response missing results")
                rerank_note = endpoint
                last_error = None
                break
            except Exception as e:
                last_error = e

        if last_error is not None:
            rerank_ok = False
            message = f"LiteLLM rerank 探测失败: {str(last_error)[:280]}"
            if rerank_strict:
                return _fail(message)
            _warn(message)
            rerank_note = "failed_non_fatal"

    logger.info(
        "startup-check ok: provider=litellm embedding_model=%s dim=%s rerank_enabled=%s rerank_model=%s rerank_endpoint=%s",
        embedding_model,
        actual_dim,
        rerank_enabled,
        rerank_model,
        rerank_note,
    )
    log_event(
        logging.INFO,
        "startup_check.ok",
        provider="litellm",
        embedding_model=embedding_model,
        dim=actual_dim,
        rerank_enabled=rerank_enabled,
        rerank_ok=rerank_ok,
        rerank_model=rerank_model,
        rerank_endpoint=rerank_note,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
