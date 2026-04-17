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

from app.core.embedding_config import resolve_embedding_config
from app.core.logger_setup import logger, log_event
from app.core.config import (
    build_config_health_summary,
    get_env_bool,
    get_env_str,
    get_embeddings_gateway_settings,
    get_rerank_gateway_settings,
)


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


def _warn(message: str) -> None:
    logger.warning("startup-check warn: %s", message)
    log_event(logging.WARNING, "startup_check.warn", message=message)


def _fail(message: str) -> int:
    logger.error("startup-check error: %s", message)
    log_event(logging.ERROR, "startup_check.error", message=message)
    return 1


def main() -> int:
    _load_env_files()

    summary = build_config_health_summary()
    log_event(
        logging.INFO if summary.get("ok", False) else logging.ERROR,
        "config.health.summary",
        ok=bool(summary.get("ok", False)),
        warnings=list(summary.get("warnings", ())),
        errors=list(summary.get("errors", ())),
        **dict(summary.get("highlights", {})),
    )

    enabled = get_env_bool("EMBEDDINGS_STARTUP_CHECK", default=True)
    if not enabled:
        logger.info("startup-check skip: EMBEDDINGS_STARTUP_CHECK=false")
        log_event(logging.INFO, "startup_check.skip", reason="EMBEDDINGS_STARTUP_CHECK=false")
        return 0

    embeddings_cfg = get_embeddings_gateway_settings()
    resolved_embedding_cfg = resolve_embedding_config(model_name=embeddings_cfg.model)
    api_base = embeddings_cfg.base_url
    api_key = embeddings_cfg.api_key
    embedding_model = resolved_embedding_cfg.model

    if not api_base:
        return _fail("OPENAI_API_BASE 未配置")
    if not api_key:
        return _fail("OPENAI_API_KEY 未配置")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    expected_dim = None
    expected_dim_text = get_env_str("EMBEDDINGS_DIMENSION", "")
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

    rerank_cfg = get_rerank_gateway_settings()
    rerank_enabled = rerank_cfg.enabled
    rerank_model = rerank_cfg.model
    rerank_ok = True
    rerank_note = "skipped"

    if rerank_enabled:
        rerank_strict = rerank_cfg.startup_strict
        direct_upstream = rerank_cfg.direct_upstream
        rerank_base = rerank_cfg.base_url
        rerank_api_key = rerank_cfg.api_key
        rerank_body = {
            "model": rerank_cfg.request_model,
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
        for url in rerank_cfg.endpoint_candidates():
            try:
                rerank_resp = requests.post(
                    url,
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
                rerank_note = url.replace(rerank_base, "") if rerank_base else url
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
