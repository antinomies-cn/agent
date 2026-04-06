import os
import sys
import logging

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


def _resolve_local_model_path(model_name: str, cache_dir: str) -> str:
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

    provider = os.getenv("EMBEDDINGS_API", "openai").strip().lower()
    if provider != "local":
        logger.info("startup-check skip: EMBEDDINGS_API=%s", provider)
        log_event(logging.INFO, "startup_check.skip", reason=f"EMBEDDINGS_API={provider}")
        return 0

    model_name = os.getenv("EMBEDDINGS_MODEL", "").strip()
    cache_dir = os.getenv("EMBEDDINGS_CACHE_DIR", "").strip()
    local_files_only = _is_true(os.getenv("EMBEDDINGS_LOCAL_FILES_ONLY", "false"), default=False)

    resolved_model = _resolve_local_model_path(model_name, cache_dir)

    if not resolved_model:
        return _fail("EMBEDDINGS_MODEL 未配置")
    if not os.path.isdir(resolved_model):
        return _fail(
            "本地模型目录不存在: "
            f"{resolved_model} | model={model_name} | cache_dir={cache_dir}"
        )

    required_files = [
        "config.json",
        "modules.json",
        "sentence_bert_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        os.path.join("1_Pooling", "config.json"),
        "model.safetensors",
    ]
    missing = [name for name in required_files if not os.path.isfile(os.path.join(resolved_model, name))]
    if missing:
        return _fail(f"模型目录缺少关键文件: {', '.join(missing)}")

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        return _fail(f"sentence-transformers 不可用: {str(e)[:200]}")

    if local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    try:
        model = SentenceTransformer(
            resolved_model,
            cache_folder=(cache_dir or None),
            local_files_only=local_files_only,
        )
        embeddings = model.encode(["startup_check_probe"])
        dim = len(embeddings[0])
    except Exception as e:
        return _fail(f"离线模型加载失败: {str(e)[:300]}")

    expected_dim_text = os.getenv("EMBEDDINGS_DIMENSION", "").strip()
    if expected_dim_text:
        try:
            expected_dim = int(expected_dim_text)
            if expected_dim > 0 and dim != expected_dim:
                return _fail(
                    f"向量维度不匹配: actual={dim}, EMBEDDINGS_DIMENSION={expected_dim}"
                )
        except ValueError:
            logger.warning("startup-check warn: EMBEDDINGS_DIMENSION invalid: %s", expected_dim_text)
            log_event(logging.WARNING, "startup_check.warn", reason="EMBEDDINGS_DIMENSION invalid")

    logger.info(
        "startup-check ok: provider=local model=%s local_files_only=%s dim=%s",
        resolved_model,
        local_files_only,
        dim,
    )
    log_event(
        logging.INFO,
        "startup_check.ok",
        provider="local",
        model=resolved_model,
        local_files_only=local_files_only,
        dim=dim,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
