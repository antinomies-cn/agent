import os
import sys

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency in local shell
    load_dotenv = None


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
    print(f"[startup-check] ERROR: {message}", file=sys.stderr)
    return 1


def main() -> int:
    _load_env_files()

    enabled = _is_true(os.getenv("EMBEDDINGS_STARTUP_CHECK", "true"), default=True)
    if not enabled:
        print("[startup-check] skip: EMBEDDINGS_STARTUP_CHECK=false")
        return 0

    provider = os.getenv("EMBEDDINGS_API", "openai").strip().lower()
    if provider != "local":
        print(f"[startup-check] skip: EMBEDDINGS_API={provider}")
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
            print(f"[startup-check] WARN: EMBEDDINGS_DIMENSION 非法: {expected_dim_text}")

    print(
        "[startup-check] ok: "
        f"provider=local model={resolved_model} local_files_only={local_files_only} dim={dim}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
