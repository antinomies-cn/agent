import os
from dataclasses import dataclass
from typing import Optional

DEFAULT_EMBEDDINGS_DIMENSION = 1024

_MODEL_DIMENSION_HINTS = {
    "bge-m3": 1024,
    "baai/bge-m3": 1024,
    "bge-small-zh-v1.5": 512,
    "baai/bge-small-zh-v1.5": 512,
}


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str
    dimensions: int
    dimension_source: str


def _normalize_model_name(model_name: str) -> str:
    return (model_name or "").strip().lower()


def _sanitize_model_name(raw_model: str) -> str:
    clean = (raw_model or "").strip()
    if not clean:
        return "bge-m3"

    # Legacy local path values are not LiteLLM model aliases.
    if clean.startswith("/") or clean.startswith(".") or "\\" in clean:
        return "bge-m3"

    return clean


def _resolve_dimension_from_env() -> Optional[int]:
    raw = os.getenv("EMBEDDINGS_DIMENSION", "").strip()
    if not raw:
        return None

    try:
        value = int(raw)
    except ValueError:
        return None

    if value > 0:
        return value
    return None


def _resolve_dimension_by_model(model_name: str) -> Optional[int]:
    return _MODEL_DIMENSION_HINTS.get(_normalize_model_name(model_name))


def resolve_embedding_config(default_dimension: Optional[int] = None, model_name: Optional[str] = None) -> EmbeddingConfig:
    resolved_model = _sanitize_model_name(model_name or os.getenv("EMBEDDINGS_MODEL", "bge-m3"))

    env_dim = _resolve_dimension_from_env()
    if env_dim is not None:
        return EmbeddingConfig(model=resolved_model, dimensions=env_dim, dimension_source="env")

    model_hint_dim = _resolve_dimension_by_model(resolved_model)
    if model_hint_dim is not None:
        return EmbeddingConfig(model=resolved_model, dimensions=model_hint_dim, dimension_source="model_hint")

    if isinstance(default_dimension, int) and default_dimension > 0:
        return EmbeddingConfig(model=resolved_model, dimensions=default_dimension, dimension_source="fallback")

    return EmbeddingConfig(model=resolved_model, dimensions=DEFAULT_EMBEDDINGS_DIMENSION, dimension_source="default")
