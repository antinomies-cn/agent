import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# 环境变量优先级：系统环境 > .env > .env.docker.example。
# 这样既支持本地/容器通过 .env 注入，也保留模板文件中的默认回退值。
_ROOT_DIR = Path(__file__).resolve().parents[2]
PRIMARY_ENV_FILE_PATH = _ROOT_DIR / ".env"
TEMPLATE_ENV_FILE_PATH = _ROOT_DIR / ".env.docker.example"

load_dotenv(dotenv_path=PRIMARY_ENV_FILE_PATH, override=False)
load_dotenv(dotenv_path=TEMPLATE_ENV_FILE_PATH, override=False)


def _to_bool(value: str, default: bool = False) -> bool:
	if value is None:
		return default
	return str(value).strip().lower() in {"1", "true", "yes", "on"}


def normalize_openai_base_url(raw_base_url: str) -> str:
	"""Normalize base URL and strip trailing /v1 when present."""
	base = (raw_base_url or "").strip().rstrip("/")
	if base.endswith("/v1"):
		base = base[:-3]
	return base.rstrip("/")


def get_env_bool(name: str, default: bool = False) -> bool:
	return _to_bool(os.getenv(name), default=default)


def get_env_int(name: str, default: int, min_value: Optional[int] = None) -> int:
	raw = os.getenv(name, str(default)).strip()
	try:
		value = int(raw)
	except ValueError:
		value = default
	if min_value is not None:
		value = max(min_value, value)
	return value


def get_env_float(name: str, default: float, min_value: Optional[float] = None) -> float:
	raw = os.getenv(name, str(default)).strip()
	try:
		value = float(raw)
	except ValueError:
		value = default
	if min_value is not None:
		value = max(min_value, value)
	return value


def resolve_openai_api_base() -> str:
	return normalize_openai_base_url(os.getenv("OPENAI_API_BASE", "").strip())


def resolve_openai_embeddings_api_base() -> str:
	base = os.getenv("OPENAI_EMBEDDINGS_API_BASE", "").strip()
	if not base:
		base = os.getenv("OPENAI_API_BASE", "").strip()
	return normalize_openai_base_url(base)


def resolve_rerank_upstream_model(model: str) -> str:
	upstream_model = os.getenv("RERANK_UPSTREAM_MODEL", "").strip()
	if upstream_model:
		return upstream_model
	return "bge-reranker-v2-m3" if model == "bge-reranker" else model


@dataclass(frozen=True)
class LLMGatewaySettings:
	api_key: str
	base_url: str
	model: str
	retry_count: int


@dataclass(frozen=True)
class EmbeddingsGatewaySettings:
	api_key: str
	base_url: str
	model: str
	timeout_seconds: float
	retry_count: int


@dataclass(frozen=True)
class RerankGatewaySettings:
	enabled: bool
	direct_upstream: bool
	model: str
	upstream_model: str
	base_url: str
	display_base_url: str
	api_key: str
	timeout_seconds: float
	retry_count: int
	top_n: Optional[int]
	startup_strict: bool

	@property
	def request_model(self) -> str:
		return self.upstream_model if self.direct_upstream else self.model

	def endpoint_candidates(self) -> list[str]:
		if self.direct_upstream:
			return [f"{self.base_url}/rerank", f"{self.base_url}/v1/rerank"]
		return [f"{self.base_url}/v1/rerank", f"{self.base_url}/rerank"]


def get_llm_gateway_settings() -> LLMGatewaySettings:
	return LLMGatewaySettings(
		api_key=os.getenv("OPENAI_API_KEY", "").strip(),
		base_url=resolve_openai_api_base(),
		model=(os.getenv("OPENAI_MODEL", "").strip()),
		retry_count=get_env_int("LLM_RETRY_COUNT", default=1, min_value=0),
	)


def get_embeddings_gateway_settings() -> EmbeddingsGatewaySettings:
	return EmbeddingsGatewaySettings(
		api_key=os.getenv("OPENAI_API_KEY", "").strip(),
		base_url=resolve_openai_embeddings_api_base(),
		model=(os.getenv("EMBEDDINGS_MODEL", "bge-m3").strip() or "bge-m3"),
		timeout_seconds=get_env_float("EMBEDDINGS_TIMEOUT", default=30.0, min_value=3.0),
		retry_count=get_env_int("EMBEDDINGS_RETRY_COUNT", default=1, min_value=0),
	)


def get_rerank_gateway_settings() -> RerankGatewaySettings:
	direct_upstream = get_env_bool("RERANK_DIRECT_UPSTREAM", default=True)
	model = os.getenv("RERANK_MODEL", "bge-reranker").strip() or "bge-reranker"
	upstream_model = resolve_rerank_upstream_model(model)

	if direct_upstream:
		raw_base_url = os.getenv("RERANK_API_BASE", "").strip()
		api_key = os.getenv("RERANK_API_KEY", "").strip()
	else:
		raw_base_url = os.getenv("OPENAI_API_BASE", "").strip()
		api_key = os.getenv("OPENAI_API_KEY", "").strip()

	display_base_url = (raw_base_url or "").rstrip("/")
	base_url = normalize_openai_base_url(raw_base_url)

	top_n_raw = os.getenv("RERANK_TOP_N", "").strip()
	top_n = None
	if top_n_raw:
		try:
			parsed_top_n = int(top_n_raw)
			if parsed_top_n > 0:
				top_n = parsed_top_n
		except ValueError:
			top_n = None

	return RerankGatewaySettings(
		enabled=get_env_bool("RERANK_ENABLED", default=True),
		direct_upstream=direct_upstream,
		model=model,
		upstream_model=upstream_model,
		base_url=base_url,
		display_base_url=display_base_url,
		api_key=api_key,
		timeout_seconds=get_env_float("RERANK_TIMEOUT", default=15.0, min_value=3.0),
		retry_count=get_env_int("RERANK_RETRY_COUNT", default=1, min_value=0),
		top_n=top_n,
		startup_strict=get_env_bool("RERANK_STARTUP_STRICT", default=False),
	)


IS_PROD = os.getenv("ENV", "dev").strip().lower() == "prod"
