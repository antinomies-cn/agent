import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

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


@dataclass(frozen=True)
class GatewaySecuritySettings:
	auth_enabled: bool
	auth_tokens: Tuple[str, ...]
	rate_limit_enabled: bool
	rate_limit_ip_requests: int
	rate_limit_session_requests: int
	rate_limit_window_seconds: int
	max_request_body_bytes: int
	request_timeout_seconds: float
	cors_allow_origins: Tuple[str, ...]
	cors_allow_methods: Tuple[str, ...]
	cors_allow_headers: Tuple[str, ...]
	cors_allow_credentials: bool
	trust_proxy_headers: bool


@dataclass(frozen=True)
class QdrantSettings:
	url: str
	api_key: str
	path: str
	collection: str
	distance: str

	@property
	def mode(self) -> str:
		return "remote" if self.url else "local"


@dataclass(frozen=True)
class ServerSettings:
	host: str
	port: int


@dataclass(frozen=True)
class RuntimeSettings:
	env: str
	is_prod: bool
	server: ServerSettings
	qdrant: QdrantSettings
	add_urls_write_enabled: bool
	web_loader_verify_ssl: bool
	embeddings_startup_check: bool


@dataclass(frozen=True)
class RedisSettings:
	host: str
	port: int
	db: int
	password: str
	url: str


def _split_csv_env(value: str, default: Tuple[str, ...] = ()) -> Tuple[str, ...]:
	text = (value or "").strip()
	if not text:
		return default
	parts = [item.strip() for item in text.split(",") if item.strip()]
	if not parts:
		return default
	return tuple(parts)


def _parse_size_bytes(raw: str, default_bytes: int) -> int:
	text = (raw or "").strip().lower()
	if not text:
		return default_bytes

	multiplier = 1
	if text.endswith("kb"):
		multiplier = 1024
		text = text[:-2]
	elif text.endswith("mb"):
		multiplier = 1024 * 1024
		text = text[:-2]
	elif text.endswith("gb"):
		multiplier = 1024 * 1024 * 1024
		text = text[:-2]

	try:
		value = int(text.strip())
	except ValueError:
		return default_bytes

	if value <= 0:
		return default_bytes
	return value * multiplier


def get_env_str(name: str, default: str = "") -> str:
	return (os.getenv(name, default) or "").strip()


def get_runtime_env(default: str = "dev") -> str:
	clean_default = (default or "dev").strip().lower() or "dev"
	clean = get_env_str("ENV", clean_default).lower()
	if clean in {"dev", "test", "prod"}:
		return clean
	return clean_default


def is_prod_runtime() -> bool:
	return get_runtime_env() == "prod"


def get_server_settings() -> ServerSettings:
	host = get_env_str("API_HOST", "127.0.0.1") or "127.0.0.1"
	raw_port = get_env_str("API_PORT", "8000") or "8000"
	try:
		port = int(raw_port)
	except ValueError:
		port = 8000
	if port <= 0 or port > 65535:
		port = 8000
	return ServerSettings(host=host, port=port)


def get_qdrant_settings() -> QdrantSettings:
	collection = get_env_str("QDRANT_COLLECTION", "divination_master_collection") or "divination_master_collection"
	distance = get_env_str("QDRANT_DISTANCE", "cosine").lower() or "cosine"
	return QdrantSettings(
		url=get_env_str("QDRANT_URL", ""),
		api_key=get_env_str("QDRANT_API_KEY", ""),
		path=get_env_str("QDRANT_DB_PATH", "./qdrant_data/qdrant.db") or "./qdrant_data/qdrant.db",
		collection=collection,
		distance=distance,
	)


def get_runtime_settings() -> RuntimeSettings:
	env = get_runtime_env()
	return RuntimeSettings(
		env=env,
		is_prod=env == "prod",
		server=get_server_settings(),
		qdrant=get_qdrant_settings(),
		add_urls_write_enabled=get_env_bool("ADD_URLS_WRITE_ENABLED", default=False),
		web_loader_verify_ssl=get_env_bool("WEB_LOADER_VERIFY_SSL", default=True),
		embeddings_startup_check=get_env_bool("EMBEDDINGS_STARTUP_CHECK", default=True),
	)


def get_redis_settings() -> RedisSettings:
	host = get_env_str("REDIS_HOST", "127.0.0.1") or "127.0.0.1"
	port = get_env_int("REDIS_PORT", default=6379, min_value=1)
	db = get_env_int("REDIS_DB", default=0, min_value=0)
	password = get_env_str("REDIS_PASSWORD", "")
	url = get_env_str("REDIS_URL", "")
	if not url:
		if password:
			url = f"redis://:{password}@{host}:{port}/{db}"
		else:
			url = f"redis://{host}:{port}/{db}"
	return RedisSettings(host=host, port=port, db=db, password=password, url=url)


def build_config_health_summary() -> dict:
	"""聚合关键配置并给出启动期可读的健康摘要。"""
	settings = get_runtime_settings()
	warnings = []
	errors = []

	raw_env = get_env_str("ENV", "dev").lower()
	if raw_env and raw_env not in {"dev", "test", "prod"}:
		warnings.append(f"ENV={raw_env} 非法，已回退为 {settings.env}")

	raw_port = get_env_str("API_PORT", "8000")
	try:
		parsed_port = int(raw_port)
		if parsed_port <= 0 or parsed_port > 65535:
			warnings.append(f"API_PORT={raw_port} 超出范围，已回退为 {settings.server.port}")
	except ValueError:
		warnings.append(f"API_PORT={raw_port} 非数字，已回退为 {settings.server.port}")

	if settings.is_prod and settings.server.host in {"127.0.0.1", "localhost"}:
		warnings.append("生产环境仍绑定本地回环地址，外部访问可能不可达")

	if settings.qdrant.distance not in {"cosine", "dot", "euclid"}:
		warnings.append(f"QDRANT_DISTANCE={settings.qdrant.distance} 未识别，将按 cosine 处理")

	if settings.qdrant.mode == "remote" and not settings.qdrant.api_key:
		warnings.append("QDRANT_URL 已配置但 QDRANT_API_KEY 为空，请确认远端是否允许匿名访问")

	if settings.qdrant.mode == "local" and not settings.qdrant.path:
		errors.append("本地模式缺少 QDRANT_DB_PATH")

	return {
		"ok": len(errors) == 0,
		"errors": tuple(errors),
		"warnings": tuple(warnings),
		"highlights": {
			"env": settings.env,
			"is_prod": settings.is_prod,
			"api_bind": f"{settings.server.host}:{settings.server.port}",
			"qdrant_mode": settings.qdrant.mode,
			"qdrant_collection": settings.qdrant.collection,
			"qdrant_distance": settings.qdrant.distance,
			"add_urls_write_enabled": settings.add_urls_write_enabled,
			"web_loader_verify_ssl": settings.web_loader_verify_ssl,
			"embeddings_startup_check": settings.embeddings_startup_check,
		},
	}


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


def get_gateway_security_settings() -> GatewaySecuritySettings:
	tokens = _split_csv_env(os.getenv("GATEWAY_AUTH_TOKENS", ""), default=())
	auth_enabled = get_env_bool("GATEWAY_AUTH_ENABLED", default=bool(tokens))

	allow_origins = _split_csv_env(os.getenv("GATEWAY_CORS_ALLOW_ORIGINS", "*"), default=("*",))
	allow_methods = _split_csv_env(os.getenv("GATEWAY_CORS_ALLOW_METHODS", "*"), default=("*",))
	allow_headers = _split_csv_env(os.getenv("GATEWAY_CORS_ALLOW_HEADERS", "*"), default=("*",))

	return GatewaySecuritySettings(
		auth_enabled=auth_enabled,
		auth_tokens=tokens,
		rate_limit_enabled=get_env_bool("GATEWAY_RATE_LIMIT_ENABLED", default=False),
		rate_limit_ip_requests=get_env_int("GATEWAY_RATE_LIMIT_IP_REQUESTS", default=60, min_value=1),
		rate_limit_session_requests=get_env_int("GATEWAY_RATE_LIMIT_SESSION_REQUESTS", default=30, min_value=1),
		rate_limit_window_seconds=get_env_int("GATEWAY_RATE_LIMIT_WINDOW_SECONDS", default=60, min_value=1),
		max_request_body_bytes=_parse_size_bytes(os.getenv("GATEWAY_MAX_REQUEST_BODY", "2097152"), default_bytes=2 * 1024 * 1024),
		request_timeout_seconds=get_env_float("GATEWAY_REQUEST_TIMEOUT_SECONDS", default=0.0, min_value=0.0),
		cors_allow_origins=allow_origins,
		cors_allow_methods=allow_methods,
		cors_allow_headers=allow_headers,
		cors_allow_credentials=get_env_bool("GATEWAY_CORS_ALLOW_CREDENTIALS", default=False),
		trust_proxy_headers=get_env_bool("GATEWAY_TRUST_PROXY_HEADERS", default=True),
	)


IS_PROD = is_prod_runtime()
