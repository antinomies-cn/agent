import time
import logging
from typing import Any, Dict, List, Optional

from app.core.embedding_config import resolve_embedding_config
from app.core.config import (
    get_embeddings_gateway_settings,
    get_rerank_gateway_settings,
    normalize_openai_base_url,
)
from app.core.gateway_http import post_json_with_retry
from app.core.logger_setup import log_event

logger = logging.getLogger(__name__)


class LiteLLMEmbeddings:
    """LangChain-compatible embeddings client backed by LiteLLM OpenAI endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        dimensions: Optional[int] = None,
        timeout: float = 30.0,
        retry_count: int = 1,
    ) -> None:
        self.model = (model or "bge-m3").strip() or "bge-m3"
        self.base_url = normalize_openai_base_url(base_url)
        self.api_key = (api_key or "").strip()
        self.dimensions = dimensions if isinstance(dimensions, int) and dimensions > 0 else None
        self.timeout = max(float(timeout or 0), 3.0)
        self.retry_count = max(int(retry_count or 0), 0)

    @staticmethod
    def _parse_embedding_payload(payload: Dict[str, Any]) -> List[List[float]]:
        data = payload.get("data")
        if not isinstance(data, list) or not data:
            raise RuntimeError("embeddings response missing data")

        vectors: List[List[float]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            emb = item.get("embedding")
            if isinstance(emb, list) and emb:
                vectors.append(emb)

        if not vectors:
            raise RuntimeError("embeddings response missing embedding vectors")
        return vectors

    def _request_embeddings(self, inputs: List[str]) -> List[List[float]]:
        if not self.base_url:
            raise RuntimeError("OPENAI_API_BASE 未配置，无法调用 LiteLLM embeddings")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY 未配置，无法调用 LiteLLM embeddings")

        url = f"{self.base_url}/v1/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        body: Dict[str, Any] = {
            "model": self.model,
            "input": inputs,
        }
        if self.dimensions:
            body["dimensions"] = self.dimensions

        start = time.perf_counter()
        try:
            resp = post_json_with_retry(
                url=url,
                headers=headers,
                body=body,
                timeout_seconds=self.timeout,
                retry_count=self.retry_count,
                component="litellm",
                operation="embeddings",
                accepted_error_statuses=[400, 404, 422],
            )
            if resp.status_code >= 400 and self.dimensions:
                fallback_body = {
                    "model": self.model,
                    "input": inputs,
                }
                fallback_resp = post_json_with_retry(
                    url=url,
                    headers=headers,
                    body=fallback_body,
                    timeout_seconds=self.timeout,
                    retry_count=self.retry_count,
                    component="litellm",
                    operation="embeddings",
                )
                payload = fallback_resp.json()
            else:
                resp.raise_for_status()
                payload = resp.json()

            vectors = self._parse_embedding_payload(payload)
            log_event(
                logging.INFO,
                "litellm.embeddings.success",
                model=self.model,
                input_count=len(inputs),
                elapsed_ms=int((time.perf_counter() - start) * 1000),
            )
            return vectors
        except Exception as e:
            log_event(
                logging.ERROR,
                "litellm.embeddings.error",
                model=self.model,
                input_count=len(inputs),
                error=str(e)[:200],
                elapsed_ms=int((time.perf_counter() - start) * 1000),
            )
            raise RuntimeError(f"LiteLLM embeddings 调用失败: {str(e)[:220]}") from e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        clean_texts = [(text or "") for text in texts]
        if not clean_texts:
            return []
        return self._request_embeddings(clean_texts)

    def embed_query(self, text: str) -> List[float]:
        vectors = self._request_embeddings([(text or "")])
        return vectors[0]

    def __call__(self, text: str) -> List[float]:
        """兼容将 Embeddings 实例当作 embedding_function 可调用对象的场景。"""
        return self.embed_query(text)


def build_litellm_embeddings_client(default_dimensions: int = 384) -> LiteLLMEmbeddings:
    embedding_cfg = resolve_embedding_config(default_dimension=default_dimensions)
    gateway_cfg = get_embeddings_gateway_settings()

    model = embedding_cfg.model or gateway_cfg.model
    base_url = gateway_cfg.base_url
    api_key = gateway_cfg.api_key
    dimensions = embedding_cfg.dimensions
    timeout = gateway_cfg.timeout_seconds

    return LiteLLMEmbeddings(
        model=model,
        base_url=base_url,
        api_key=api_key,
        dimensions=dimensions,
        timeout=timeout,
        retry_count=gateway_cfg.retry_count,
    )


def rerank_texts_with_litellm(query: str, texts: List[str], top_n: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return sorted rerank results with stable schema: index/relevance_score."""
    clean_query = (query or "").strip()
    if not clean_query:
        return []

    clean_texts = [(text or "") for text in texts]
    if not clean_texts:
        return []

    rerank_cfg = get_rerank_gateway_settings()
    direct_upstream = rerank_cfg.direct_upstream
    base_url = rerank_cfg.base_url
    api_key = rerank_cfg.api_key
    request_model = rerank_cfg.request_model

    if not base_url:
        raise RuntimeError("RERANK_API_BASE/OPENAI_API_BASE 未配置，无法调用 rerank")
    if not api_key:
        raise RuntimeError("RERANK_API_KEY/OPENAI_API_KEY 未配置，无法调用 rerank")

    timeout = rerank_cfg.timeout_seconds

    body: Dict[str, Any] = {
        "model": request_model,
        "query": clean_query,
        "documents": clean_texts,
    }
    effective_top_n = top_n if isinstance(top_n, int) and top_n > 0 else rerank_cfg.top_n
    if isinstance(effective_top_n, int) and effective_top_n > 0:
        body["top_n"] = effective_top_n

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    urls = rerank_cfg.endpoint_candidates()
    last_error: Optional[Exception] = None

    for idx, url in enumerate(urls):
        start = time.perf_counter()
        try:
            resp = post_json_with_retry(
                url=url,
                headers=headers,
                body=body,
                timeout_seconds=timeout,
                retry_count=rerank_cfg.retry_count,
                component="litellm",
                operation="rerank",
                accepted_error_statuses=[404],
            )
            if resp.status_code == 404 and idx < len(urls) - 1:
                continue
            resp.raise_for_status()
            payload = resp.json()

            raw_results = payload.get("results")
            if not isinstance(raw_results, list):
                raw_results = payload.get("data")
            if not isinstance(raw_results, list):
                raise RuntimeError("rerank response missing results")

            results: List[Dict[str, Any]] = []
            for item in raw_results:
                if not isinstance(item, dict):
                    continue
                index = item.get("index")
                if not isinstance(index, int):
                    continue
                score = item.get("relevance_score")
                if not isinstance(score, (int, float)):
                    score = item.get("score")
                if not isinstance(score, (int, float)):
                    score = 0.0
                results.append({"index": index, "relevance_score": float(score)})

            if not results:
                raise RuntimeError("rerank response empty")

            log_event(
                logging.INFO,
                "litellm.rerank.success",
                model=request_model,
                direct_upstream=direct_upstream,
                query_len=len(clean_query),
                docs=len(clean_texts),
                elapsed_ms=int((time.perf_counter() - start) * 1000),
            )
            return sorted(results, key=lambda x: x["relevance_score"], reverse=True)
        except Exception as e:
            last_error = e
            log_event(
                logging.WARNING,
                "litellm.rerank.retry",
                model=request_model,
                direct_upstream=direct_upstream,
                url=url,
                error=str(e)[:180],
                elapsed_ms=int((time.perf_counter() - start) * 1000),
            )

    raise RuntimeError(f"LiteLLM rerank 调用失败: {str(last_error)[:220] if last_error else 'unknown error'}")
