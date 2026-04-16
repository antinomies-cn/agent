import os
import time
import logging
from typing import Any, Dict, List, Optional

import requests

from app.core.embedding_config import resolve_embedding_config
from app.core.logger_setup import log_event

logger = logging.getLogger(__name__)


def _normalize_openai_base_url(raw_base_url: str) -> str:
    """Normalize base URL and strip trailing /v1 when present."""
    base = (raw_base_url or "").strip().rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base.rstrip("/")


def _resolve_litellm_base_url() -> str:
    base = os.getenv("OPENAI_EMBEDDINGS_API_BASE", "").strip()
    if not base:
        base = os.getenv("OPENAI_API_BASE", "").strip()
    return _normalize_openai_base_url(base)


def _resolve_litellm_api_key() -> str:
    return os.getenv("OPENAI_API_KEY", "").strip()


class LiteLLMEmbeddings:
    """LangChain-compatible embeddings client backed by LiteLLM OpenAI endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        dimensions: Optional[int] = None,
        timeout: float = 30.0,
    ) -> None:
        self.model = (model or "bge-m3").strip() or "bge-m3"
        self.base_url = _normalize_openai_base_url(base_url)
        self.api_key = (api_key or "").strip()
        self.dimensions = dimensions if isinstance(dimensions, int) and dimensions > 0 else None
        self.timeout = max(float(timeout or 0), 3.0)

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
            resp = requests.post(url, headers=headers, json=body, timeout=self.timeout)
            if resp.status_code >= 400 and self.dimensions:
                fallback_body = {
                    "model": self.model,
                    "input": inputs,
                }
                fallback_resp = requests.post(url, headers=headers, json=fallback_body, timeout=self.timeout)
                fallback_resp.raise_for_status()
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
    model = embedding_cfg.model
    base_url = _resolve_litellm_base_url()
    api_key = _resolve_litellm_api_key()
    dimensions = embedding_cfg.dimensions

    timeout = 30.0
    raw_timeout = os.getenv("EMBEDDINGS_TIMEOUT", "30").strip()
    if raw_timeout:
        try:
            timeout = float(raw_timeout)
        except ValueError:
            timeout = 30.0

    return LiteLLMEmbeddings(
        model=model,
        base_url=base_url,
        api_key=api_key,
        dimensions=dimensions,
        timeout=timeout,
    )


def rerank_texts_with_litellm(query: str, texts: List[str], top_n: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return sorted rerank results with stable schema: index/relevance_score."""
    clean_query = (query or "").strip()
    if not clean_query:
        return []

    clean_texts = [(text or "") for text in texts]
    if not clean_texts:
        return []

    direct_upstream = os.getenv("RERANK_DIRECT_UPSTREAM", "true").strip().lower() in {"1", "true", "yes", "on"}
    model = os.getenv("RERANK_MODEL", "bge-reranker").strip() or "bge-reranker"
    upstream_model = os.getenv("RERANK_UPSTREAM_MODEL", "").strip()
    if not upstream_model:
        upstream_model = "bge-reranker-v2-m3" if model == "bge-reranker" else model

    if direct_upstream:
        base_url = _normalize_openai_base_url(os.getenv("RERANK_API_BASE", "").strip())
        api_key = os.getenv("RERANK_API_KEY", "").strip()
        request_model = upstream_model
    else:
        base_url = _resolve_litellm_base_url()
        api_key = _resolve_litellm_api_key()
        request_model = model

    if not base_url:
        raise RuntimeError("RERANK_API_BASE/OPENAI_API_BASE 未配置，无法调用 rerank")
    if not api_key:
        raise RuntimeError("RERANK_API_KEY/OPENAI_API_KEY 未配置，无法调用 rerank")

    timeout = 15.0
    raw_timeout = os.getenv("RERANK_TIMEOUT", "15").strip()
    if raw_timeout:
        try:
            timeout = float(raw_timeout)
        except ValueError:
            timeout = 15.0
    timeout = max(timeout, 3.0)

    body: Dict[str, Any] = {
        "model": request_model,
        "query": clean_query,
        "documents": clean_texts,
    }
    if isinstance(top_n, int) and top_n > 0:
        body["top_n"] = top_n

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    if direct_upstream:
        urls = [f"{base_url}/rerank", f"{base_url}/v1/rerank"]
    else:
        urls = [f"{base_url}/v1/rerank", f"{base_url}/rerank"]
    last_error: Optional[Exception] = None

    for idx, url in enumerate(urls):
        start = time.perf_counter()
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=timeout)
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
