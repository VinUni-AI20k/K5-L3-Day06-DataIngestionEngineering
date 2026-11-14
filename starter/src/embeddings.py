"""Real embedding provider: google/gemini-embedding-2 through the OpenRouter API.

Queries and documents are embedded the same way: OpenRouter's /embeddings
endpoint takes no task-type hint (unlike Gemini's RETRIEVAL_QUERY /
RETRIEVAL_DOCUMENT), so both sides go through the identical call and model.

Errors are raised, never replaced by mock vectors. Transient HTTP errors
(429/5xx, timeouts) are retried with backoff; anything else fails fast.
"""
from __future__ import annotations

import os
import time
from typing import Protocol

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings"


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class EmbeddingError(RuntimeError):
    pass


class OpenRouterEmbedder:
    """OpenRouter adapter; tracks tokens and USD cost reported by the API."""

    def __init__(self, model_name: str | None = None, api_key: str | None = None,
                 batch_size: int = 64, max_retries: int = 4, timeout: int = 120) -> None:
        self.model_name = model_name or os.environ["EMBEDDING_MODEL"]
        self.api_key = api_key or os.environ["OPENROUTER_API_KEY"]
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout
        self.tokens = 0
        self.cost_usd = 0.0
        self.calls = 0
        # Probe once so the store can validate its vector size up front.
        self.dimensions = len(self._post(["dimension probe"])[0])

    def _post(self, texts: list[str]) -> list[list[float]]:
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    OPENROUTER_URL,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model_name, "input": texts},
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                if attempt == self.max_retries:
                    raise EmbeddingError(f"OpenRouter request failed: {exc}") from exc
                time.sleep(delay)
                delay *= 2
                continue
            if resp.status_code == 200:
                body = resp.json()
                data = sorted(body["data"], key=lambda d: d["index"])
                if len(data) != len(texts):
                    raise EmbeddingError(f"expected {len(texts)} vectors, got {len(data)}")
                usage = body.get("usage") or {}
                self.tokens += int(usage.get("prompt_tokens") or usage.get("total_tokens") or 0)
                self.cost_usd += float(usage.get("cost") or 0.0)
                self.calls += 1
                return [d["embedding"] for d in data]
            if resp.status_code in (408, 429, 500, 502, 503, 504) and attempt < self.max_retries:
                time.sleep(delay)
                delay *= 2
                continue
            raise EmbeddingError(f"OpenRouter HTTP {resp.status_code}: {resp.text[:300]}")
        raise EmbeddingError("unreachable")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = self._post(texts[i:i + self.batch_size])
            for v in batch:
                if len(v) != self.dimensions:
                    raise EmbeddingError(f"vector size {len(v)} != {self.dimensions}")
            vectors.extend(batch)
        return vectors

    def embed_query(self, question: str) -> list[float]:
        return self._post([question])[0]
