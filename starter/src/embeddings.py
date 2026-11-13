"""OpenAI embedding adapter (one real model for every strategy).

Errors from the API are raised, never replaced by mock/random vectors.
"""
from __future__ import annotations

import time
from typing import Protocol

from openai import OpenAI

# Native output dimensions of OpenAI embedding models.
KNOWN_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
# USD per 1M input tokens, from OpenAI's pricing page; recheck before reporting cost.
PRICE_PER_1M_TOKENS = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class OpenAIEmbedder:
    """Documents and queries are embedded the same way (OpenAI models use no prefixes)."""

    def __init__(self, model_name: str, batch_size: int = 96, max_retries: int = 3) -> None:
        if model_name not in KNOWN_DIMENSIONS:
            raise ValueError(f"Unknown OpenAI embedding model: {model_name!r}")
        self.model_name = model_name
        self.dimensions = KNOWN_DIMENSIONS[model_name]
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.client = OpenAI()  # reads OPENAI_API_KEY; raises if missing
        self.total_tokens = 0
        self.calls = 0

    def _embed(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.embeddings.create(model=self.model_name, input=texts)
                break
            except Exception:  # rate limit / transient network error
                if attempt == self.max_retries:
                    raise
                time.sleep(2 ** attempt)
        self.calls += 1
        self.total_tokens += resp.usage.total_tokens
        vectors = [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
        if len(vectors) != len(texts) or any(len(v) != self.dimensions for v in vectors):
            raise RuntimeError("Embedding API returned unexpected count or dimensions")
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._embed(texts[i:i + self.batch_size]))
        return out

    def embed_query(self, question: str) -> list[float]:
        return self._embed([question])[0]

    @property
    def cost_usd(self) -> float:
        return self.total_tokens / 1_000_000 * PRICE_PER_1M_TOKENS[self.model_name]
