"""OpenRouter embedding adapter."""
import json
import math
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from typing import Protocol


OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
MODEL_DIMENSIONS = {
    "openai/text-embedding-3-large": 3072,
}
MAX_BATCH_SIZE = 64


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """Create real embeddings through OpenRouter."""

    def __init__(self, model_name: str) -> None:
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")

        if model_name not in MODEL_DIMENSIONS:
            supported = ", ".join(sorted(MODEL_DIMENSIONS))
            raise ValueError(
                f"Unsupported embedding model {model_name!r}; supported: {supported}"
            )

        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is missing from starter/.env")

        self.model_name = model_name
        self.dimensions = MODEL_DIMENSIONS[model_name]
        self._api_key = api_key
        self._reported_tokens = 0
        self._reported_cost_usd = 0.0
        self._tokens_complete = True
        self._cost_complete = True

    @property
    def reported_input_tokens(self) -> int | None:
        return self._reported_tokens if self._tokens_complete else None

    @property
    def reported_cost_usd(self) -> float | None:
        return self._reported_cost_usd if self._cost_complete else None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("texts must contain only non-empty strings")

        vectors = []
        for start in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[start : start + MAX_BATCH_SIZE]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def embed_query(self, question: str) -> list[float]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        return self._embed_batch([question])[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps(
            {"model": self.model_name, "input": texts},
            ensure_ascii=False,
        ).encode("utf-8")
        request = Request(
            OPENROUTER_EMBEDDINGS_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=60) as response:
                body = response.read()
        except HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenRouter embeddings request failed ({error.code}): "
                f"{details[:500]}"
            ) from error
        except (URLError, TimeoutError) as error:
            raise RuntimeError(
                f"Could not reach OpenRouter embeddings API: {error}"
            ) from error

        try:
            result = json.loads(body)
        except json.JSONDecodeError as error:
            raise RuntimeError("OpenRouter returned invalid JSON") from error

        usage = result.get("usage") if isinstance(result, dict) else None
        if isinstance(usage, dict):
            token_count = usage.get("prompt_tokens", usage.get("total_tokens"))
            if (
                isinstance(token_count, int)
                and not isinstance(token_count, bool)
                and token_count >= 0
            ):
                self._reported_tokens += token_count
            else:
                self._tokens_complete = False

            cost = usage.get("cost")
            if (
                isinstance(cost, (int, float))
                and not isinstance(cost, bool)
                and math.isfinite(cost)
                and cost >= 0
            ):
                self._reported_cost_usd += float(cost)
            else:
                self._cost_complete = False
        else:
            self._tokens_complete = False
            self._cost_complete = False

        data = result.get("data") if isinstance(result, dict) else None
        if not isinstance(data, list) or len(data) != len(texts):
            raise RuntimeError(
                "OpenRouter response must contain one embedding per input text"
            )

        vectors: list[list[float] | None] = [None] * len(texts)
        for item in data:
            if not isinstance(item, dict):
                raise RuntimeError("OpenRouter returned an invalid embedding item")
            index = item.get("index")
            embedding = item.get("embedding")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= len(texts)
                or vectors[index] is not None
            ):
                raise RuntimeError("OpenRouter returned invalid embedding indexes")
            if not isinstance(embedding, list) or len(embedding) != self.dimensions:
                raise RuntimeError(
                    f"Expected {self.dimensions}-dimension embeddings from "
                    f"{self.model_name}"
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in embedding
            ):
                raise RuntimeError("OpenRouter returned non-numeric embedding values")
            vectors[index] = [float(value) for value in embedding]

        if any(vector is None for vector in vectors):
            raise RuntimeError("OpenRouter omitted one or more embeddings")
        return [vector for vector in vectors if vector is not None]
