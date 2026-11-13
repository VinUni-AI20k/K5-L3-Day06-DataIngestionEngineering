"""Real embeddings from an OpenAI-compatible API or Sentence Transformers."""
import json
import os
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, question: str) -> list[float]: ...


class MyEmbedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.provider = os.getenv("EMBEDDING_PROVIDER", "openai_compatible")
        if self.provider == "local":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise ValueError("Install sentence-transformers for EMBEDDING_PROVIDER=local.") from error
            self.model = SentenceTransformer(model_name)
            self.dimensions = self.model.get_sentence_embedding_dimension()
        elif self.provider == "openai_compatible":
            key_name = os.getenv("EMBEDDING_API_KEY_ENV") or "OPENAI_API_KEY"
            self.key = os.getenv(key_name)
            if not self.key:
                raise ValueError(f"Set {key_name} in starter/.env.")
            base_url = os.getenv("EMBEDDING_BASE_URL") or "https://api.openai.com/v1"
            self.url = base_url.rstrip("/") + "/embeddings"
            self.dimensions = len(self.embed_query("Embedding dimension check."))
        else:
            raise ValueError("EMBEDDING_PROVIDER must be openai_compatible or local.")

    def _request(self, texts: list[str]) -> list[list[float]]:
        request = Request(
            self.url,
            data=json.dumps({"model": self.model_name, "input": texts}).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=60) as response:
                rows = json.load(response)["data"]
        except HTTPError as error:
            raise ValueError(f"Embedding API returned HTTP {error.code}. Check API access, model and quota.") from error
        except (URLError, TimeoutError) as error:
            raise ValueError("Could not reach the embedding API. Check the network and EMBEDDING_BASE_URL.") from error
        rows = sorted(rows, key=lambda row: row["index"])
        if [row["index"] for row in rows] != list(range(len(texts))):
            raise ValueError("Embedding API returned unexpected vector indices.")
        return [row["embedding"] for row in rows]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.provider == "local":
            return self.model.encode(texts, batch_size=32).tolist()
        vectors = []
        for start in range(0, len(texts), 64):
            vectors.extend(self._request(texts[start:start + 64]))
        return vectors

    def embed_query(self, question: str) -> list[float]:
        return self.embed_documents([question])[0]
