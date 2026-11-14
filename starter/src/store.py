"""Qdrant vector store (replaces the pgvector starter; same contract).

One collection per chunking strategy: A -> day06_rhm_a, B -> day06_rhm_b.
Strategy C reads collection B and only adds a disease_no filter.
Cosine distance; search uses exact (brute-force) scoring so results are
deterministic and filtering cannot shrink an approximate candidate set.
"""
from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from qdrant_client import QdrantClient, models

from .models import Chunk

COLLECTIONS = {"A": "day06_rhm_a", "B": "day06_rhm_b", "C": "day06_rhm_b"}
FILTERABLE = {"disease_no": models.PayloadSchemaType.INTEGER,
              "document_id": models.PayloadSchemaType.KEYWORD,
              "special_id": models.PayloadSchemaType.KEYWORD,
              "section": models.PayloadSchemaType.KEYWORD}


def make_client(url: str, api_key: str | None, timeout: int = 60) -> QdrantClient:
    # qdrant-client defaults to port 6333 when the URL has none; an HTTPS
    # reverse proxy listens on 443, so pass the port explicitly.
    parsed = urlparse(url)
    port = parsed.port or (443 if parsed.scheme == "https" else 6333)
    return QdrantClient(url=url, port=port, api_key=api_key, timeout=timeout,
                        check_compatibility=False)


class QdrantStore:
    def __init__(self, url: str, api_key: str | None, dimensions: int) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.client = make_client(url, api_key)
        self.dimensions = dimensions

    @staticmethod
    def collection(strategy: str) -> str:
        return COLLECTIONS[strategy]

    def create_schema(self, strategy: str) -> None:
        """Create the collection and payload indexes if missing; validate size."""
        name = self.collection(strategy)
        if self.client.collection_exists(name):
            size = self.client.get_collection(name).config.params.vectors.size
            if size != self.dimensions:
                raise ValueError(f"{name} has vector size {size}, model gives {self.dimensions}")
        else:
            self.client.create_collection(
                name, vectors_config=models.VectorParams(size=self.dimensions,
                                                         distance=models.Distance.COSINE))
        for field, schema in FILTERABLE.items():
            self.client.create_payload_index(name, field_name=field, field_schema=schema, wait=True)

    def count(self, strategy: str) -> int:
        return self.client.count(self.collection(strategy), exact=True).count

    def existing_ids(self, strategy: str, document_id: str) -> set[str]:
        name, ids, offset = self.collection(strategy), set(), None
        flt = models.Filter(must=[models.FieldCondition(
            key="document_id", match=models.MatchValue(value=document_id))])
        while True:
            points, offset = self.client.scroll(name, scroll_filter=flt, limit=512, offset=offset,
                                                with_payload=False, with_vectors=False)
            ids.update(str(p.id) for p in points)
            if offset is None:
                return ids

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]],
               batch_size: int = 64) -> int:
        """Write points keyed by stable chunk_id: re-running overwrites, never duplicates."""
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        for v in vectors:
            if len(v) != self.dimensions:
                raise ValueError(f"vector size {len(v)} != {self.dimensions}")
        points = [models.PointStruct(
            id=c.chunk_id, vector=v,
            payload={"chunk_id": c.chunk_id, "document_id": c.document_id, "source": c.source,
                     "page": c.page, "text": c.text, **c.metadata})
            for c, v in zip(chunks, vectors)]
        name = self.collection(strategy)
        for i in range(0, len(points), batch_size):
            self.client.upsert(name, points=points[i:i + batch_size], wait=True)
        return len(points)

    def delete_ids(self, strategy: str, ids: list[str]) -> int:
        """Remove obsolete chunks (e.g. after the source document changed)."""
        if ids:
            self.client.delete(self.collection(strategy),
                               points_selector=models.PointIdsList(points=ids), wait=True)
        return len(ids)

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Filter on indexed payload fields in Qdrant, rank by cosine, return top-k."""
        flt = None
        if filters:
            unknown = set(filters) - set(FILTERABLE)
            if unknown:
                raise ValueError(f"filter on non-indexed field(s): {unknown}")
            flt = models.Filter(must=[models.FieldCondition(key=k, match=models.MatchValue(value=v))
                                      for k, v in filters.items()])
        res = self.client.query_points(
            self.collection(strategy), query=query_vector, query_filter=flt, limit=top_k,
            with_payload=True, search_params=models.SearchParams(exact=True))
        return [{"chunk_id": str(p.id), "score": p.score, "text": p.payload.get("text"),
                 "metadata": {k: v for k, v in p.payload.items() if k != "text"}}
                for p in res.points]
