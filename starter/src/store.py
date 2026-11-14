"""Store and search vectors with metadata filtering using PostgreSQL + pgvector or Qdrant Persistent."""
import json
import uuid
from pathlib import Path
from typing import Any

from .models import Chunk

ALLOWED_FILTER_KEYS = {"category", "year", "section", "document_id", "page", "source"}


class PgVectorStore:
    """PostgreSQL + pgvector store supporting idempotent upsert, cosine search, and SQL metadata filtering."""

    def __init__(self, database_url: str, dimensions: int) -> None:
        if not database_url or not database_url.strip():
            raise ValueError("database_url must be a non-empty connection string")
        if dimensions <= 0:
            raise ValueError(f"dimensions must be positive, got {dimensions}")
        self.database_url = database_url.strip()
        self.dimensions = int(dimensions)

        try:
            import psycopg  # type: ignore
            self._psycopg = psycopg
        except ImportError as exc:
            raise RuntimeError(
                "psycopg is not installed. Install psycopg[binary]==3.3.8 or use QdrantPersistentStore."
            ) from exc

    def _connect(self):
        return self._psycopg.connect(self.database_url, connect_timeout=5)

    def create_schema(self) -> None:
        """Create pgvector extension, document_chunks table, and cosine HNSW + metadata indexes."""
        ddl = f"""
        CREATE EXTENSION IF NOT EXISTS vector;

        CREATE TABLE IF NOT EXISTS document_chunks (
            strategy TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            source TEXT NOT NULL,
            page INTEGER NOT NULL,
            section TEXT,
            category TEXT,
            year INTEGER,
            text TEXT NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            embedding vector({self.dimensions}) NOT NULL,
            PRIMARY KEY (strategy, chunk_id)
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_strategy_meta
            ON document_chunks (strategy, category, year, document_id, page);

        CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
            ON document_chunks USING hnsw (embedding vector_cosine_ops);
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
            conn.commit()

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Validate dimensions, remove obsolete chunks for re-ingested documents, and upsert idempotently."""
        if not strategy or not strategy.strip():
            raise ValueError("strategy must be non-empty")
        if len(chunks) != len(vectors):
            raise ValueError(f"Length mismatch: {len(chunks)} chunks vs {len(vectors)} vectors")
        if not chunks:
            return

        for idx, vec in enumerate(vectors):
            if len(vec) != self.dimensions:
                raise ValueError(
                    f"Vector at index {idx} has dimension {len(vec)}, expected {self.dimensions}"
                )

        doc_ids = sorted({c.document_id for c in chunks})
        chunk_ids = [c.chunk_id for c in chunks]

        with self._connect() as conn:
            with conn.cursor() as cur:
                # Remove obsolete chunks belonging to re-ingested documents that are no longer in chunks
                cur.execute(
                    """
                    DELETE FROM document_chunks
                    WHERE strategy = %s
                      AND document_id = ANY(%s)
                      AND NOT (chunk_id = ANY(%s))
                    """,
                    (strategy, doc_ids, chunk_ids),
                )

                upsert_sql = """
                INSERT INTO document_chunks (
                    strategy, chunk_id, document_id, source, page,
                    section, category, year, text, metadata, embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
                ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                    document_id = EXCLUDED.document_id,
                    source = EXCLUDED.source,
                    page = EXCLUDED.page,
                    section = EXCLUDED.section,
                    category = EXCLUDED.category,
                    year = EXCLUDED.year,
                    text = EXCLUDED.text,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding;
                """
                for chunk, vec in zip(chunks, vectors):
                    meta_json = json.dumps(chunk.metadata, ensure_ascii=False)
                    vec_literal = "[" + ",".join(f"{float(x):.8f}" for x in vec) + "]"
                    cur.execute(
                        upsert_sql,
                        (
                            strategy,
                            chunk.chunk_id,
                            chunk.document_id,
                            chunk.source,
                            int(chunk.page),
                            chunk.metadata.get("section"),
                            chunk.metadata.get("category"),
                            chunk.metadata.get("year"),
                            chunk.text,
                            meta_json,
                            vec_literal,
                        ),
                    )
            conn.commit()

    def count_chunks(self, strategy: str) -> int:
        """Return total stored chunks for a given strategy."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM document_chunks WHERE strategy = %s",
                    (strategy,),
                )
                row = cur.fetchone()
                return int(row[0]) if row else 0

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Filter permitted metadata in SQL, rank by cosine distance (<=>), and return top-k."""
        if len(query_vector) != self.dimensions:
            raise ValueError(
                f"Query vector dimension {len(query_vector)} does not match store dimension {self.dimensions}"
            )
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        vec_literal = "[" + ",".join(f"{float(x):.8f}" for x in query_vector) + "]"
        where_clauses = ["strategy = %s"]
        params: list[Any] = [vec_literal, strategy]

        if filters:
            for key, val in filters.items():
                if key not in ALLOWED_FILTER_KEYS:
                    raise ValueError(f"Unsupported or unsafe metadata filter key: {key}")
                if key in {"category", "section", "document_id", "source", "page", "year"}:
                    where_clauses.append(f"{key} = %s")
                    params.append(val)
                else:
                    where_clauses.append("metadata ->> %s = %s")
                    params.extend([key, str(val)])

        params.extend([vec_literal, int(top_k)])
        where_sql = " AND ".join(where_clauses)

        sql = f"""
        SELECT
            chunk_id,
            document_id,
            source,
            page,
            text,
            1 - (embedding <=> %s::vector) AS cosine_similarity,
            metadata
        FROM document_chunks
        WHERE {where_sql}
        ORDER BY embedding <=> %s::vector ASC
        LIMIT %s;
        """

        results: list[dict[str, Any]] = []
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                for row in cur.fetchall():
                    meta = row[6] if isinstance(row[6], dict) else json.loads(row[6])
                    results.append(
                        {
                            "chunk_id": row[0],
                            "document_id": row[1],
                            "source": row[2],
                            "page": int(row[3]),
                            "text": row[4],
                            "score": round(float(row[5]), 6),
                            "metadata": meta,
                        }
                    )
        return results


class QdrantPersistentStore:
    """Persistent local Qdrant vector store with identical contract to PgVectorStore."""

    COLLECTION_NAME = "day06_document_chunks"

    def __init__(self, storage_path: str | Path, dimensions: int) -> None:
        if dimensions <= 0:
            raise ValueError(f"dimensions must be positive, got {dimensions}")
        self.storage_path = Path(storage_path)
        self.dimensions = int(dimensions)

        try:
            from qdrant_client import QdrantClient
            from qdrant_client import models as qmodels
        except ImportError as exc:
            raise RuntimeError("qdrant-client is not installed.") from exc

        self._qmodels = qmodels
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=str(self.storage_path))

    def close(self) -> None:
        self._client.close()

    def create_schema(self) -> None:
        """Create persistent Qdrant collection configured with Cosine distance if not exists."""
        if not self._client.collection_exists(self.COLLECTION_NAME):
            self._client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=self._qmodels.VectorParams(
                    size=self.dimensions,
                    distance=self._qmodels.Distance.COSINE,
                ),
            )

    @staticmethod
    def _point_uuid(strategy: str, chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{strategy}::{chunk_id}"))

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Validate lengths/dimensions, purge obsolete chunks for re-ingested docs, and upsert deterministically."""
        if not strategy or not strategy.strip():
            raise ValueError("strategy must be non-empty")
        if len(chunks) != len(vectors):
            raise ValueError(f"Length mismatch: {len(chunks)} chunks vs {len(vectors)} vectors")
        if not chunks:
            return

        for idx, vec in enumerate(vectors):
            if len(vec) != self.dimensions:
                raise ValueError(
                    f"Vector at index {idx} has dimension {len(vec)}, expected {self.dimensions}"
                )

        doc_ids = sorted({c.document_id for c in chunks})
        new_chunk_ids = {c.chunk_id for c in chunks}

        # Find and delete any obsolete chunks for (strategy, doc_id) that are not in new_chunk_ids
        for doc_id in doc_ids:
            scroll_filter = self._qmodels.Filter(
                must=[
                    self._qmodels.FieldCondition(
                        key="strategy", match=self._qmodels.MatchValue(value=strategy)
                    ),
                    self._qmodels.FieldCondition(
                        key="document_id", match=self._qmodels.MatchValue(value=doc_id)
                    ),
                ]
            )
            existing_points, _ = self._client.scroll(
                collection_name=self.COLLECTION_NAME,
                scroll_filter=scroll_filter,
                limit=10000,
                with_payload=True,
                with_vectors=False,
            )
            obsolete_ids = [
                pt.id
                for pt in existing_points
                if pt.payload and pt.payload.get("chunk_id") not in new_chunk_ids
            ]
            if obsolete_ids:
                self._client.delete(
                    collection_name=self.COLLECTION_NAME,
                    points_selector=self._qmodels.PointIdsList(points=obsolete_ids),
                )

        points = []
        for chunk, vec in zip(chunks, vectors):
            point_id = self._point_uuid(strategy, chunk.chunk_id)
            payload = {
                "strategy": strategy,
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "source": chunk.source,
                "page": int(chunk.page),
                "section": chunk.metadata.get("section"),
                "category": chunk.metadata.get("category"),
                "year": chunk.metadata.get("year"),
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            points.append(
                self._qmodels.PointStruct(
                    id=point_id,
                    vector=vec,
                    payload=payload,
                )
            )

        self._client.upsert(collection_name=self.COLLECTION_NAME, points=points)

    def count_chunks(self, strategy: str) -> int:
        """Count stored chunks for a strategy to verify persistence and idempotency."""
        count_res = self._client.count(
            collection_name=self.COLLECTION_NAME,
            count_filter=self._qmodels.Filter(
                must=[
                    self._qmodels.FieldCondition(
                        key="strategy", match=self._qmodels.MatchValue(value=strategy)
                    )
                ]
            ),
            exact=True,
        )
        return int(count_res.count)

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Filter permitted metadata in Qdrant payload, rank by Cosine similarity, and return top-k."""
        if len(query_vector) != self.dimensions:
            raise ValueError(
                f"Query vector dimension {len(query_vector)} does not match store dimension {self.dimensions}"
            )
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        must_conditions = [
            self._qmodels.FieldCondition(
                key="strategy", match=self._qmodels.MatchValue(value=strategy)
            )
        ]
        if filters:
            for key, val in filters.items():
                if key not in ALLOWED_FILTER_KEYS:
                    raise ValueError(f"Unsupported or unsafe metadata filter key: {key}")
                must_conditions.append(
                    self._qmodels.FieldCondition(
                        key=key,
                        match=self._qmodels.MatchValue(value=val),
                    )
                )

        query_filter = self._qmodels.Filter(must=must_conditions)
        response = self._client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        hits: list[dict[str, Any]] = []
        for pt in response.points:
            payload = pt.payload or {}
            hits.append(
                {
                    "chunk_id": payload.get("chunk_id"),
                    "document_id": payload.get("document_id"),
                    "source": payload.get("source"),
                    "page": int(payload.get("page", 0)),
                    "text": payload.get("text", ""),
                    "score": round(float(pt.score), 6),
                    "metadata": payload.get("metadata", {}),
                }
            )
        return hits
