"""Persistent vector storage and cosine search using PostgreSQL + pgvector."""
import math
from collections import defaultdict
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from .models import Chunk


class PgVectorStore:
    MAX_TOP_K = 1000

    def __init__(self, database_url: str, dimensions: int) -> None:
        if not isinstance(database_url, str) or not database_url.strip():
            raise ValueError("database_url must be a non-empty string")
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 1:
            raise ValueError("dimensions must be a positive integer")

        self.database_url = database_url
        self.dimensions = dimensions

    def create_schema(self) -> None:
        """Create the extension, vector table, and metadata/filter indexes."""
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS ingestion_chunks (
                        strategy TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        document_id TEXT NOT NULL,
                        source TEXT NOT NULL,
                        page INTEGER NOT NULL CHECK (page > 0),
                        text TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding VECTOR({self.dimensions}) NOT NULL,
                        PRIMARY KEY (strategy, chunk_id)
                    )
                    """
                )
                cursor.execute(
                    """
                    SELECT format_type(attribute.atttypid, attribute.atttypmod)
                    FROM pg_attribute AS attribute
                    WHERE attribute.attrelid = 'ingestion_chunks'::regclass
                      AND attribute.attname = 'embedding'
                      AND NOT attribute.attisdropped
                    """
                )
                row = cursor.fetchone()
                expected_type = f"vector({self.dimensions})"
                if row is None or row[0] != expected_type:
                    actual_type = row[0] if row is not None else "missing"
                    raise RuntimeError(
                        "Existing ingestion_chunks embedding column has type "
                        f"{actual_type}; expected {expected_type}. Use a new table "
                        "or intentionally migrate the existing embeddings."
                    )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS ingestion_chunks_document_idx
                    ON ingestion_chunks (strategy, document_id)
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS ingestion_chunks_metadata_idx
                    ON ingestion_chunks USING GIN (metadata)
                    """
                )

    def upsert(
        self,
        strategy: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        """Replace one strategy's chunks for each supplied document atomically.

        Call once with the complete chunk set for the strategy and all source
        documents. Existing chunks for those documents that are absent from this
        batch are deleted, so changed source/chunking data cannot leave stale rows.
        """
        if not isinstance(strategy, str) or not strategy.strip():
            raise ValueError("strategy must be a non-empty string")
        if not isinstance(chunks, list) or not isinstance(vectors, list):
            raise ValueError("chunks and vectors must be lists")
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return

        seen_ids = set()
        rows = []
        chunks_by_document: dict[str, list[str]] = defaultdict(list)

        for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
            if not isinstance(chunk, Chunk):
                raise ValueError(f"chunks[{index}] must be a Chunk")
            if not isinstance(chunk.chunk_id, str) or not chunk.chunk_id.strip():
                raise ValueError(f"chunks[{index}] has an invalid chunk_id")
            if chunk.chunk_id in seen_ids:
                raise ValueError(f"duplicate chunk_id in input: {chunk.chunk_id}")
            seen_ids.add(chunk.chunk_id)
            if not isinstance(chunk.document_id, str) or not chunk.document_id.strip():
                raise ValueError(f"chunks[{index}] has an invalid document_id")
            if not isinstance(chunk.source, str) or not chunk.source.strip():
                raise ValueError(f"chunks[{index}] has an invalid source")
            if isinstance(chunk.page, bool) or not isinstance(chunk.page, int) or chunk.page < 1:
                raise ValueError(f"chunks[{index}] page must be a positive integer")
            if not isinstance(chunk.text, str) or not chunk.text.strip():
                raise ValueError(f"chunks[{index}] has empty text")
            if not isinstance(chunk.metadata, dict):
                raise ValueError(f"chunks[{index}] metadata must be a dictionary")
            if not isinstance(vector, list) or len(vector) != self.dimensions:
                raise ValueError(
                    f"vectors[{index}] must contain exactly {self.dimensions} values"
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in vector
            ):
                raise ValueError(f"vectors[{index}] contains a non-finite value")
            if not any(value != 0 for value in vector):
                raise ValueError(f"vectors[{index}] must not be a zero vector")

            chunks_by_document[chunk.document_id].append(chunk.chunk_id)
            vector_literal = "[" + ",".join(repr(float(value)) for value in vector) + "]"
            rows.append(
                (
                    strategy,
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.source,
                    chunk.page,
                    chunk.text,
                    Jsonb(chunk.metadata),
                    vector_literal,
                )
            )

        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                for document_id, chunk_ids in chunks_by_document.items():
                    cursor.execute(
                        """
                        DELETE FROM ingestion_chunks
                        WHERE strategy = %s
                          AND document_id = %s
                          AND NOT (chunk_id = ANY(%s))
                        """,
                        (strategy, document_id, chunk_ids),
                    )

                cursor.executemany(
                    """
                    INSERT INTO ingestion_chunks (
                        strategy, chunk_id, document_id, source, page, text,
                        metadata, embedding
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                    ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                        document_id = EXCLUDED.document_id,
                        source = EXCLUDED.source,
                        page = EXCLUDED.page,
                        text = EXCLUDED.text,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding
                    """,
                    rows,
                )

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return exact top-k cosine matches, applying JSON metadata filters."""
        if not isinstance(strategy, str) or not strategy.strip():
            raise ValueError("strategy must be a non-empty string")
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= self.MAX_TOP_K
        ):
            raise ValueError(f"top_k must be an integer from 1 to {self.MAX_TOP_K}")
        if not isinstance(query_vector, list) or len(query_vector) != self.dimensions:
            raise ValueError(
                f"query_vector must contain exactly {self.dimensions} values"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in query_vector
        ):
            raise ValueError("query_vector contains a non-finite value")
        if not any(value != 0 for value in query_vector):
            raise ValueError("query_vector must not be a zero vector")
        if filters is None:
            filters = {}
        if not isinstance(filters, dict):
            raise ValueError("filters must be a dictionary or None")
        if any(not isinstance(key, str) or not key for key in filters):
            raise ValueError("filter keys must be non-empty strings")

        vector_literal = "[" + ",".join(repr(float(value)) for value in query_vector) + "]"
        clauses = ["strategy = %s"]
        params: list[Any] = [strategy]
        for key, value in filters.items():
            clauses.append("metadata @> %s::jsonb")
            params.append(Jsonb({key: value}))

        where_clause = " AND ".join(clauses)
        params.extend((vector_literal, top_k))
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT chunk_id, document_id, source, page, text, metadata,
                           1 - (embedding <=> %s::vector) AS score
                    FROM ingestion_chunks
                    WHERE {where_clause}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (vector_literal, *params),
                )
                rows = cursor.fetchall()

        return [
            {
                "chunk_id": row[0],
                "document_id": row[1],
                "source": row[2],
                "page": row[3],
                "text": row[4],
                "metadata": row[5],
                "score": float(row[6]),
            }
            for row in rows
        ]
