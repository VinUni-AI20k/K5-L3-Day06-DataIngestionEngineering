"""Persist strategy snapshots and retrieve chunks using pgvector cosine distance."""
import math
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .models import Chunk


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        if not isinstance(dimensions, int) or not 1 <= dimensions <= 16000:
            raise ValueError("Embedding dimensions are outside the pgvector range.")
        self.database_url = database_url
        self.dimensions = dimensions

    def _connect(self):
        return psycopg.connect(self.database_url, connect_timeout=5, row_factory=dict_row)

    def create_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(f"""
                CREATE TABLE IF NOT EXISTS day06_chunks (
                    strategy text NOT NULL,
                    chunk_id text NOT NULL,
                    document_id text NOT NULL,
                    source text NOT NULL,
                    page integer NOT NULL,
                    text text NOT NULL,
                    metadata jsonb NOT NULL,
                    embedding vector({self.dimensions}) NOT NULL,
                    PRIMARY KEY (strategy, chunk_id)
                )
            """)
            row = connection.execute("""
                SELECT format_type(a.atttypid, a.atttypmod) AS type
                FROM pg_attribute AS a
                WHERE a.attrelid = 'day06_chunks'::regclass AND a.attname = 'embedding'
            """).fetchone()
            if row["type"] != f"vector({self.dimensions})":
                raise ValueError("Existing day06_chunks uses a different embedding dimension. Use a separate lab database.")

    def _vector(self, values: list[float]) -> str:
        if (len(values) != self.dimensions or not all(math.isfinite(value) for value in values)
                or not any(values)):
            raise ValueError("Expected a finite nonzero vector with the configured dimensions.")
        return "[" + ",".join(str(float(value)) for value in values) + "]"

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Replace the strategy snapshot atomically, including obsolete chunks."""
        rows = [
            (strategy, chunk.chunk_id, chunk.document_id, chunk.source, chunk.page,
             chunk.text, Jsonb(chunk.metadata), self._vector(vector))
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(6006, hashtext(%s))", (strategy,))
            cursor.execute("DELETE FROM day06_chunks WHERE strategy = %s", (strategy,))
            cursor.executemany("""
                INSERT INTO day06_chunks
                (strategy, chunk_id, document_id, source, page, text, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
            """, rows)

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        with self._connect() as connection:
            return connection.execute("""
                SELECT chunk_id, document_id, source, page, text, metadata,
                       1 - (embedding <=> %s::vector) AS score
                FROM day06_chunks
                WHERE strategy = %s AND metadata @> %s
                ORDER BY score DESC, chunk_id
                LIMIT %s
            """, (self._vector(query_vector), strategy, Jsonb(filters or {}), top_k)).fetchall()
