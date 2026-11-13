"""Store and search vectors in PostgreSQL + pgvector.

Driver: pg8000 (pure Python). psycopg-binary's compiled DLL is blocked by Windows
Smart App Control on the author's machine; pg8000 needs no native library.
Search is exact (sequential scan, no ANN index), ranked by cosine distance (<=>).
"""
from __future__ import annotations

import json
from typing import Any
from urllib.parse import unquote, urlparse

import pg8000.native

from .models import Chunk

# Metadata keys a query may filter on (whitelist: keys are interpolated into SQL).
FILTERABLE = {"edition", "part", "language"}


def _vec(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in v) + "]"


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int, table: str = "chunks") -> None:
        if not 0 < dimensions <= 16000:
            raise ValueError("pgvector supports 1..16000 dimensions")
        u = urlparse(database_url)
        self.conn = pg8000.native.Connection(
            unquote(u.username or ""), password=unquote(u.password or ""),
            host=u.hostname, port=u.port or 5432, database=u.path.lstrip("/"))
        self.dimensions = dimensions
        self.table = table

    def create_schema(self) -> None:
        self.conn.run("CREATE EXTENSION IF NOT EXISTS vector")
        self.conn.run(f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                strategy     text    NOT NULL,
                chunk_id     text    NOT NULL,
                document_id  text    NOT NULL,
                source       text    NOT NULL,
                page         integer NOT NULL,
                metadata     jsonb   NOT NULL,
                text         text    NOT NULL,
                embedding    vector({self.dimensions}) NOT NULL,
                updated_at   timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (strategy, chunk_id))""")
        dims = self.conn.run(
            "SELECT atttypmod FROM pg_attribute WHERE attrelid = CAST(:t AS regclass) "
            "AND attname = 'embedding'", t=self.table)
        if dims and dims[0][0] != self.dimensions:
            raise RuntimeError(f"Table {self.table} has vector({dims[0][0]}), "
                               f"model gives {self.dimensions}")

    def existing_ids(self, strategy: str) -> set[str]:
        rows = self.conn.run(f"SELECT chunk_id FROM {self.table} WHERE strategy = :s",
                             s=strategy)
        return {r[0] for r in rows}

    def count(self, strategy: str) -> int:
        return self.conn.run(f"SELECT count(*) FROM {self.table} WHERE strategy = :s",
                             s=strategy)[0][0]

    def remove_obsolete(self, strategy: str, chunks: list[Chunk]) -> int:
        """Delete stored chunks of the re-ingested documents that are no longer produced."""
        keep = {c.chunk_id for c in chunks}
        removed = 0
        for doc in sorted({c.document_id for c in chunks}):
            rows = self.conn.run(
                f"SELECT chunk_id FROM {self.table} WHERE strategy = :s AND document_id = :d",
                s=strategy, d=doc)
            stale = [r[0] for r in rows if r[0] not in keep]
            for cid in stale:
                self.conn.run(f"DELETE FROM {self.table} WHERE strategy = :s AND chunk_id = :c",
                              s=strategy, c=cid)
            removed += len(stale)
        return removed

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Insert or update by (strategy, chunk_id); stable IDs make re-runs idempotent."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors differ in length")
        if any(len(v) != self.dimensions for v in vectors):
            raise ValueError("vector dimension mismatch")
        self.conn.run("BEGIN")
        try:
            for c, v in zip(chunks, vectors):
                self.conn.run(
                    f"""INSERT INTO {self.table}
                        (strategy, chunk_id, document_id, source, page, metadata, text, embedding)
                        VALUES (:s, :id, :doc, :src, :page, CAST(:meta AS jsonb), :text,
                                CAST(:vec AS vector))
                        ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                          metadata = EXCLUDED.metadata, text = EXCLUDED.text,
                          embedding = EXCLUDED.embedding, updated_at = now()""",
                    s=strategy, id=c.chunk_id, doc=c.document_id, src=c.source, page=c.page,
                    meta=json.dumps(c.metadata, ensure_ascii=False), text=c.text, vec=_vec(v))
            self.conn.run("COMMIT")
        except Exception:
            self.conn.run("ROLLBACK")
            raise

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if len(query_vector) != self.dimensions:
            raise ValueError("query vector dimension mismatch")
        where, params = ["strategy = :s"], {"s": strategy, "q": _vec(query_vector), "k": top_k}
        for i, (key, value) in enumerate((filters or {}).items()):
            if key not in FILTERABLE:
                raise ValueError(f"filter key {key!r} not allowed")
            where.append(f"metadata->>'{key}' = :f{i}")
            params[f"f{i}"] = str(value)
        rows = self.conn.run(
            f"""SELECT chunk_id, document_id, source, page, metadata, text,
                       1 - (embedding <=> CAST(:q AS vector)) AS score
                FROM {self.table} WHERE {' AND '.join(where)}
                ORDER BY embedding <=> CAST(:q AS vector) LIMIT :k""", **params)
        return [{"chunk_id": r[0], "document_id": r[1], "source": r[2], "page": r[3],
                 "metadata": r[4] if isinstance(r[4], dict) else json.loads(r[4]),
                 "text": r[5], "score": round(float(r[6]), 4)} for r in rows]

    def close(self) -> None:
        self.conn.close()
