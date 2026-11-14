"""Lưu / truy xuất vector bằng PostgreSQL + pgvector (cosine).

Không phụ thuộc package Python `pgvector`: truyền vector dạng '[1,2,...]'::vector.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .models import Chunk

logger = logging.getLogger(__name__)

# Chỉ cho phép filter theo cột đã biết — tránh SQL injection / cột lạ.
ALLOWED_FILTER_KEYS = {"pillar", "document_id", "source", "page"}


def _vector_literal(values: list[float]) -> str:
    """Đổi list[float] → literal pgvector, ví dụ '[0.1,0.2,…]'."""
    return "[" + ",".join(f"{x:.8f}" for x in values) + "]"


class PgVectorStore:
    """Một bảng `rag_chunks` dùng chung cho strategy A/B/C (cột strategy)."""

    def __init__(self, database_url: str, dimensions: int) -> None:
        if not database_url:
            raise ValueError("database_url trống — kiểm tra DATABASE_URL trong .env")
        if dimensions <= 0:
            raise ValueError(f"dimensions phải > 0, nhận được {dimensions}")

        self.database_url = database_url
        self.dimensions = dimensions

        # Fail-fast nếu Docker/pgvector chưa sẵn sàng (không fallback mock)
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
                )
                row = cur.fetchone()
                if not row:
                    raise RuntimeError(
                        "pgvector extension chưa bật. "
                        "Chạy docker compose / xem starter/README.md"
                    )
                logger.info(
                    "PgVectorStore OK | dim=%s | pgvector=%s", dimensions, row[0]
                )

    def create_schema(self) -> None:
        """Tạo bảng + index. Idempotent (IF NOT EXISTS)."""
        # vector(N) phải khớp đúng số chiều model (bge-m3 = 1024)
        ddl = f"""
        CREATE EXTENSION IF NOT EXISTS vector;

        CREATE TABLE IF NOT EXISTS rag_chunks (
            strategy     TEXT        NOT NULL,
            chunk_id     TEXT        NOT NULL,
            document_id  TEXT        NOT NULL,
            source       TEXT        NOT NULL,
            page         INTEGER     NOT NULL,
            text         TEXT        NOT NULL,
            pillar       TEXT,
            metadata     JSONB       NOT NULL DEFAULT '{{}}'::jsonb,
            embedding    vector({self.dimensions}) NOT NULL,
            PRIMARY KEY (strategy, chunk_id)
        );

        CREATE INDEX IF NOT EXISTS idx_rag_chunks_strategy_pillar
            ON rag_chunks (strategy, pillar);
        CREATE INDEX IF NOT EXISTS idx_rag_chunks_strategy_page
            ON rag_chunks (strategy, page);

        -- HNSW + cosine: ORDER BY embedding <=> query
        CREATE INDEX IF NOT EXISTS idx_rag_chunks_embedding_hnsw
            ON rag_chunks
            USING hnsw (embedding vector_cosine_ops);
        """
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
            conn.commit()
        logger.info("Schema rag_chunks sẵn sàng (vector(%s))", self.dimensions)

    def upsert(
        self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None:
        """Ghi chunk + vector theo strategy; xóa chunk cũ không còn trong batch.

        Re-ingest cùng corpus → không nhân đôi (PK = strategy + chunk_id).
        Đổi nguồn → chunk_id cũ biến mất khỏi batch → bị DELETE.
        """
        if strategy not in {"A", "B", "C"}:
            logger.warning("strategy=%r khác A/B/C — vẫn lưu", strategy)

        if len(chunks) != len(vectors):
            raise ValueError(
                f"Số chunk ({len(chunks)}) ≠ số vector ({len(vectors)})"
            )

        for i, vec in enumerate(vectors):
            if len(vec) != self.dimensions:
                raise ValueError(
                    f"Vector[{i}] dim={len(vec)} nhưng store khai báo {self.dimensions}"
                )

        keep_ids = [c.chunk_id for c in chunks]

        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                # 1) Xóa obsolete của strategy này
                if keep_ids:
                    cur.execute(
                        """
                        DELETE FROM rag_chunks
                        WHERE strategy = %s
                          AND NOT (chunk_id = ANY(%s))
                        """,
                        (strategy, keep_ids),
                    )
                else:
                    cur.execute(
                        "DELETE FROM rag_chunks WHERE strategy = %s",
                        (strategy,),
                    )

                # 2) Upsert từng dòng
                for chunk, vec in zip(chunks, vectors):
                    pillar = (chunk.metadata or {}).get("pillar")
                    cur.execute(
                        """
                        INSERT INTO rag_chunks (
                            strategy, chunk_id, document_id, source, page,
                            text, pillar, metadata, embedding
                        ) VALUES (
                            %s, %s, %s, %s, %s,
                            %s, %s, %s, %s::vector
                        )
                        ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                            document_id = EXCLUDED.document_id,
                            source      = EXCLUDED.source,
                            page        = EXCLUDED.page,
                            text        = EXCLUDED.text,
                            pillar      = EXCLUDED.pillar,
                            metadata    = EXCLUDED.metadata,
                            embedding   = EXCLUDED.embedding
                        """,
                        (
                            strategy,
                            chunk.chunk_id,
                            chunk.document_id,
                            chunk.source,
                            chunk.page,
                            chunk.text,
                            pillar,
                            Jsonb(chunk.metadata or {}),
                            _vector_literal(vec),
                        ),
                    )
            conn.commit()

        logger.info("upsert strategy=%s | %s chunks", strategy, len(chunks))

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Top-k theo cosine; optional filter metadata (vd. {"pillar": "Environment"}).

        Trả về dict: chunk_id, text, score, page, source, document_id, pillar, metadata.
        score ≈ cosine similarity = 1 - (embedding <=> query).
        """
        if len(query_vector) != self.dimensions:
            raise ValueError(
                f"query_vector dim={len(query_vector)} ≠ store dim={self.dimensions}"
            )
        if top_k <= 0:
            raise ValueError("top_k phải > 0")

        # Dùng subquery q(v) để truyền query vector đúng 1 lần (score + ORDER BY)
        where_parts = ["c.strategy = %s"]
        params: list[Any] = [_vector_literal(query_vector), strategy]

        if filters:
            for key, value in filters.items():
                if key not in ALLOWED_FILTER_KEYS:
                    raise ValueError(
                        f"Filter key không được phép: {key!r}. "
                        f"Chỉ dùng {sorted(ALLOWED_FILTER_KEYS)}"
                    )
                if value is None:
                    continue
                # Tên cột = key đã whitelist → an toàn khi nội suy vào SQL
                where_parts.append(f"c.{key} = %s")
                params.append(value)

        params.append(top_k)

        sql = f"""
            SELECT
                c.chunk_id,
                c.document_id,
                c.source,
                c.page,
                c.text,
                c.pillar,
                c.metadata,
                1 - (c.embedding <=> q.v) AS score
            FROM rag_chunks AS c,
                 (SELECT %s::vector AS v) AS q
            WHERE {" AND ".join(where_parts)}
            ORDER BY c.embedding <=> q.v
            LIMIT %s
        """

        with psycopg.connect(self.database_url) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            meta = row["metadata"]
            if isinstance(meta, str):
                meta = json.loads(meta)
            results.append(
                {
                    "chunk_id": row["chunk_id"],
                    "document_id": row["document_id"],
                    "source": row["source"],
                    "page": int(row["page"]),
                    "text": row["text"],
                    "pillar": row["pillar"],
                    "metadata": meta or {},
                    "score": float(row["score"]) if row["score"] is not None else None,
                }
            )
        return results

    def count(self, strategy: str | None = None) -> int:
        """Đếm chunk — kiểm tra re-ingest không nhân đôi."""
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                if strategy is None:
                    cur.execute("SELECT COUNT(*) FROM rag_chunks")
                else:
                    cur.execute(
                        "SELECT COUNT(*) FROM rag_chunks WHERE strategy = %s",
                        (strategy,),
                    )
                return int(cur.fetchone()[0])
