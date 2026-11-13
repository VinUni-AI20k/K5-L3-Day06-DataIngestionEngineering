"""PostgreSQL + pgvector store: one table, a `strategy` column, cosine distance search."""
import json
from typing import Any

import psycopg

from .models import Chunk

ALLOWED_FILTERS = {'chapter': int}  # metadata keys that may be used as filters


def _vector_literal(vector: list[float]) -> str:
    return '[' + ','.join(repr(float(x)) for x in vector) + ']'


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        if not database_url:
            raise ValueError('DATABASE_URL is empty')
        if dimensions <= 0:
            raise ValueError('dimensions must be positive')
        self.database_url = database_url
        self.dimensions = dimensions

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self.database_url, autocommit=True, connect_timeout=5)

    def create_schema(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
            cur.execute(f'''
                CREATE TABLE IF NOT EXISTS chunks (
                    id          TEXT NOT NULL,
                    strategy    TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    source      TEXT NOT NULL,
                    page        INTEGER NOT NULL,
                    chapter     INTEGER,
                    text        TEXT NOT NULL,
                    metadata    JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding   vector({self.dimensions}) NOT NULL,
                    PRIMARY KEY (strategy, id)
                )''')
            cur.execute('CREATE INDEX IF NOT EXISTS chunks_strategy_idx ON chunks (strategy)')
            cur.execute('CREATE INDEX IF NOT EXISTS chunks_chapter_idx ON chunks (chapter)')
            cur.execute("SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                        "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'")
            (column_type,) = cur.fetchone()
            if column_type != f'vector({self.dimensions})':
                raise RuntimeError(f'table chunks has {column_type}, expected vector({self.dimensions})')

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Insert or update by stable ID; delete chunks of the same document that no longer exist."""
        if len(chunks) != len(vectors):
            raise ValueError(f'{len(chunks)} chunks but {len(vectors)} vectors')
        if any(len(v) != self.dimensions for v in vectors):
            raise ValueError(f'vector dimension must be {self.dimensions}')
        if len({c.chunk_id for c in chunks}) != len(chunks):
            raise ValueError('duplicate chunk_id in batch')

        rows = [(c.chunk_id, strategy, c.document_id, c.source, c.page, c.metadata.get('chapter'),
                 c.text, json.dumps(c.metadata, ensure_ascii=False), _vector_literal(v))
                for c, v in zip(chunks, vectors)]
        with self._connect() as conn, conn.cursor() as cur:
            with conn.transaction():
                cur.executemany(
                    '''INSERT INTO chunks (id, strategy, document_id, source, page, chapter, text, metadata, embedding)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
                       ON CONFLICT (strategy, id) DO UPDATE SET text = EXCLUDED.text, metadata = EXCLUDED.metadata,
                           embedding = EXCLUDED.embedding, chapter = EXCLUDED.chapter''',
                    rows)
                for document_id in {c.document_id for c in chunks}:
                    keep = [c.chunk_id for c in chunks if c.document_id == document_id]
                    cur.execute('DELETE FROM chunks WHERE strategy = %s AND document_id = %s AND NOT (id = ANY(%s))',
                                (strategy, document_id, keep))

    def count(self, strategy: str) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute('SELECT count(*) FROM chunks WHERE strategy = %s', (strategy,))
            return cur.fetchone()[0]

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if len(query_vector) != self.dimensions:
            raise ValueError(f'query vector dimension must be {self.dimensions}')
        clauses, params = ['strategy = %s'], [strategy]
        for key, value in (filters or {}).items():
            if key not in ALLOWED_FILTERS:
                raise ValueError(f'filter {key!r} is not allowed')
            if not isinstance(value, ALLOWED_FILTERS[key]):
                raise ValueError(f'filter {key!r} must be {ALLOWED_FILTERS[key].__name__}')
            clauses.append(f'{key} = %s')
            params.append(value)
        vec = _vector_literal(query_vector)
        sql = f'''SELECT id, text, document_id, source, page, chapter, metadata,
                         1 - (embedding <=> %s::vector) AS score
                  FROM chunks WHERE {' AND '.join(clauses)}
                  ORDER BY embedding <=> %s::vector LIMIT %s'''
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, [vec, *params, vec, top_k])
            cols = [d.name for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
