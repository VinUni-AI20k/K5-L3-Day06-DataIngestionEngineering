"""Embed chunks and load them into Qdrant (strategies A and B; C reuses B).

Incremental and idempotent:
  - chunk ids are UUID5 of (strategy, document, position, content hash), so an
    unchanged chunk keeps its id across runs;
  - ids already in the collection are not re-embedded (no repeated API cost);
  - ids in the collection for this document but absent from the current
    chunking are deleted (handles a changed source document);
  - --force re-embeds and re-upserts everything with the same ids, to show
    that upsert overwrites instead of duplicating.
Each run appends one line per strategy to results/index_log.jsonl.

Run from starter/ after `python -m src.chunking`:
    python -m src.index            # A and B
    python -m src.index --force    # re-embed + re-upsert everything
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from .embeddings import OpenRouterEmbedder
from .models import Chunk
from .store import QdrantStore

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "starter" / "results"
LOG = RESULTS / "index_log.jsonl"


def load_chunks(strategy: str) -> list[Chunk]:
    path = RESULTS / f"chunks_{strategy}.jsonl"
    return [Chunk(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()]


def index_strategy(strategy: str, embedder: OpenRouterEmbedder, store: QdrantStore,
                   force: bool) -> dict:
    chunks = load_chunks(strategy)
    document_ids = {c.document_id for c in chunks}
    store.create_schema(strategy)
    before = store.count(strategy)
    existing = set().union(*(store.existing_ids(strategy, d) for d in document_ids))
    current = {c.chunk_id for c in chunks}

    todo = chunks if force else [c for c in chunks if c.chunk_id not in existing]
    cost0, tok0, t0 = embedder.cost_usd, embedder.tokens, time.perf_counter()
    vectors = embedder.embed_documents([c.text for c in todo]) if todo else []
    t_embed = time.perf_counter() - t0
    upserted = store.upsert(strategy, todo, vectors)
    deleted = store.delete_ids(strategy, sorted(existing - current))
    seconds = time.perf_counter() - t0
    after = store.count(strategy)

    row = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "strategy": strategy, "collection": store.collection(strategy), "force": force,
        "model": embedder.model_name, "dimensions": embedder.dimensions,
        "chunks_current": len(chunks), "points_before": before, "points_after": after,
        "embedded": len(todo), "upserted": upserted, "deleted_obsolete": deleted,
        "chars_embedded": sum(len(c.text) for c in todo),
        "tokens": embedder.tokens - tok0, "cost_usd": round(embedder.cost_usd - cost0, 6),
        "embed_seconds": round(t_embed, 1), "total_seconds": round(seconds, 1),
    }
    if after != len(chunks):
        raise RuntimeError(f"{strategy}: collection has {after} points, expected {len(chunks)}")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", nargs="+", default=["A", "B"], choices=["A", "B"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    load_dotenv(ROOT / "starter" / ".env")
    embedder = OpenRouterEmbedder()
    store = QdrantStore(os.environ["QDRANT_URL"], os.environ.get("QDRANT_API_KEY"),
                        embedder.dimensions)
    with LOG.open("a", encoding="utf-8") as f:
        for s in args.strategy:
            row = index_strategy(s, embedder, store, args.force)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
