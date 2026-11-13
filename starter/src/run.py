"""Day 06 pipeline: ingest corpus into pgvector, then evaluate strategies A/B/C.

From starter/:
    python -m src.run ingest              # chunk + embed + upsert A and B (idempotent)
    python -m src.run evaluate --split dev
    python -m src.run evaluate --split test   # once, after choosing on dev

Strategies (same corpus, same embedding model, same benchmark):
    A  fixed-size 500 chars, overlap 50, no filter
    B  paragraph-packed, max 800 chars, no filter
    C  = B's index (same chunks/vectors) + metadata filter when the question gives one
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from dotenv import load_dotenv

from . import chunking
from .embeddings import OpenAIEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_corpus
from .store import PgVectorStore

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT.parent / "benchmark"
RESULTS = ROOT / "results"

INDEXES = {
    "A": lambda page: chunking.fixed_size(page, size=500, overlap=50),
    "B": lambda page: chunking.by_paragraph(page, max_size=800),
}
# strategy -> (index to search, apply question filter?)
STRATEGIES = {"A": ("A", False), "B": ("B", False), "C": ("B", True)}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _env() -> dict:
    return {"python": platform.python_version(), "pypdf": version("pypdf"),
            "pg8000": version("pg8000"), "openai": version("openai")}


def _check_frozen_benchmark() -> dict:
    # Hash with LF line endings so a CRLF checkout (git autocrlf on Windows) still matches.
    data = (BENCH / "questions.json").read_bytes().replace(b"\r\n", b"\n")
    digest = hashlib.sha256(data).hexdigest()
    if digest not in (BENCH / "FROZEN.md").read_text(encoding="utf-8"):
        raise SystemExit(f"benchmark/questions.json changed after freezing (sha256 {digest})")
    return json.loads(data)


def ingest(embedder: OpenAIEmbedder, store: PgVectorStore) -> None:
    pages, events = load_corpus(ROOT / "data" / "manifest.json", ROOT)
    if len(pages) < 30:
        raise SystemExit(f"only {len(pages)} content pages; need >= 30")
    store.create_schema()
    for name, chunker in INDEXES.items():
        t0 = time.perf_counter()
        chunks = [c for p in pages for c in chunker(p)]
        existing = store.existing_ids(name)
        todo = [c for c in chunks if c.chunk_id not in existing]
        tokens_before = embedder.total_tokens
        vectors = embedder.embed_documents([c.text for c in todo]) if todo else []
        store.upsert(name, todo, vectors)
        removed = store.remove_obsolete(name, chunks)
        lengths = [len(c.text) for c in chunks]
        record = {
            "time": _now(), "index": name, "pages": len(pages), "chunks": len(chunks),
            "embedded_now": len(todo), "already_stored": len(chunks) - len(todo),
            "removed_obsolete": removed, "rows_in_store": store.count(name),
            "total_chars": sum(lengths), "len_min": min(lengths), "len_median":
            int(statistics.median(lengths)), "len_max": max(lengths),
            "embed_tokens": embedder.total_tokens - tokens_before,
            "seconds": round(time.perf_counter() - t0, 1),
        }
        print(json.dumps(record, ensure_ascii=False))
        _append_jsonl(RESULTS / "ingest_runs.jsonl", record)
    (RESULTS / "ingest_events.json").write_text(
        json.dumps(events, ensure_ascii=False, indent=1), encoding="utf-8")


def evaluate(embedder: OpenAIEmbedder, store: PgVectorStore, split: str) -> None:
    bench = _check_frozen_benchmark()
    questions = [q for q in bench["questions"] if q["split"] == split]
    traces = []
    t0 = time.perf_counter()
    qvecs = {q["id"]: embedder.embed_query(q["question"]) for q in questions}
    for strategy, (index, use_filter) in STRATEGIES.items():
        for q in questions:
            filters = q["filter"] if use_filter and q["filter"] else None
            hits = store.search(index, qvecs[q["id"]], top_k=3, filters=filters)
            trace = evaluate_query(q, hits)
            trace.update(strategy=strategy, index=index, applied_filter=filters)
            traces.append(trace)
    with (RESULTS / f"traces_{split}.jsonl").open("w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    rows = summarize(traces)
    with (RESULTS / f"summary_{split}.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    _append_jsonl(RESULTS / "eval_runs.jsonl", {
        "time": _now(), "split": split, "questions": len(questions),
        "query_tokens": embedder.total_tokens, "seconds": round(time.perf_counter() - t0, 1),
        "summary": rows})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--split", choices=["dev", "test"], required=True)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    load_dotenv(ROOT / ".env")
    RESULTS.mkdir(exist_ok=True)

    embedder = OpenAIEmbedder(os.environ["EMBEDDING_MODEL"])
    store = PgVectorStore(os.environ["DATABASE_URL"], embedder.dimensions)
    try:
        if args.cmd == "ingest":
            ingest(embedder, store)
        else:
            evaluate(embedder, store, args.split)
    finally:
        store.close()
    info = {"time": _now(), "cmd": args.cmd, "model": embedder.model_name,
            "dimensions": embedder.dimensions, "api_calls": embedder.calls,
            "tokens": embedder.total_tokens, "cost_usd": round(embedder.cost_usd, 6),
            "vector_store": "PostgreSQL 16 + pgvector (exact cosine search)", **_env()}
    print(json.dumps(info))
    _append_jsonl(RESULTS / "run_info.jsonl", info)


if __name__ == "__main__":
    main()
