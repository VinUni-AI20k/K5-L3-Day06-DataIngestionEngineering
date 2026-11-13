"""Run the HUST retrieval benchmark against the local pgvector database."""
import argparse
import json
import logging
import os
from collections import Counter
from pathlib import Path
from time import perf_counter

import psycopg
from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import MyEmbedder
from .evaluation import evidence_text, evaluate_query, summarize
from .ingestion import load_pdf
from .store import PgVectorStore


STARTER = Path(__file__).resolve().parents[1]
ROOT = STARTER.parent
SOURCE_ID = "hust-academic-regulations-2025"
PDF = STARTER / "data" / "corpus" / "HUST's ACADEMIC REGULATIONS-2025.pdf"
QUESTIONS = ROOT / "benchmark" / "questions.json"
RESULTS = ROOT / "results"
PROGRAM_LEVELS = {
    "undergraduate": range(13, 21),
    "engineer": range(21, 24),
    "masters": range(24, 29),
}


def _load_inputs():
    source = PDF.relative_to(ROOT).as_posix()
    pages = [page for page in load_pdf(PDF, SOURCE_ID, source) if page.page >= 5]
    if len(pages) < 30:
        raise ValueError(f"Expected at least 30 content pages, got {len(pages)}.")
    for page in pages:
        for level, numbers in PROGRAM_LEVELS.items():
            if page.page in numbers:
                page.metadata["program_level"] = level
                break

    questions = json.loads(QUESTIONS.read_text(encoding="utf-8-sig"))
    counts = Counter(question["split"] for question in questions)
    if counts["dev"] < 15 or counts["test"] < 5 or counts.keys() != {"dev", "test"}:
        raise ValueError("Expected at least 15 dev and 5 test questions.")
    if len({question["id"] for question in questions}) != len(questions):
        raise ValueError("Question IDs must be unique.")
    filtered = [question for question in questions if question.get("filter")]
    if len(filtered) < 3 or not any(question["split"] == "test" for question in filtered):
        raise ValueError("Expected at least three filter questions, including one in test.")
    by_source_page = {(page.document_id, page.page): page for page in pages}
    for question in questions:
        for gold in [question, *question.get("alternative_evidence", [])]:
            page = by_source_page.get((gold["gold_source_id"], gold["gold_page"]))
            quote = evidence_text(gold["evidence_quote"])
            if page is None or not quote or quote not in evidence_text(page.text):
                raise ValueError(f"{question['id']}: evidence does not match the cited PDF page.")
            if any(page.metadata.get(key) != value for key, value in question.get("filter", {}).items()):
                raise ValueError(f"{question['id']}: gold page does not match the requested filter.")
    return pages, questions


def _write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run(split, chunks, questions):
    database_url = os.getenv("DATABASE_URL")
    model_name = os.getenv("EMBEDDING_MODEL")
    if not database_url or not model_name:
        raise ValueError("Set DATABASE_URL and EMBEDDING_MODEL in starter/.env.")
    with psycopg.connect(database_url, connect_timeout=5) as connection:
        version = connection.execute(
            "SELECT current_setting('server_version'), extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
    if version is None:
        raise ValueError("The database is reachable but pgvector is missing. Run starter/init.sql there.")

    embedder = MyEmbedder(model_name)
    store = PgVectorStore(database_url, embedder.dimensions)
    store.create_schema()
    stats = {}
    for strategy in ("A", "B", "C"):
        started = perf_counter()
        if strategy != "C":
            vectors = embedder.embed_documents([chunk.text for chunk in chunks[strategy]])
        store.upsert(strategy, chunks[strategy], vectors)
        lengths = [len(chunk.text) for chunk in chunks[strategy]]
        stats[strategy] = {
            "chunks": len(lengths), "min_chars": min(lengths), "max_chars": max(lengths),
            "mean_chars": sum(lengths) / len(lengths), "total_chars": sum(lengths),
            "ingest_seconds": round(perf_counter() - started, 3),
            "reuses_embeddings_from": "B" if strategy == "C" else None,
        }

    query_vectors = embedder.embed_documents([question["question"] for question in questions])
    traces = []
    for question, vector in zip(questions, query_vectors, strict=True):
        for strategy in ("A", "B", "C"):
            filters = question.get("filter") if strategy == "C" else None
            hits = store.search(strategy, vector, filters=filters)
            traces.append({
                **evaluate_query(question, hits), "strategy": strategy, "applied_filter": filters,
            })

    summary = summarize(traces)
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / f"trace_{split}.jsonl").open("w", encoding="utf-8") as output:
        for trace in traces:
            output.write(json.dumps(trace, ensure_ascii=False) + "\n")
    _write_json(RESULTS / f"metrics_{split}.json", summary)
    _write_json(RESULTS / f"ingestion_{split}.json", {
        "source": PDF.relative_to(ROOT).as_posix(), "split": split, "model": model_name,
        "provider": embedder.provider, "dimensions": embedder.dimensions,
        "postgresql": version[0], "pgvector": version[1], "search": "exact cosine",
        "fixed_size": 500, "overlap": 50, "paragraph_max_size": 500,
        "strategies": stats,
        "api_cost": 0 if embedder.provider == "local" else None,
    })
    for strategy, splits in summary.items():
        scores = splits[split]
        n = scores["questions"]
        print(f"{strategy} {split}: Hit@1 {scores['hit_at_1']}/{n}, "
              f"Evidence Hit@3 {scores['evidence_hit_at_3']}/{n}")
    print(f"Results: {RESULTS}")


def main():
    load_dotenv(STARTER / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"))
    parser.add_argument("--validate-only", action="store_true", help="Check the PDF, gold evidence and chunks offline.")
    args = parser.parse_args()
    split = args.split or os.getenv("EVALUATION_SPLIT") or "dev"
    if split not in {"dev", "test"}:
        parser.error("EVALUATION_SPLIT must be dev or test.")

    try:
        pages, questions = _load_inputs()
        fixed = [chunk for page in pages for chunk in fixed_size(page, 500, 50)]
        paragraphs = [chunk for page in pages for chunk in by_paragraph(page, 500)]
        print(f"Validated {len(pages)} PDF content pages and {len(questions)} questions. "
              f"Chunks: A={len(fixed)}, B=C={len(paragraphs)}.")
        if not args.validate_only:
            _run(split, {"A": fixed, "B": paragraphs, "C": paragraphs},
                 [question for question in questions if question["split"] == split])
    except psycopg.OperationalError:
        parser.exit(1, "PostgreSQL connection failed. Check Docker and DATABASE_URL in starter/.env.\n")
    except (ValueError, OSError, psycopg.Error) as error:
        logging.error("%s", error)
        parser.exit(1)


if __name__ == "__main__":
    main()
