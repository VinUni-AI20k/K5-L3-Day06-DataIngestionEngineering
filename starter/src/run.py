"""Run the three-configuration Day 06 retrieval experiment."""
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import MODEL_DIMENSIONS, MyEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_jsonl, normalize_text
from .models import Chunk, DocumentPage
from .store import PgVectorStore


STARTER_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = STARTER_DIR / "data" / "luat-shtt-155-vbhn-vpqh-2025-units.jsonl"
MANIFEST_PATH = STARTER_DIR / "data" / "luat-shtt-155-vbhn-vpqh-2025-manifest.json"
QUESTIONS_PATH = STARTER_DIR / "benchmark" / "questions.jsonl"
RESULTS_DIR = STARTER_DIR / "results"
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-large"
MINIMUM_PAGES = 30
MINIMUM_QUESTIONS = 20
MINIMUM_DEV_QUESTIONS = 15
MINIMUM_TEST_QUESTIONS = 5
TOP_K = 3


def _read_questions(path: Path) -> list[dict[str, Any]]:
    questions = []
    seen_ids = set()
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                question = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}, line {line_number}: invalid JSON") from error
            if not isinstance(question, dict):
                raise ValueError(f"{path}, line {line_number}: expected a JSON object")
            question_id = question.get("id")
            if not isinstance(question_id, str) or not question_id.strip():
                raise ValueError(f"{path}, line {line_number}: missing question id")
            if question_id in seen_ids:
                raise ValueError(f"{path}, line {line_number}: duplicate id {question_id}")
            seen_ids.add(question_id)
            questions.append(question)
    if not questions:
        raise ValueError(f"{path}: no questions found")
    return questions


def _validate_manifest(path: Path, pages: list[DocumentPage]) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON manifest") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"{path}: manifest must be a JSON object")

    page_numbers = sorted(page.page for page in pages)
    document_ids = {page.document_id for page in pages}
    if len(document_ids) != 1 or manifest.get("source_id") not in document_ids:
        raise ValueError("Manifest source_id does not match the corpus document_id")
    if manifest.get("page_count") != len(page_numbers):
        raise ValueError("Manifest page_count does not match the corpus")
    if manifest.get("selected_source_pdf_pages") != page_numbers:
        raise ValueError("Manifest selected pages do not match the corpus pages")

    manifest_pages = manifest.get("pages")
    if not isinstance(manifest_pages, list):
        raise ValueError("Manifest pages must be a list")
    characters_by_page = {}
    for item in manifest_pages:
        if not isinstance(item, dict):
            raise ValueError("Manifest page entries must be objects")
        page_number = item.get("page")
        characters = item.get("characters")
        if (
            isinstance(page_number, bool)
            or not isinstance(page_number, int)
            or isinstance(characters, bool)
            or not isinstance(characters, int)
        ):
            raise ValueError("Manifest page and character counts must be integers")
        if page_number in characters_by_page:
            raise ValueError(f"Manifest repeats page {page_number}")
        characters_by_page[page_number] = characters
    if set(characters_by_page) != set(page_numbers):
        raise ValueError("Manifest page entries do not match the corpus pages")
    for page in pages:
        if characters_by_page[page.page] != len(page.text):
            raise ValueError(f"Manifest character count mismatch on page {page.page}")
    return manifest


def _gold_options(question: dict[str, Any]) -> list[tuple[str, int, str]]:
    options = [
        (
            question.get("gold_source_id"),
            question.get("gold_page"),
            question.get("evidence_quote"),
        )
    ]
    alternatives = question.get("gold_alternatives", [])
    if not isinstance(alternatives, list):
        raise ValueError(f"{question.get('id')}: gold_alternatives must be a list")
    for alternative in alternatives:
        if not isinstance(alternative, dict):
            raise ValueError(f"{question.get('id')}: invalid gold alternative")
        options.append(
            (
                alternative.get("gold_source_id", question.get("gold_source_id")),
                alternative.get("gold_page"),
                alternative.get("evidence_quote"),
            )
        )
    return options


def _normalized_text(text: str) -> str:
    return " ".join(normalize_text(text).split())


def _validate_inputs(
    pages: list[DocumentPage],
    questions: list[dict[str, Any]],
) -> None:
    page_by_key = {(page.document_id, page.page): page for page in pages}
    if len(page_by_key) < MINIMUM_PAGES:
        raise ValueError(
            f"Corpus has {len(page_by_key)} non-empty pages; at least "
            f"{MINIMUM_PAGES} are required"
        )
    if any(not page.text.strip() for page in pages):
        raise ValueError("Corpus contains an empty page")
    sections = {
        page.metadata.get("section")
        for page in pages
        if isinstance(page.metadata.get("section"), str)
        and page.metadata["section"].strip()
    }
    if len(sections) < 2:
        raise ValueError("Corpus must have at least two verified section values")

    if len(questions) < MINIMUM_QUESTIONS:
        raise ValueError(
            f"Benchmark has {len(questions)} questions; at least "
            f"{MINIMUM_QUESTIONS} are required"
        )
    split_counts = {
        split: sum(question.get("split") == split for question in questions)
        for split in ("dev", "test")
    }
    if split_counts["dev"] < MINIMUM_DEV_QUESTIONS:
        raise ValueError("Benchmark needs at least 15 dev questions")
    if split_counts["test"] < MINIMUM_TEST_QUESTIONS:
        raise ValueError("Benchmark needs at least 5 test questions")

    filtered_questions = [
        question for question in questions if question.get("filter") is not None
    ]
    if len(filtered_questions) < 3 or not any(
        question.get("split") == "test" for question in filtered_questions
    ):
        raise ValueError("Benchmark needs at least 3 filters, including one test filter")

    for question in questions:
        question_id = question["id"]
        if question.get("split") not in {"dev", "test"}:
            raise ValueError(f"{question_id}: split must be dev or test")
        if not isinstance(question.get("question"), str) or not question["question"].strip():
            raise ValueError(f"{question_id}: question must be non-empty text")
        if (
            not isinstance(question.get("expected_answer"), str)
            or not question["expected_answer"].strip()
        ):
            raise ValueError(f"{question_id}: expected_answer must be non-empty text")

        filters = question.get("filter")
        if filters is not None and (
            not isinstance(filters, dict) or not filters
        ):
            raise ValueError(f"{question_id}: filter must be a non-empty object or null")

        for source_id, page_number, evidence in _gold_options(question):
            if not isinstance(source_id, str) or not source_id:
                raise ValueError(f"{question_id}: invalid gold_source_id")
            if (
                isinstance(page_number, bool)
                or not isinstance(page_number, int)
                or page_number < 1
            ):
                raise ValueError(f"{question_id}: invalid gold_page")
            if not isinstance(evidence, str) or not evidence.strip():
                raise ValueError(f"{question_id}: evidence_quote must be non-empty")
            page = page_by_key.get((source_id, page_number))
            if page is None:
                raise ValueError(
                    f"{question_id}: gold page {source_id}/{page_number} "
                    "is not in the corpus"
                )
            if _normalized_text(evidence) not in _normalized_text(page.text):
                raise ValueError(
                    f"{question_id}: evidence quote not found on gold page "
                    f"{source_id}/{page_number}"
                )
            if filters and any(page.metadata.get(key) != value for key, value in filters.items()):
                raise ValueError(
                    f"{question_id}: filter values do not match gold page metadata"
                )


def _build_strategies(pages: list[DocumentPage]) -> dict[str, list[Chunk]]:
    strategies = {
        "A": [
            chunk
            for page in pages
            for chunk in fixed_size(page, size=500, overlap=50)
        ],
        "B": [
            chunk
            for page in pages
            for chunk in by_paragraph(page, max_size=1000)
        ],
    }
    strategies["C"] = strategies["B"]
    for name, chunks in strategies.items():
        if not chunks:
            raise ValueError(f"Strategy {name} produced no chunks")
    return strategies


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _database_info(database_url: str) -> dict[str, Any]:
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT current_database(),
                       current_setting('server_version'),
                       (SELECT extversion FROM pg_extension WHERE extname = 'vector')
                """
            )
            row = cursor.fetchone()
    if row is None or row[2] is None:
        raise RuntimeError("Could not verify PostgreSQL and pgvector versions")
    return {
        "database": row[0],
        "postgresql_version": row[1],
        "pgvector_version": row[2],
        "search_metric": "cosine distance (<=>)",
        "search_mode": "exact scan; no approximate vector index configured",
    }


def _length_stats(chunks: list[Chunk]) -> dict[str, int | float]:
    lengths = [len(chunk.text) for chunk in chunks]
    return {
        "chunk_count": len(chunks),
        "total_chunk_characters": sum(lengths),
        "min_chunk_characters": min(lengths),
        "median_chunk_characters": statistics.median(lengths),
        "max_chunk_characters": max(lengths),
    }


def _manual_review_rows(
    strategies: dict[str, list[Chunk]],
    traces: list[dict[str, Any]],
    pages: list[DocumentPage],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    coherence_rows = []
    sample_pages = sorted(pages, key=lambda page: (page.document_id, page.page))[:10]
    for strategy, chunks in strategies.items():
        by_page: dict[tuple[str, int], list[Chunk]] = {}
        for chunk in chunks:
            by_page.setdefault((chunk.document_id, chunk.page), []).append(chunk)
        for page in sample_pages:
            matching_chunks = by_page.get((page.document_id, page.page), [])
            if matching_chunks:
                coherence_rows.append(
                    {
                        "review_type": "chunk_coherence",
                        "strategy": strategy,
                        "document_id": page.document_id,
                        "source": page.source,
                        "page": page.page,
                        "source_position_character": 0,
                        "chunk_id": matching_chunks[0].chunk_id,
                        "chunk_text": matching_chunks[0].text,
                        "score_0_to_2": None,
                        "reason": "",
                    }
                )

    grounding_rows = []
    metadata_rows = []
    for trace in traces:
        if trace["split"] != "test":
            continue
        grounding_rows.append(
            {
                "review_type": "grounding_quality",
                "strategy": trace["strategy"],
                "question_id": trace["id"],
                "question": trace["question"],
                "expected_answer": trace["expected_answer"],
                "evidence_quote": trace["evidence_quote"],
                "top_hits": trace["top_hits"],
                "score_0_to_2": None,
                "reason": "",
            }
        )
        for rank, hit in enumerate(trace["top_hits"], start=1):
            metadata_rows.append(
                {
                    "review_type": "metadata_correctness",
                    "strategy": trace["strategy"],
                    "question_id": trace["id"],
                    "rank": rank,
                    "document_id": hit.get("document_id"),
                    "source": hit.get("source"),
                    "page": hit.get("page"),
                    "metadata": hit.get("metadata"),
                    "filter": trace["filter"],
                    "automatic_filter_match": hit.get("filter_match"),
                    "metadata_complete_and_correct": None,
                    "reason": "",
                }
            )
    return coherence_rows, grounding_rows, metadata_rows


def run() -> Path:
    """Validate inputs, run A/B/C, and write trace and review files."""
    run_started = time.perf_counter()
    load_dotenv(STARTER_DIR / ".env")
    pages = load_jsonl(DATA_PATH)
    _validate_manifest(MANIFEST_PATH, pages)
    questions = _read_questions(QUESTIONS_PATH)
    _validate_inputs(pages, questions)
    strategies = _build_strategies(pages)

    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError(f"DATABASE_URL is missing from {STARTER_DIR / '.env'}")
    model_name = os.getenv("EMBEDDING_MODEL", "").strip() or DEFAULT_EMBEDDING_MODEL
    dimensions = MODEL_DIMENSIONS.get(model_name)
    if dimensions is None:
        supported_models = ", ".join(sorted(MODEL_DIMENSIONS))
        raise ValueError(f"Unsupported EMBEDDING_MODEL; choose one of: {supported_models}")

    store = PgVectorStore(database_url, dimensions)
    embedder = MyEmbedder(model_name)
    store.create_schema()
    database_details = _database_info(database_url)

    unique_chunks: dict[str, Chunk] = {}
    for chunks in strategies.values():
        for chunk in chunks:
            unique_chunks.setdefault(chunk.chunk_id, chunk)
    unique_texts = [chunk.text for chunk in unique_chunks.values()]
    embedding_started = time.perf_counter()
    vectors = embedder.embed_documents(unique_texts)
    embedding_seconds = time.perf_counter() - embedding_started
    if len(vectors) != len(unique_texts):
        raise RuntimeError(
            f"Embedder returned {len(vectors)} vectors for {len(unique_texts)} chunks"
        )
    vector_by_id = {
        chunk_id: vector
        for chunk_id, vector in zip(unique_chunks, vectors)
    }

    ingest_stats = {}
    for strategy, chunks in strategies.items():
        started = time.perf_counter()
        store.upsert(
            strategy,
            chunks,
            [vector_by_id[chunk.chunk_id] for chunk in chunks],
        )
        ingest_stats[strategy] = {
            **_length_stats(chunks),
            "upsert_seconds": time.perf_counter() - started,
        }

    traces = []
    search_seconds = {strategy: 0.0 for strategy in strategies}
    query_embedding_started = time.perf_counter()
    query_vectors = {
        question["id"]: embedder.embed_query(question["question"])
        for question in questions
    }
    query_embedding_seconds = time.perf_counter() - query_embedding_started
    for strategy in strategies:
        for question in questions:
            filters = question.get("filter") if strategy == "C" else None
            started = time.perf_counter()
            hits = store.search(
                strategy,
                query_vectors[question["id"]],
                top_k=TOP_K,
                filters=filters,
            )
            search_seconds[strategy] += time.perf_counter() - started
            traces.append(
                {
                    "strategy": strategy,
                    "embedding_model": model_name,
                    **evaluate_query(question, hits),
                }
            )

    summary = summarize(traces)
    timestamp = datetime.now(timezone.utc)
    run_dir = RESULTS_DIR / timestamp.strftime("run-%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=False)

    coherence_rows, grounding_rows, metadata_rows = _manual_review_rows(
        strategies,
        traces,
        pages,
    )
    metadata = {
        "created_at_utc": timestamp.isoformat(),
        "runtime_seconds": time.perf_counter() - run_started,
        "python_version": platform.python_version(),
        "package_versions": {
            "psycopg": version("psycopg"),
            "pypdf": version("pypdf"),
            "python-dotenv": version("python-dotenv"),
        },
        "corpus_file": str(DATA_PATH.relative_to(STARTER_DIR)),
        "manifest_file": str(MANIFEST_PATH.relative_to(STARTER_DIR)),
        "corpus_pages": len({(page.document_id, page.page) for page in pages}),
        "corpus_characters": sum(len(page.text) for page in pages),
        "benchmark_file": str(QUESTIONS_PATH.relative_to(STARTER_DIR)),
        "question_count": len(questions),
        "filtered_question_count": sum(
            question.get("filter") is not None for question in questions
        ),
        "split_counts": {
            split: sum(question["split"] == split for question in questions)
            for split in ("dev", "test")
        },
        "metadata_schema": {
            "document_id": "string; stable source document identifier",
            "source": "string; source location carried from corpus record",
            "page": "integer; 1-based original PDF physical page",
            "section": {
                "type": "string",
                "values": sorted(
                    {
                        page.metadata["section"]
                        for page in pages
                        if isinstance(page.metadata.get("section"), str)
                    }
                ),
                "source": "section values recorded in the selected-page corpus metadata",
                "purpose": "filter retrieval to a section when a question supplies that scope",
            },
        },
        "strategies": {
            "A": "fixed-size, 500 characters with 50-character overlap, no filter",
            "B": "legal article/clause-aware grouping, maximum 1000 characters, no filter",
            "C": "same article/clause-aware chunks as B, metadata filter only when question supplies one",
        },
        "embedding": {
            "provider": "OpenRouter",
            "model": model_name,
            "dimensions": dimensions,
            "unique_document_chunks_embedded": len(unique_chunks),
            "embedding_seconds": embedding_seconds,
            "query_count": len(questions),
            "query_embedding_seconds": query_embedding_seconds,
            "cost_usd": getattr(embedder, "reported_cost_usd", None),
            "cost_note": (
                "Provider-reported usage."
                if getattr(embedder, "reported_cost_usd", None) is not None
                else "Provider response did not report cost; fill from provider usage/pricing."
            ),
            "input_tokens": getattr(embedder, "reported_input_tokens", None),
        },
        "vector_store": database_details,
        "ingest": ingest_stats,
        "search_seconds_by_strategy": search_seconds,
        "automatic_metrics": summary,
        "manual_assessment": {
            "scores_assigned": False,
            "chunk_coherence_review_rows": len(coherence_rows),
            "grounding_review_rows": len(grounding_rows),
            "metadata_review_rows": len(metadata_rows),
            "rubric": "Score 0-2 and add a reason for each row; do not treat automatic hits as manual labels.",
        },
        "reingestion": {
            "idempotency": "Primary key (strategy, chunk_id) with ON CONFLICT UPDATE.",
            "stale_chunks": "Old chunks for each supplied document_id are removed during upsert.",
        },
    }

    _write_jsonl(run_dir / "retrieval_traces.jsonl", traces)
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "run_metadata.json", metadata)
    _write_jsonl(run_dir / "manual_coherence_review.jsonl", coherence_rows)
    _write_jsonl(run_dir / "manual_grounding_review.jsonl", grounding_rows)
    _write_jsonl(run_dir / "manual_metadata_review.jsonl", metadata_rows)
    return run_dir


def main() -> None:
    run_dir = run()
    print(f"Experiment complete. Results: {run_dir}")
    print(f"Summary: {run_dir / 'summary.json'}")
    print(f"Traces: {run_dir / 'retrieval_traces.jsonl'}")
    print("Manual review templates were created; automatic metrics are not manual scores.")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
