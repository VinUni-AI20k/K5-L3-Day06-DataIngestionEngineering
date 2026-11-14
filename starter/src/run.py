"""Main pipeline entry point for Day 06 Data Ingestion Engineering."""
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import MyEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_pdf
from .models import Chunk, DocumentPage
from .store import PgVectorStore, QdrantPersistentStore

STARTER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = STARTER_DIR.parent


def _compute_chunk_stats(chunks: list[Chunk]) -> dict[str, Any]:
    lengths = [len(c.text) for c in chunks]
    approx_tokens = [max(1, round(len(c.text) / 4)) for c in chunks]
    if not lengths:
        return {"count": 0}
    sorted_lens = sorted(lengths)
    n = len(sorted_lens)
    return {
        "chunk_count": n,
        "total_chars": sum(lengths),
        "total_approx_tokens": sum(approx_tokens),
        "min_chars": sorted_lens[0],
        "p25_chars": sorted_lens[max(0, int(0.25 * (n - 1)))],
        "median_chars": sorted_lens[n // 2],
        "mean_chars": round(sum(lengths) / n, 1),
        "p75_chars": sorted_lens[min(n - 1, int(0.75 * (n - 1)))],
        "max_chars": sorted_lens[-1],
    }


def _select_vector_store(dimensions: int, results_dir: Path):
    """Use PgVectorStore if PostgreSQL is reachable; otherwise use persistent Qdrant store if configured."""
    load_dotenv(STARTER_DIR / ".env")
    backend_pref = os.getenv("VECTOR_STORE_BACKEND", "auto").strip().lower()
    db_url = os.getenv("DATABASE_URL", "").strip()

    if backend_pref in {"pgvector", "auto"} and db_url:
        try:
            pg_store = PgVectorStore(database_url=db_url, dimensions=dimensions)
            pg_store.create_schema()
            return pg_store, {
                "backend": "PostgreSQL + pgvector",
                "connection": db_url.split("@")[-1],
                "distance_metric": "cosine (<=>)",
                "search_mode": "HNSW cosine index (vector_cosine_ops)",
            }
        except Exception as pg_exc:
            if backend_pref == "pgvector":
                raise RuntimeError(
                    f"PostgreSQL + pgvector requested but failed to connect: {pg_exc}"
                ) from pg_exc
            print(
                f"[WARN] PostgreSQL container not reachable ({pg_exc}). "
                f"Switching to persistent local Qdrant store per GUIDE.md / CHECKPOINTS.md."
            )

    qdrant_path = results_dir / "qdrant_db"
    q_store = QdrantPersistentStore(storage_path=qdrant_path, dimensions=dimensions)
    q_store.create_schema()
    return q_store, {
        "backend": "Qdrant Local Persistent (qdrant-client 1.20.0)",
        "storage_path": str(qdrant_path.relative_to(REPO_ROOT)),
        "collection_name": QdrantPersistentStore.COLLECTION_NAME,
        "distance_metric": "Cosine (Distance.COSINE)",
        "search_mode": "Exact Cosine similarity + structured payload Filter",
    }


def _embed_with_disk_cache(
    embedder: MyEmbedder,
    texts: list[str],
    cache_path: Path,
) -> tuple[list[list[float]], float, bool]:
    """Encode texts with real embedder; store/reuse verified cache keyed by model+sha256 for fast re-runs."""
    cache_data: dict[str, list[float]] = {}
    if cache_path.exists():
        try:
            cache_data = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache_data = {}

    keys = [
        hashlib.sha256(f"{embedder.model_name}::{t}".encode("utf-8")).hexdigest()
        for t in texts
    ]
    missing_indices = [i for i, k in enumerate(keys) if k not in cache_data]

    t0 = time.perf_counter()
    if missing_indices:
        missing_texts = [texts[i] for i in missing_indices]
        new_vecs = embedder.embed_documents(missing_texts)
        for idx, vec in zip(missing_indices, new_vecs):
            cache_data[keys[idx]] = vec
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache_data), encoding="utf-8")
        used_cache = False
    else:
        used_cache = True
    elapsed = time.perf_counter() - t0

    vectors = [cache_data[k] for k in keys]
    return vectors, elapsed, used_cache


def _build_coherence_samples(
    pages: list[DocumentPage],
    chunks_A: list[Chunk],
    chunks_B: list[Chunk],
) -> list[dict[str, Any]]:
    """Select the first 10 sorted page IDs (fixed before viewing scores) for manual Chunk Coherence audit."""
    sorted_pages = sorted(pages, key=lambda p: str(p.metadata.get("page_id", "")))[:10]
    samples: list[dict[str, Any]] = []

    for pos_idx, page in enumerate(sorted_pages, start=1):
        pid = page.metadata.get("page_id")
        # Anchor position: around character offset 350 of the normalized page text (first body transition)
        anchor_offset = min(350, max(0, len(page.text) // 3))
        anchor_snippet = page.text[anchor_offset : anchor_offset + 140].replace("\n", " ").strip()

        page_chunks_a = [c for c in chunks_A if c.document_id == page.document_id and c.page == page.page]
        page_chunks_b = [c for c in chunks_B if c.document_id == page.document_id and c.page == page.page]

        # Pick chunk covering the anchor snippet (or chunk index 1 / 0)
        chosen_a = page_chunks_a[min(1, len(page_chunks_a) - 1)] if page_chunks_a else None
        chosen_b = page_chunks_b[min(1, len(page_chunks_b) - 1)] if page_chunks_b else None
        for ca in page_chunks_a:
            if anchor_snippet[:50] in ca.text:
                chosen_a = ca
                break
        for cb in page_chunks_b:
            if anchor_snippet[:50] in cb.text:
                chosen_b = cb
                break

        samples.append(
            {
                "position_id": f"pos_{pos_idx:02d}",
                "page_id": pid,
                "document_id": page.document_id,
                "page": page.page,
                "section": page.metadata.get("section"),
                "anchor_snippet": anchor_snippet,
                "strategy_A_chunk": {
                    "chunk_id": chosen_a.chunk_id if chosen_a else None,
                    "char_length": len(chosen_a.text) if chosen_a else 0,
                    "text": chosen_a.text if chosen_a else "",
                },
                "strategy_BC_chunk": {
                    "chunk_id": chosen_b.chunk_id if chosen_b else None,
                    "char_length": len(chosen_b.text) if chosen_b else 0,
                    "text": chosen_b.text if chosen_b else "",
                },
            }
        )
    return samples


def main() -> None:
    print("=" * 76)
    print("DAY 06 DATA INGESTION ENGINEERING PIPELINE & BENCHMARK EVALUATION")
    print("=" * 76)

    manifest_path = REPO_ROOT / "data" / "manifest.json"
    benchmark_path = REPO_ROOT / "benchmark" / "questions.json"
    results_dir = REPO_ROOT / "results"
    starter_results_dir = STARTER_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    starter_results_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest file: {manifest_path}")
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Missing benchmark file: {benchmark_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    questions: list[dict[str, Any]] = benchmark["questions"]

    # 1. Load & normalize corpus pages
    t_ingest0 = time.perf_counter()
    all_pages: list[DocumentPage] = []
    error_log: list[dict[str, Any]] = []

    for doc in manifest["documents"]:
        pdf_path = REPO_ROOT / doc["local_path"]
        doc_pages = load_pdf(
            path=pdf_path,
            document_id=doc["source_id"],
            source=doc["local_path"],
            valid_pages=set(doc["valid_page_numbers"]),
            base_metadata={
                "year": int(doc["year"]),
                "category": str(doc["category"]),
                "title": str(doc["title"]),
            },
            error_log=error_log,
        )
        all_pages.extend(doc_pages)
    t_ingest = time.perf_counter() - t_ingest0

    if len(all_pages) < 30:
        raise RuntimeError(
            f"Corpus validation failed: minimum 30 valid content pages required, got {len(all_pages)}"
        )
    print(f"[1. Ingestion] Loaded {len(all_pages)} valid pages across {len(manifest['documents'])} PDFs in {t_ingest:.3f}s")

    # Write error & exclusion log
    err_json_path = results_dir / "ingestion_errors.json"
    err_log_path = results_dir / "ingestion_errors.log"
    err_json_path.write_text(json.dumps(error_log, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(err_log_path, "w", encoding="utf-8") as f:
        for ev in error_log:
            f.write(
                f"[{ev['event']}] doc={ev['document_id']} page={ev['page']:02d} "
                f"source={ev['source']} | {ev['detail']}\n"
            )

    # 2. Build chunks for Strategy A (fixed-size 500/50) and Strategy B/C (by_paragraph max_size=800)
    t_chunk_a0 = time.perf_counter()
    chunks_A = [c for p in all_pages for c in fixed_size(p, size=500, overlap=50)]
    t_chunk_a = time.perf_counter() - t_chunk_a0

    t_chunk_b0 = time.perf_counter()
    chunks_B = [c for p in all_pages for c in by_paragraph(p, max_size=800)]
    t_chunk_b = time.perf_counter() - t_chunk_b0

    stats_A = _compute_chunk_stats(chunks_A)
    stats_B = _compute_chunk_stats(chunks_B)
    print(
        f"[2. Chunking]  Strategy A (fixed 500/50): {stats_A['chunk_count']} chunks (mean={stats_A['mean_chars']} chars, {t_chunk_a:.4f}s)"
    )
    print(
        f"               Strategy B/C (paragraph 800): {stats_B['chunk_count']} chunks (mean={stats_B['mean_chars']} chars, {t_chunk_b:.4f}s)"
    )

    # 3. Initialize real semantic embedder & compute embeddings
    model_name = os.getenv("EMBEDDING_MODEL", "google/embeddinggemma-2").strip() or "google/embeddinggemma-2"
    t_load0 = time.perf_counter()
    embedder = MyEmbedder(model_name=model_name, batch_size=8, local_files_only=True)
    t_model_load = time.perf_counter() - t_load0
    print(f"[3. Embedder]  Loaded '{embedder.model_name}' (dim={embedder.dimensions}) in {t_model_load:.2f}s")

    cache_file = results_dir / "embedding_cache.json"
    print("               Encoding Strategy A chunks...")
    vectors_A, t_embed_a, cached_a = _embed_with_disk_cache(
        embedder, [c.text for c in chunks_A], cache_file
    )
    print(f"               Strategy A encoded ({len(vectors_A)} vectors) in {t_embed_a:.2f}s (cached={cached_a})")

    print("               Encoding Strategy B/C chunks...")
    vectors_B, t_embed_b, cached_b = _embed_with_disk_cache(
        embedder, [c.text for c in chunks_B], cache_file
    )
    print(f"               Strategy B/C encoded ({len(vectors_B)} vectors) in {t_embed_b:.2f}s (cached={cached_b})")

    print("               Encoding 20 benchmark queries...")
    query_vectors, t_embed_q, _ = _embed_with_disk_cache(
        embedder, [q["question"] for q in questions], cache_file
    )

    # 4. Initialize Vector Store, upsert all 3 strategies, and verify idempotency
    store, store_info = _select_vector_store(dimensions=embedder.dimensions, results_dir=results_dir)
    print(f"[4. Store]     Backend: {store_info['backend']}")

    t_up_a0 = time.perf_counter()
    store.upsert("A", chunks_A, vectors_A)
    t_upsert_a = time.perf_counter() - t_up_a0

    t_up_b0 = time.perf_counter()
    store.upsert("B", chunks_B, vectors_B)
    t_upsert_b = time.perf_counter() - t_up_b0

    t_up_c0 = time.perf_counter()
    store.upsert("C", chunks_B, vectors_B)
    t_upsert_c = time.perf_counter() - t_up_c0

    # Verify idempotency: re-upsert Strategy B and confirm count does not double
    count_b_before = store.count_chunks("B")
    store.upsert("B", chunks_B, vectors_B)
    count_b_after = store.count_chunks("B")
    idempotent_ok = (count_b_before == len(chunks_B)) and (count_b_after == len(chunks_B))
    print(
        f"               Idempotency check on Strategy B: before={count_b_before}, "
        f"after_reingest={count_b_after}, passed={idempotent_ok}"
    )

    # 5. Run retrieval & evaluation for Strategies A, B, C
    all_traces: list[dict[str, Any]] = []
    traces_by_strategy: dict[str, list[dict[str, Any]]] = {"A": [], "B": [], "C": []}
    search_timings: dict[str, float] = {}

    for strat in ["A", "B", "C"]:
        t_s0 = time.perf_counter()
        for q, q_vec in zip(questions, query_vectors):
            active_filter = q.get("filter") if strat == "C" else None
            hits = store.search(
                strategy=strat,
                query_vector=q_vec,
                top_k=3,
                filters=active_filter,
            )
            trace = evaluate_query(question=q, hits=hits)
            trace["strategy"] = strat
            trace["applied_filter"] = active_filter
            traces_by_strategy[strat].append(trace)
            all_traces.append(trace)
        search_timings[strat] = round(time.perf_counter() - t_s0, 4)

    if hasattr(store, "close"):
        store.close()

    # 6. Summarize metrics and save artifacts
    metrics_summary = summarize(all_traces)
    for strat in ["A", "B", "C"]:
        (results_dir / f"traces_{strat}.json").write_text(
            json.dumps(traces_by_strategy[strat], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    (results_dir / "all_traces.json").write_text(
        json.dumps(all_traces, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (results_dir / "metrics_summary.json").write_text(
        json.dumps(metrics_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    coherence_samples = _build_coherence_samples(all_pages, chunks_A, chunks_B)
    (results_dir / "coherence_samples_10_positions.json").write_text(
        json.dumps(coherence_samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    ingestion_stats = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus_valid_pages": len(all_pages),
        "embedding_model": {
            "provider": "SentenceTransformers (Local HuggingFace)",
            "model_name": embedder.model_name,
            "dimensions": embedder.dimensions,
            "normalize_embeddings": True,
            "model_load_sec": round(t_model_load, 4),
            "api_cost_usd": 0.0,
        },
        "vector_store": store_info,
        "idempotency_verification": {
            "strategy_tested": "B",
            "expected_count": len(chunks_B),
            "count_before_reingest": count_b_before,
            "count_after_reingest": count_b_after,
            "idempotent_passed": idempotent_ok,
        },
        "strategies": {
            "A": {
                "description": "Fixed-size chunking (size=500 chars, overlap=50 chars), no metadata filtering",
                "chunking_method": "fixed_size",
                "parameters": {"size": 500, "overlap": 50, "metadata_filter_enabled": False},
                "chunk_stats": stats_A,
                "timings_sec": {
                    "pdf_ingest_and_normalize": round(t_ingest, 4),
                    "chunking": round(t_chunk_a, 4),
                    "embedding": round(t_embed_a, 4),
                    "upsert": round(t_upsert_a, 4),
                    "search_20_queries": search_timings["A"],
                },
                "cost_usd": 0.0,
            },
            "B": {
                "description": "Paragraph/Heading-aware chunking (max_size=800 chars), no metadata filtering",
                "chunking_method": "by_paragraph",
                "parameters": {"max_size": 800, "metadata_filter_enabled": False},
                "chunk_stats": stats_B,
                "timings_sec": {
                    "pdf_ingest_and_normalize": round(t_ingest, 4),
                    "chunking": round(t_chunk_b, 4),
                    "embedding": round(t_embed_b, 4),
                    "upsert": round(t_upsert_b, 4),
                    "search_20_queries": search_timings["B"],
                },
                "cost_usd": 0.0,
            },
            "C": {
                "description": "Paragraph/Heading-aware chunking (max_size=800 chars) + Metadata Filtering on filter-eligible queries",
                "chunking_method": "by_paragraph",
                "parameters": {
                    "max_size": 800,
                    "metadata_filter_enabled": True,
                    "filter_fields": ["category", "year"],
                },
                "chunk_stats": stats_B,
                "timings_sec": {
                    "pdf_ingest_and_normalize": round(t_ingest, 4),
                    "chunking": round(t_chunk_b, 4),
                    "embedding": round(t_embed_b, 4),
                    "upsert": round(t_upsert_c, 4),
                    "search_20_queries": search_timings["C"],
                },
                "cost_usd": 0.0,
            },
        },
    }
    (results_dir / "ingestion_stats.json").write_text(
        json.dumps(ingestion_stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Mirror JSON files into starter/results/ for compatibility
    for fname in [
        "traces_A.json",
        "traces_B.json",
        "traces_C.json",
        "all_traces.json",
        "metrics_summary.json",
        "ingestion_stats.json",
        "ingestion_errors.json",
        "ingestion_errors.log",
        "coherence_samples_10_positions.json",
    ]:
        shutil.copy2(results_dir / fname, starter_results_dir / fname)

    print("\n[5. Evaluation Summary]")
    for strat in ["A", "B", "C"]:
        dev_m = metrics_summary[strat]["dev"]
        test_m = metrics_summary[strat]["test"]
        all_m = metrics_summary[strat]["all"]
        print(
            f"  Strategy {strat} | "
            f"DEV: Hit@1={dev_m['hit_at_1_fraction']} ({dev_m['hit_at_1_pct']}%), "
            f"Hit@3={dev_m['evidence_hit_at_3_fraction']} ({dev_m['evidence_hit_at_3_pct']}%) | "
            f"TEST: Hit@1={test_m['hit_at_1_fraction']} ({test_m['hit_at_1_pct']}%), "
            f"Hit@3={test_m['evidence_hit_at_3_fraction']} ({test_m['evidence_hit_at_3_pct']}%) | "
            f"ALL: Hit@1={all_m['hit_at_1_fraction']}, Hit@3={all_m['evidence_hit_at_3_fraction']}"
        )


if __name__ == "__main__":
    main()
