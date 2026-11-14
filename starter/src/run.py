"""Điểm vào pipeline Day 06 — đọc PDF → chunk → embed → lưu pgvector → chấm benchmark.

Chạy từ thư mục starter/:
    python -m src.run
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import DEFAULT_MODEL, MyEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_pdf
from .models import Chunk, DocumentPage
from .store import PgVectorStore

# ---------------------------------------------------------------------------
# Đường dẫn: file này nằm ở starter/src/run.py
#   STARTER_DIR  = starter/
#   PROJECT_DIR  = repo (chứa data/, BENCHMARK.md, …)
# ---------------------------------------------------------------------------
STARTER_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = STARTER_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = STARTER_DIR / "results"

PDF_PATH = DATA_DIR / "fpt_esg_report_2025.pdf"
BENCHMARK_PATH = DATA_DIR / "benchmark.json"
DOCUMENT_ID = "fpt_esg_report_2025"
SOURCE_LABEL = "data/fpt_esg_report_2025.pdf"

# Ít nhất 30 trang nội dung — yêu cầu GUIDE
MIN_CONTENT_PAGES = 30

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("day06.run")


def _load_benchmark(path: Path) -> list[dict[str, Any]]:
    """Đọc bộ câu hỏi đã chốt (15 dev / 5 test) — không sửa khi chạy."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    questions = raw["questions"] if isinstance(raw, dict) else raw
    if len(questions) < 20:
        raise RuntimeError(
            f"Benchmark cần ≥20 câu, hiện có {len(questions)}: {path}"
        )
    return questions


def _chunk_all_pages(
    pages: list[DocumentPage],
) -> tuple[list[Chunk], list[Chunk]]:
    """Tạo hai bộ chunk trên CÙNG corpus:

    - A: cắt cố định 500 ký tự, overlap 50 (có thể cắt giữa câu)
    - B: theo đoạn/bullet, tối đa 800 ký tự (giữ ý hơn)
    Config C không tạo chunk mới — tái dùng B, chỉ khác lúc search có filter.
    """
    chunks_a: list[Chunk] = []
    chunks_b: list[Chunk] = []
    for page in pages:
        chunks_a.extend(fixed_size(page, size=500, overlap=50))
        chunks_b.extend(by_paragraph(page, max_size=800))
    return chunks_a, chunks_b


def _embed_and_store(
    store: PgVectorStore,
    embedder: MyEmbedder,
    strategy: str,
    chunks: list[Chunk],
) -> list[list[float]]:
    """Đưa text → vector → ghi PostgreSQL cho một strategy.

    Trả về vectors để strategy C có thể tái dùng vector của B (tiết kiệm thời gian).
    """
    logger.info(
        "Strategy %s: đang embed %s chunk bằng %s …",
        strategy,
        len(chunks),
        embedder.model_name,
    )
    t0 = time.perf_counter()
    vectors = embedder.embed_documents([c.text for c in chunks])
    embed_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    store.upsert(strategy, chunks, vectors)
    upsert_s = time.perf_counter() - t1

    # Kiểm tra nhanh: số dòng trong DB phải khớp số chunk vừa nạp
    n_db = store.count(strategy)
    if n_db != len(chunks):
        raise RuntimeError(
            f"Strategy {strategy}: DB có {n_db} dòng nhưng vừa upsert {len(chunks)} chunk"
        )

    logger.info(
        "Strategy %s xong | embed=%.1fs | upsert=%.1fs | db_rows=%s",
        strategy,
        embed_s,
        upsert_s,
        n_db,
    )
    return vectors


def _filters_for_question(
    strategy: str, question: dict[str, Any]
) -> dict[str, Any] | None:
    """Chỉ strategy C mới lọc metadata; và chỉ khi câu hỏi CÓ field filter.

    Ví dụ filter hợp lệ: {"pillar": "Environment"}
    Câu không có filter → C chạy giống B (không lọc) — đúng BENCHMARK.md.
    Không bao giờ nhét gold_source / đáp án vào filter.
    """
    if strategy != "C":
        return None
    filt = question.get("filter")
    if not filt:
        return None
    if not isinstance(filt, dict):
        raise ValueError(f"filter phải là object JSON, nhận được: {filt!r}")
    return dict(filt)


def _run_retrieval_eval(
    store: PgVectorStore,
    embedder: MyEmbedder,
    questions: list[dict[str, Any]],
    strategies: list[str] = ("A", "B", "C"),
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Chạy cùng một bộ câu hỏi cho mọi strategy, lưu full top-3 trace."""
    traces: list[dict[str, Any]] = []

    for strategy in strategies:
        logger.info("=== Đánh giá strategy %s (%s câu) ===", strategy, len(questions))
        for q in questions:
            q_text = q["question"]
            filters = _filters_for_question(strategy, q)

            # 1) Embed câu hỏi (cùng model với document)
            q_vec = embedder.embed_query(q_text)

            # 2) Lấy top-3 chunk gần nhất (có/không filter)
            hits = store.search(
                strategy=strategy,
                query_vector=q_vec,
                top_k=top_k,
                filters=filters,
            )

            # 3) Chấm Hit@1 / Evidence Hit@3 + giữ text để audit thủ công sau
            trace = evaluate_query(q, hits)
            trace["strategy"] = strategy
            trace["applied_filter"] = filters
            traces.append(trace)

            mark = "✓" if trace["evidence_hit_at_3"] else "✗"
            logger.info(
                "  %s %s [%s] hit@1=%s hit@3=%s filter=%s",
                mark,
                q.get("id"),
                strategy,
                trace["hit_at_1"],
                trace["evidence_hit_at_3"],
                filters,
            )

    return traces


def _save_results(
    traces: list[dict[str, Any]],
    summary: dict[str, Any],
    run_meta: dict[str, Any],
) -> None:
    """Ghi kết quả vào starter/results/ để nộp bài / xem lại."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    (RESULTS_DIR / "run_meta.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (RESULTS_DIR / "traces.json").write_text(
        json.dumps(traces, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Đã lưu kết quả vào %s", RESULTS_DIR)


def _print_summary_table(summary: dict[str, Any]) -> None:
    """In bảng Hit@1 / Evidence Hit@3 cho dễ nhìn trên terminal."""
    print("\n========== TÓM TẮT BENCHMARK ==========")
    print(f"Tổng trace: {summary.get('n_traces')}")
    print(f"Overall Hit@1: {summary['overall']['hit_at_1']}")
    print(f"Overall Evidence Hit@3: {summary['overall']['evidence_hit_at_3']}")
    print("-" * 52)
    print(f"{'strategy:split':<16} {'Hit@1':<10} {'Evidence Hit@3':<16}")
    for key, row in summary.get("by_strategy_split", {}).items():
        print(f"{key:<16} {row['hit_at_1']:<10} {row['evidence_hit_at_3']:<16}")
    print("=======================================\n")
    print("File chi tiết:")
    print(f"  - {RESULTS_DIR / 'summary.json'}")
    print(f"  - {RESULTS_DIR / 'traces.json'}")
    print(f"  - {RESULTS_DIR / 'run_meta.json'}")


def main() -> None:
    """Orchestration đủ 5 bước trong docstring starter gốc."""
    started = time.perf_counter()

    # ----- 0) Cấu hình môi trường -----
    # Đọc DATABASE_URL từ starter/.env (Docker pgvector local)
    load_dotenv(STARTER_DIR / ".env")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "Thiếu DATABASE_URL. Copy starter/.env.example → starter/.env rồi "
            "chạy: docker compose -p day6_student_starter up -d --wait"
        )

    model_name = os.getenv("EMBEDDING_MODEL") or DEFAULT_MODEL

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"Không thấy PDF corpus: {PDF_PATH}")
    if not BENCHMARK_PATH.exists():
        raise FileNotFoundError(f"Không thấy benchmark: {BENCHMARK_PATH}")

    # ----- 1) Đọc corpus + benchmark -----
    logger.info("Đang đọc PDF: %s", PDF_PATH)
    pages = load_pdf(PDF_PATH, document_id=DOCUMENT_ID, source=SOURCE_LABEL)
    if len(pages) < MIN_CONTENT_PAGES:
        raise RuntimeError(
            f"Cần ≥{MIN_CONTENT_PAGES} trang nội dung, mới có {len(pages)}"
        )
    logger.info("Đã load %s trang nội dung (đạt ngưỡng ≥%s)", len(pages), MIN_CONTENT_PAGES)

    questions = _load_benchmark(BENCHMARK_PATH)
    n_dev = sum(1 for q in questions if q.get("split") == "dev")
    n_test = sum(1 for q in questions if q.get("split") == "test")
    logger.info("Benchmark: %s câu (dev=%s, test=%s)", len(questions), n_dev, n_test)

    # ----- 2) Chunk A và B -----
    chunks_a, chunks_b = _chunk_all_pages(pages)
    logger.info("Chunk A (fixed)=%s | Chunk B (paragraph)=%s", len(chunks_a), len(chunks_b))

    # ----- 3) Embed + lưu vector (cùng 1 model cho cả A/B/C) -----
    embedder = MyEmbedder(model_name=model_name, local_files_only=True)
    store = PgVectorStore(database_url, dimensions=embedder.dimensions)
    store.create_schema()

    _embed_and_store(store, embedder, "A", chunks_a)
    vectors_b = _embed_and_store(store, embedder, "B", chunks_b)

    # C dùng ĐÚNG chunk + vector của B; khác biệt chỉ ở bước search (filter)
    logger.info("Strategy C: tái dùng %s chunk/vector của B (không embed lại)", len(chunks_b))
    store.upsert("C", chunks_b, vectors_b)
    if store.count("C") != len(chunks_b):
        raise RuntimeError("Upsert strategy C thất bại — số dòng không khớp")

    # ----- 4) Chạy cùng câu hỏi, lưu top-3 trace -----
    traces = _run_retrieval_eval(store, embedder, questions, strategies=["A", "B", "C"])
    summary = summarize(traces)

    # ----- 5) Lưu kết quả + in bảng -----
    elapsed = time.perf_counter() - started
    run_meta = {
        "document_id": DOCUMENT_ID,
        "pdf_path": str(PDF_PATH),
        "benchmark_path": str(BENCHMARK_PATH),
        "n_pages": len(pages),
        "n_questions": len(questions),
        "n_dev": n_dev,
        "n_test": n_test,
        "n_chunks_A": len(chunks_a),
        "n_chunks_B": len(chunks_b),
        "n_chunks_C": len(chunks_b),
        "embedding_model": embedder.model_name,
        "embedding_dimensions": embedder.dimensions,
        "vector_store": "PostgreSQL + pgvector (cosine / HNSW)",
        "strategies": {
            "A": "fixed_size(size=500, overlap=50), không filter",
            "B": "by_paragraph(max_size=800), không filter",
            "C": "giống B + metadata filter (pillar) khi câu hỏi có filter",
        },
        "elapsed_seconds": round(elapsed, 2),
        "note": (
            "Grounding/coherence thủ công chưa tự chấm ở đây — "
            "xem traces.json để audit theo BENCHMARK.md."
        ),
    }
    _save_results(traces, summary, run_meta)
    _print_summary_table(summary)
    logger.info("Pipeline xong trong %.1f giây", elapsed)


if __name__ == "__main__":
    main()
