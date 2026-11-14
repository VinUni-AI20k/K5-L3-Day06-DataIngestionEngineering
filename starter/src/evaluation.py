"""Đánh giá truy xuất: Hit@1 + Evidence Hit@3, giữ full trace theo câu hỏi."""

from __future__ import annotations

import re
from typing import Any


def _norm_ws(text: str) -> str:
    """Chuẩn hóa khoảng trắng để so evidence (khớp cách kiểm chứng benchmark)."""
    return re.sub(r"\s+", " ", text or "").strip()


def _as_list(value: Any) -> list[Any]:
    """Cho phép gold_page / evidence_quote là 1 giá trị hoặc list (nhiều nguồn hợp lệ)."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _source_matches(hit: dict[str, Any], gold_source_id: str) -> bool:
    """Khớp nguồn: document_id hoặc đường dẫn source chứa gold_source_id."""
    if not gold_source_id:
        return True
    doc_id = str(hit.get("document_id") or "")
    source = str(hit.get("source") or "")
    return (
        doc_id == gold_source_id
        or source == gold_source_id
        or gold_source_id in source
    )


def _evidence_in_text(chunk_text: str, evidence_quote: str) -> bool:
    """Hit evidence khi quote gold (sau chuẩn hóa WS) nằm trọn trong text chunk."""
    if not evidence_quote:
        return False
    return _norm_ws(evidence_quote) in _norm_ws(chunk_text)


def is_retrieval_hit(question: dict[str, Any], hit: dict[str, Any]) -> bool:
    """Một hit hợp lệ theo GUIDE/BENCHMARK:

    chunk thuộc nguồn/trang gold VÀ chứa đủ evidence_quote đã ghi.
    Không suy ra nhãn grounding/coherence thủ công từ exact match.
    """
    gold_pages = {int(p) for p in _as_list(question.get("gold_page"))}
    gold_sources = [str(s) for s in _as_list(question.get("gold_source_id"))]
    evidences = [str(e) for e in _as_list(question.get("evidence_quote"))]

    try:
        hit_page = int(hit.get("page"))
    except (TypeError, ValueError):
        return False

    if gold_pages and hit_page not in gold_pages:
        return False

    if gold_sources and not any(_source_matches(hit, s) for s in gold_sources):
        return False

    # Đủ evidence: chấp nhận bất kỳ quote hợp lệ nào (nếu có nhiều)
    if evidences:
        text = str(hit.get("text") or "")
        if not any(_evidence_in_text(text, ev) for ev in evidences):
            return False

    return True


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Chấm 1 câu hỏi + top-k hits → trace đầy đủ cho báo cáo.

    hit_at_1: top-1 có phải retrieval hit?
    evidence_hit_at_3: có ≥1 hit trong top-3?
    """
    ranked = list(hits or [])
    hit_flags = [is_retrieval_hit(question, h) for h in ranked]

    hit_at_1 = bool(hit_flags[0]) if hit_flags else False
    evidence_hit_at_3 = any(hit_flags[:3])

    # Vị trí hit đầu tiên (1-based); None nếu không có trong top-k
    first_hit_rank = next(
        (i + 1 for i, ok in enumerate(hit_flags) if ok),
        None,
    )

    return {
        "question_id": question.get("id"),
        "split": question.get("split"),
        "type": question.get("type"),
        "question": question.get("question"),
        "gold_page": question.get("gold_page"),
        "gold_source_id": question.get("gold_source_id"),
        "evidence_quote": question.get("evidence_quote"),
        "filter": question.get("filter"),
        "expected_answer": question.get("expected_answer"),
        # Kết quả truy xuất
        "hit_at_1": hit_at_1,
        "evidence_hit_at_3": evidence_hit_at_3,
        "first_hit_rank": first_hit_rank,
        "hits": [
            {
                "rank": i + 1,
                "chunk_id": h.get("chunk_id"),
                "page": h.get("page"),
                "document_id": h.get("document_id"),
                "source": h.get("source"),
                "pillar": h.get("pillar"),
                "score": h.get("score"),
                "is_hit": hit_flags[i],
                "text": h.get("text"),
                "metadata": h.get("metadata"),
            }
            for i, h in enumerate(ranked)
        ],
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Gộp Hit@1 và Evidence Hit@3 theo strategy + split (dev/test).

    Mỗi trace nên có thêm khóa `strategy` (do pipeline gắn khi chạy).
    Báo số tử/mẫu, không chỉ %.
    """
    # Nhóm: (strategy, split) → counters
    groups: dict[tuple[str, str], dict[str, int]] = {}

    for tr in traces:
        strategy = str(tr.get("strategy") or "unknown")
        split = str(tr.get("split") or "unknown")
        key = (strategy, split)
        bucket = groups.setdefault(
            key,
            {"n": 0, "hit_at_1": 0, "evidence_hit_at_3": 0},
        )
        bucket["n"] += 1
        if tr.get("hit_at_1"):
            bucket["hit_at_1"] += 1
        if tr.get("evidence_hit_at_3"):
            bucket["evidence_hit_at_3"] += 1

    by_strategy_split: dict[str, Any] = {}
    for (strategy, split), b in sorted(groups.items()):
        n = b["n"]
        by_strategy_split[f"{strategy}:{split}"] = {
            "strategy": strategy,
            "split": split,
            "n": n,
            # Tử/mẫu đúng yêu cầu GUIDE
            "hit_at_1": f"{b['hit_at_1']}/{n}",
            "evidence_hit_at_3": f"{b['evidence_hit_at_3']}/{n}",
            "hit_at_1_rate": (b["hit_at_1"] / n) if n else 0.0,
            "evidence_hit_at_3_rate": (b["evidence_hit_at_3"] / n) if n else 0.0,
        }

    # Tổng quan toàn bộ (mọi strategy/split)
    n_all = len(traces)
    h1 = sum(1 for t in traces if t.get("hit_at_1"))
    h3 = sum(1 for t in traces if t.get("evidence_hit_at_3"))

    return {
        "n_traces": n_all,
        "overall": {
            "hit_at_1": f"{h1}/{n_all}",
            "evidence_hit_at_3": f"{h3}/{n_all}",
        },
        "by_strategy_split": by_strategy_split,
    }
