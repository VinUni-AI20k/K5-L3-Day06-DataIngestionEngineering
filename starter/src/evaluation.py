"""Evaluate retrieved evidence and preserve per-query traces without conflating hit metrics with manual audits."""
import re
from typing import Any


def _flat_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Check gold source/page and complete evidence per chunk; record Hit@1, Evidence Hit@3, and split-evidence diagnostics.

    Accepts documented alternative gold pages/evidence. Does not infer human Grounding Quality
    or Chunk Coherence labels from automated string matches.
    """
    gold_source_id = question["gold_source_id"]
    gold_page = int(question["gold_page"])
    alt_pages = [int(p) for p in question.get("alternative_gold_pages", [])]
    valid_gold_pages = {gold_page, *alt_pages}

    flat_quote = _flat_ws(question["evidence_quote"])
    anchors: list[str] = [_flat_ws(a) for a in question.get("evidence_anchors", []) if a.strip()]
    q_filter: dict[str, Any] | None = question.get("filter")

    evaluated_hits: list[dict[str, Any]] = []
    first_hit_rank: int | None = None
    partial_matches_on_gold_page: int = 0
    anchors_covered_across_top3_gold: set[str] = set()

    for rank_idx, h in enumerate(hits[:3], start=1):
        chunk_doc_id = h.get("document_id")
        chunk_page = int(h.get("page", -1))
        chunk_text_flat = _flat_ws(h.get("text", ""))
        chunk_meta = h.get("metadata", {}) or {}

        is_gold_source_and_page = (chunk_doc_id == gold_source_id) and (chunk_page in valid_gold_pages)
        exact_quote_in_chunk = bool(flat_quote and flat_quote in chunk_text_flat)

        matched_anchors = [a for a in anchors if a in chunk_text_flat]
        all_anchors_in_chunk = bool(anchors) and (len(matched_anchors) == len(anchors))

        # A chunk is a valid single-chunk evidence hit if it comes from the gold source/page
        # AND contains either the full verbatim evidence quote or all required evidence anchors.
        has_complete_evidence = exact_quote_in_chunk or all_anchors_in_chunk
        is_valid_hit = is_gold_source_and_page and has_complete_evidence

        is_partial_split = (
            is_gold_source_and_page
            and not has_complete_evidence
            and (len(matched_anchors) > 0)
        )
        if is_partial_split:
            partial_matches_on_gold_page += 1
        if is_gold_source_and_page:
            anchors_covered_across_top3_gold.update(matched_anchors)

        if is_valid_hit and first_hit_rank is None:
            first_hit_rank = rank_idx

        filter_matches = None
        if q_filter:
            filter_matches = all(chunk_meta.get(k) == v for k, v in q_filter.items())

        evaluated_hits.append(
            {
                "rank": rank_idx,
                "chunk_id": h.get("chunk_id"),
                "document_id": chunk_doc_id,
                "source": h.get("source"),
                "page": chunk_page,
                "score": h.get("score"),
                "metadata": chunk_meta,
                "text": h.get("text", ""),
                "is_gold_source_and_page": is_gold_source_and_page,
                "exact_quote_in_chunk": exact_quote_in_chunk,
                "matched_anchors": matched_anchors,
                "total_anchors": len(anchors),
                "is_valid_hit": is_valid_hit,
                "is_partial_evidence_split": is_partial_split,
                "filter_compliant": filter_matches,
            }
        )

    hit_at_1 = first_hit_rank == 1
    evidence_hit_at_3 = first_hit_rank is not None
    multi_chunk_combined_hit = (
        not evidence_hit_at_3
        and bool(anchors)
        and len(anchors_covered_across_top3_gold) == len(anchors)
    )

    all_top3_filter_compliant = None
    if q_filter:
        all_top3_filter_compliant = bool(evaluated_hits) and all(
            eh["filter_compliant"] is True for eh in evaluated_hits
        )

    return {
        "question_id": question["id"],
        "split": question["split"],
        "type": question.get("type", "unknown"),
        "question": question["question"],
        "expected_answer": question["expected_answer"],
        "gold_source_id": gold_source_id,
        "gold_page": gold_page,
        "alternative_gold_pages": alt_pages,
        "evidence_quote": question["evidence_quote"],
        "evidence_anchors": anchors,
        "filter": q_filter,
        "hit_at_1": hit_at_1,
        "evidence_hit_at_3": evidence_hit_at_3,
        "first_hit_rank": first_hit_rank,
        "evidence_split_across_chunks_error": (not evidence_hit_at_3 and partial_matches_on_gold_page > 0),
        "multi_chunk_combined_hit": multi_chunk_combined_hit,
        "all_top3_filter_compliant": all_top3_filter_compliant,
        "top_3_hits": evaluated_hits,
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Hit@1 and Evidence Hit@3 with exact numerators/denominators (n/N) by strategy and split."""
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = {}
    strategies = sorted({str(t.get("strategy", "default")) for t in traces})

    for t in traces:
        strat = str(t.get("strategy", "default"))
        split = str(t.get("split", "all"))
        by_group.setdefault((strat, split), []).append(t)
        by_group.setdefault((strat, "all"), []).append(t)

    summary: dict[str, Any] = {}
    for strat in strategies:
        summary[strat] = {}
        for split in ["dev", "test", "all"]:
            items = by_group.get((strat, split), [])
            n_total = len(items)
            if n_total == 0:
                continue
            n_hit1 = sum(1 for x in items if x["hit_at_1"])
            n_hit3 = sum(1 for x in items if x["evidence_hit_at_3"])
            n_split_err = sum(1 for x in items if x.get("evidence_split_across_chunks_error"))
            n_multi_hit = sum(1 for x in items if x.get("multi_chunk_combined_hit"))

            filter_items = [x for x in items if x.get("filter") is not None]
            n_filter_total = len(filter_items)
            n_filter_ok = sum(1 for x in filter_items if x.get("all_top3_filter_compliant") is True)

            summary[strat][split] = {
                "total_questions": n_total,
                "hit_at_1_count": n_hit1,
                "hit_at_1_fraction": f"{n_hit1}/{n_total}",
                "hit_at_1_pct": round(100.0 * n_hit1 / n_total, 2),
                "evidence_hit_at_3_count": n_hit3,
                "evidence_hit_at_3_fraction": f"{n_hit3}/{n_total}",
                "evidence_hit_at_3_pct": round(100.0 * n_hit3 / n_total, 2),
                "evidence_split_error_count": n_split_err,
                "multi_chunk_combined_extra_hits": n_multi_hit,
                "filter_questions_count": n_filter_total,
                "filter_compliant_count": n_filter_ok if n_filter_total > 0 else None,
                "filter_compliance_fraction": (
                    f"{n_filter_ok}/{n_filter_total}" if n_filter_total > 0 else "N/A"
                ),
                "filter_compliance_pct": (
                    round(100.0 * n_filter_ok / n_filter_total, 2) if n_filter_total > 0 else None
                ),
            }
    return summary
