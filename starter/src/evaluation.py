"""Evaluate retrieved evidence and summarize retrieval metrics."""
import unicodedata
from typing import Any


def _normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def _gold_options(question: dict[str, Any]) -> list[dict[str, Any]]:
    source_id = question.get("gold_source_id")
    page = question.get("gold_page")
    evidence = question.get("evidence_quote")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("question must contain a non-empty gold_source_id")
    if isinstance(page, bool) or not isinstance(page, int) or page < 1:
        raise ValueError("question must contain a positive integer gold_page")
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("question must contain a non-empty evidence_quote")

    options = [{"source_id": source_id, "page": page, "evidence_quote": evidence}]
    alternatives = question.get("gold_alternatives", [])
    if not isinstance(alternatives, list):
        raise ValueError("gold_alternatives must be a list when provided")
    for alternative in alternatives:
        if not isinstance(alternative, dict):
            raise ValueError("each gold alternative must be an object")
        alt_source = alternative.get("gold_source_id", source_id)
        alt_page = alternative.get("gold_page")
        alt_evidence = alternative.get("evidence_quote")
        if not isinstance(alt_source, str) or not alt_source:
            raise ValueError("each gold alternative needs a gold_source_id")
        if isinstance(alt_page, bool) or not isinstance(alt_page, int) or alt_page < 1:
            raise ValueError("each gold alternative needs a positive integer gold_page")
        if not isinstance(alt_evidence, str) or not alt_evidence.strip():
            raise ValueError("each gold alternative needs a non-empty evidence_quote")
        options.append(
            {
                "source_id": alt_source,
                "page": alt_page,
                "evidence_quote": alt_evidence,
            }
        )
    return options


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate exact gold evidence matches in the retrieved top three.

    Documented alternatives may be supplied in ``gold_alternatives`` as objects
    with ``gold_source_id``, ``gold_page``, and ``evidence_quote``. Exact matching
    is only an automatic retrieval proxy; it does not assign manual grounding or
    chunk-coherence labels.
    """
    if not isinstance(question, dict):
        raise ValueError("question must be an object")
    if not isinstance(hits, list):
        raise ValueError("hits must be a list")
    question_id = question.get("id")
    query = question.get("question")
    split = question.get("split")
    if not isinstance(question_id, str) or not question_id:
        raise ValueError("question must contain a non-empty id")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("question must contain non-empty question text")
    if split not in {"dev", "test"}:
        raise ValueError("question split must be 'dev' or 'test'")

    gold_options = _gold_options(question)
    filters = question.get("filter")
    if filters is not None and not isinstance(filters, dict):
        raise ValueError("question filter must be an object or null")

    hit_details = []
    for rank, hit in enumerate(hits[:3], start=1):
        if not isinstance(hit, dict):
            raise ValueError(f"hit at rank {rank} must be an object")
        text = hit.get("text")
        if not isinstance(text, str):
            raise ValueError(f"hit at rank {rank} must contain text")
        metadata = hit.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError(f"hit metadata at rank {rank} must be an object")

        source_id = hit.get("document_id", metadata.get("document_id"))
        page = hit.get("page", metadata.get("page"))
        if page is not None and (isinstance(page, bool) or not isinstance(page, int)):
            raise ValueError(f"hit page at rank {rank} must be an integer")

        matched_gold = next(
            (
                option
                for option in gold_options
                if source_id == option["source_id"]
                and page == option["page"]
                and _normalized_text(option["evidence_quote"])
                in _normalized_text(text)
            ),
            None,
        )

        filter_match = (
            all(metadata.get(key) == value for key, value in filters.items())
            if filters
            else None
        )
        hit_details.append(
            {
                **hit,
                "rank": rank,
                "gold_match": matched_gold is not None,
                "matched_gold": matched_gold,
                "filter_match": filter_match,
            }
        )

    hit_rank = next(
        (detail["rank"] for detail in hit_details if detail["gold_match"]),
        None,
    )
    return {
        "id": question_id,
        "split": split,
        "question": query,
        "expected_answer": question.get("expected_answer"),
        "gold_source_id": question["gold_source_id"],
        "gold_page": question["gold_page"],
        "evidence_quote": question["evidence_quote"],
        "filter": filters,
        "top_hits": hit_details,
        "hit_at_1": hit_rank == 1,
        "evidence_hit_at_3": hit_rank is not None,
        "hit_rank": hit_rank,
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Hit@1 and Evidence Hit@3 as counts and rates by strategy/split."""
    if not isinstance(traces, list):
        raise ValueError("traces must be a list")

    groups: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for trace in traces:
        if not isinstance(trace, dict):
            raise ValueError("each trace must be an object")
        strategy = trace.get("strategy")
        split = trace.get("split")
        if not isinstance(strategy, str) or not strategy:
            raise ValueError("each trace must contain a strategy name")
        if split not in {"dev", "test"}:
            raise ValueError("each trace split must be 'dev' or 'test'")
        if not isinstance(trace.get("hit_at_1"), bool):
            raise ValueError("each trace must contain boolean hit_at_1")
        if not isinstance(trace.get("evidence_hit_at_3"), bool):
            raise ValueError("each trace must contain boolean evidence_hit_at_3")
        groups.setdefault(strategy, {}).setdefault(split, []).append(trace)

    summary: dict[str, Any] = {}
    for strategy, splits in groups.items():
        summary[strategy] = {}
        for split, split_traces in splits.items():
            total = len(split_traces)
            hit_at_1 = sum(trace["hit_at_1"] for trace in split_traces)
            evidence_hit_at_3 = sum(
                trace["evidence_hit_at_3"] for trace in split_traces
            )
            summary[strategy][split] = {
                "total": total,
                "hit_at_1": {
                    "hits": hit_at_1,
                    "total": total,
                    "rate": hit_at_1 / total,
                },
                "evidence_hit_at_3": {
                    "hits": evidence_hit_at_3,
                    "total": total,
                    "rate": evidence_hit_at_3 / total,
                },
            }
    return summary
