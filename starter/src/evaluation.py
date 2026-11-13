"""Score top-three retrieval against source, page and evidence text."""
import unicodedata
from typing import Any


def evidence_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    gold = [question, *question.get("alternative_evidence", [])]
    ranked_hits = []
    for rank, hit in enumerate(hits[:3], start=1):
        normalized = evidence_text(hit["text"])
        match = any(
            hit["document_id"] == item["gold_source_id"]
            and hit["page"] == item["gold_page"]
            and bool(item["evidence_quote"].strip())
            and evidence_text(item["evidence_quote"]) in normalized
            for item in gold
        )
        ranked_hits.append({**hit, "rank": rank, "evidence_hit": match})
    return {
        **question,
        "hits": ranked_hits,
        "hit_at_1": bool(ranked_hits and ranked_hits[0]["evidence_hit"]),
        "evidence_hit_at_3": any(hit["evidence_hit"] for hit in ranked_hits),
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for trace in traces:
        group = summary.setdefault(trace["strategy"], {}).setdefault(
            trace["split"], {"questions": 0, "hit_at_1": 0, "evidence_hit_at_3": 0}
        )
        group["questions"] += 1
        group["hit_at_1"] += trace["hit_at_1"]
        group["evidence_hit_at_3"] += trace["evidence_hit_at_3"]
    return summary
