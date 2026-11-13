"""Evaluate retrieved evidence and preserve per-query traces.

A retrieved chunk is a *hit* when it comes from a gold (source, page) - the main gold or
a documented alternative - AND contains that gold's full evidence quote (whitespace
collapsed). Evidence split across two chunks is therefore NOT a hit; it is flagged as
``evidence_split`` for error analysis. Coherence and grounding are human labels and are
not inferred here.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _golds(question: dict[str, Any]) -> list[tuple[str, int, str]]:
    golds = [(question["gold_source_id"], question["gold_page"], question["evidence_quote"])]
    golds += [(a["source_id"], a["page"], a["evidence_quote"])
              for a in question.get("alternatives", [])]
    return golds


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    golds = _golds(question)
    ranked = []
    first_hit = None
    split_evidence = False
    for rank, h in enumerate(hits, start=1):
        text = _squash(h["text"])
        page_match = any(h["document_id"] == s and h["page"] == p for s, p, _ in golds)
        evidence_match = any(h["document_id"] == s and h["page"] == p and _squash(q) in text
                             for s, p, q in golds)
        # partial evidence: chunk on a gold page holds the first or last 40 chars only
        if page_match and not evidence_match:
            for s, p, q in golds:
                q = _squash(q)
                if h["document_id"] == s and h["page"] == p and (q[:40] in text or q[-40:] in text):
                    split_evidence = True
        if evidence_match and first_hit is None:
            first_hit = rank
        ranked.append({"rank": rank, "chunk_id": h["chunk_id"], "document_id": h["document_id"],
                       "page": h["page"], "score": h["score"], "metadata": h["metadata"],
                       "page_match": page_match, "evidence_hit": evidence_match,
                       "text": h["text"]})
    filt = question.get("filter")
    compliant = None
    if filt:
        compliant = all(all(str(r["metadata"].get(k)) == str(v) for k, v in filt.items())
                        for r in ranked)
    return {
        "id": question["id"], "split": question["split"], "type": question.get("type"),
        "question": question["question"], "filter": filt,
        "gold": [{"source_id": s, "page": p} for s, p, _ in golds],
        "hit_at_1": first_hit == 1, "hit_at_3": first_hit is not None and first_hit <= 3,
        "first_hit_rank": first_hit,
        "page_hit_at_3": any(r["page_match"] for r in ranked),
        "evidence_split": split_evidence and first_hit is None,
        "filter_compliant_top3": compliant,
        "hits": ranked,
    }


def summarize(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per (strategy, split); counts are reported as n/N, never % alone."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for t in traces:
        groups[(t["strategy"], t["split"])].append(t)
    rows = []
    for (strategy, split), ts in sorted(groups.items()):
        n = len(ts)
        filt = [t for t in ts if t["filter"]]
        rows.append({
            "strategy": strategy, "split": split, "N": n,
            "hit_at_1": f"{sum(t['hit_at_1'] for t in ts)}/{n}",
            "evidence_hit_at_3": f"{sum(t['hit_at_3'] for t in ts)}/{n}",
            "page_hit_at_3": f"{sum(t['page_hit_at_3'] for t in ts)}/{n}",
            "evidence_split": sum(t["evidence_split"] for t in ts),
            "filter_compliant_top3": (f"{sum(bool(t['filter_compliant_top3']) for t in filt)}"
                                      f"/{len(filt)}" if filt else "N/A"),
        })
    return rows
