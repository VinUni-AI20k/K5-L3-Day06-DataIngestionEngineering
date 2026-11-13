"""Automatic retrieval metrics: Hit@1 and Evidence Hit@3, with per-query traces.

A hit = a retrieved chunk from the gold source and gold page whose text contains the whole
evidence_quote (whitespace-normalised). No alternative gold evidence is documented in this benchmark,
so only one evidence string per question is accepted.
"""
import re
from typing import Any


def _norm(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def gold_page_number(gold_page: str) -> int:
    """'pride-and-prejudice:p001' -> 1"""
    match = re.search(r':p(\d+)$', gold_page)
    if not match:
        raise ValueError(f'gold_page must look like doc:pNNN, got {gold_page!r}')
    return int(match.group(1))


def is_evidence_hit(question: dict[str, Any], hit: dict[str, Any]) -> bool:
    return (hit['document_id'] == question['gold_source_id']
            and hit['page'] == gold_page_number(question['gold_page'])
            and _norm(question['evidence_quote']) in _norm(hit['text']))


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    flags = [is_evidence_hit(question, h) for h in hits[:3]]
    return {
        'id': question['id'],
        'split': question['split'],
        'filtered': question.get('filter') is not None,
        'hit1': bool(flags[:1] and flags[0]),
        'hit3': any(flags),
        'top3': [{'chunk_id': h['id'], 'page': h['page'], 'chapter': h.get('chapter'),
                  'score': round(float(h['score']), 4), 'is_gold': f}
                 for h, f in zip(hits[:3], flags)],
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Hit@1 and Evidence Hit@3 counts by split, reported as n/N and a percentage."""
    out: dict[str, Any] = {}
    for split in ('dev', 'test', 'all'):
        rows = [t for t in traces if split == 'all' or t['split'] == split]
        n = len(rows)
        h1 = sum(t['hit1'] for t in rows)
        h3 = sum(t['hit3'] for t in rows)
        out[split] = {'n': n, 'hit1': h1, 'hit3': h3,
                      'hit1_pct': round(100 * h1 / n, 1) if n else None,
                      'hit3_pct': round(100 * h3 / n, 1) if n else None}
    return out
