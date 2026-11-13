"""Verify benchmark/questions.json against the normalized corpus.

Checks, for every question:
- required fields, split counts (>=15 dev, >=5 test), >=3 filter questions with >=1 in test;
- the gold evidence (and each documented alternative) occurs verbatim on the stated page
  after whitespace collapsing;
- for filter questions, whether the evidence also appears in the other edition
  (informational: it tells whether the filter is what disambiguates the answer).

Usage (from starter/):  python scripts/verify_benchmark.py
Exit code 1 if any check fails.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion import load_corpus  # noqa: E402

REQUIRED = ["id", "split", "question", "expected_answer", "gold_source_id",
            "gold_page", "evidence_quote", "filter"]


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    bench = json.loads((ROOT.parent / "benchmark" / "questions.json").read_text(encoding="utf-8"))
    pages, _ = load_corpus(ROOT / "data" / "manifest.json", ROOT)
    text = {(p.document_id, p.page): squash(p.text) for p in pages}

    errors: list[str] = []
    qs = bench["questions"]
    ids = [q["id"] for q in qs]
    if len(set(ids)) != len(ids):
        errors.append("duplicate ids")
    for q in qs:
        missing = [f for f in REQUIRED if f not in q]
        if missing:
            errors.append(f"{q.get('id')}: missing {missing}")
            continue
        golds = [(q["gold_source_id"], q["gold_page"], q["evidence_quote"])]
        golds += [(a["source_id"], a["page"], a["evidence_quote"]) for a in q.get("alternatives", [])]
        for src, page, quote in golds:
            page_text = text.get((src, page))
            if page_text is None:
                errors.append(f"{q['id']}: page {src} p{page} is not in the counted corpus")
            elif squash(quote) not in page_text:
                errors.append(f"{q['id']}: evidence not found on {src} p{page}: {quote[:60]}…")
            else:
                print(f"OK   {q['id']:4} {q['split']:4} {src} p{page}")
        if q["filter"]:
            others = [k for k, v in text.items()
                      if k[0] != q["gold_source_id"] and squash(q["evidence_quote"]) in v]
            print(f"     {q['id']} filter={q['filter']} evidence also in other edition: {bool(others)}")

    dev = sum(q["split"] == "dev" for q in qs)
    test = sum(q["split"] == "test" for q in qs)
    filt = [q for q in qs if q["filter"]]
    filt_test = sum(q["split"] == "test" for q in filt)
    print(f"\nquestions={len(qs)} dev={dev} test={test} filter={len(filt)} (test={filt_test})")
    if dev < 15 or test < 5:
        errors.append("need >=15 dev and >=5 test")
    if len(filt) < 3 or filt_test < 1:
        errors.append("need >=3 filter questions with >=1 in test")
    for e in errors:
        print("FAIL", e)
    print("ALL CHECKS PASSED" if not errors else f"{len(errors)} check(s) failed")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
