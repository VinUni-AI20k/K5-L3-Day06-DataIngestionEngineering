"""Verify benchmark/questions.json against the normalized corpus (no retrieval).

Checks, per BENCHMARK.md / GUIDE.md:
  - every evidence_quote occurs on its gold page(s) after whitespace collapsing;
  - reports any OTHER page that also contains the quote (possible extra gold);
  - filter values exist in the corpus metadata and match the gold page;
  - counts: >=20 questions, >=15 dev, >=5 test, >=3 filter, >=1 filter in test;
  - unique ids; test diseases do not appear in dev (leakage guard).
Writes benchmark/verification.json. Exit code 1 if any hard check fails.

Run from starter/ after `python -m src.ingestion`:
    python scripts/verify_benchmark.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = ROOT / "benchmark" / "questions.json"
PAGES = ROOT / "starter" / "results" / "pages_normalized.jsonl"
OUT = ROOT / "benchmark" / "verification.json"


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    bench = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    qs = bench["questions"]
    pages = {}
    meta = {}
    for line in PAGES.read_text(encoding="utf-8").splitlines():
        p = json.loads(line)
        pages[p["page"]] = squash(p["text"])
        meta[p["page"]] = p["metadata"]

    errors, warnings, rows = [], [], []
    for q in qs:
        ev = squash(q["evidence_quote"])
        found_on = [pg for pg, txt in pages.items() if ev in txt]
        on_gold = [pg for pg in q["gold_pages"] if pg in found_on]
        extra = [pg for pg in found_on if pg not in q["gold_pages"]]
        if q["gold_page"] not in q["gold_pages"]:
            errors.append(f"{q['id']}: gold_page not in gold_pages")
        if not on_gold:
            errors.append(f"{q['id']}: evidence not found on gold page(s) {q['gold_pages']}")
        if extra:
            warnings.append(f"{q['id']}: evidence also on pages {extra} (review as extra gold)")
        if q["filter"]:
            for key, val in q["filter"].items():
                values = {m.get(key) for m in meta.values()}
                if val not in values:
                    errors.append(f"{q['id']}: filter {key}={val} not present in corpus metadata")
                if meta[q["gold_page"]].get(key) != val:
                    errors.append(f"{q['id']}: gold page metadata {key} != filter value {val}")
        if q.get("disease_no") is not None and meta[q["gold_page"]]["disease_no"] != q["disease_no"]:
            errors.append(f"{q['id']}: disease_no label does not match gold page metadata")
        rows.append({"id": q["id"], "split": q["split"], "found_on_pages": found_on,
                     "evidence_on_gold": bool(on_gold), "gold_section":
                     meta[q["gold_page"]]["section_at_start"]})

    ids = [q["id"] for q in qs]
    dev = [q for q in qs if q["split"] == "dev"]
    test = [q for q in qs if q["split"] == "test"]
    filt = [q for q in qs if q["filter"]]
    counts = {"total": len(qs), "dev": len(dev), "test": len(test), "with_filter": len(filt),
              "with_filter_test": sum(1 for q in filt if q["split"] == "test"),
              "by_type": {t: sum(1 for q in qs if q["type"] == t)
                          for t in sorted({q["type"] for q in qs})}}
    rules = [(len(set(ids)) == len(ids), "ids unique"),
             (counts["total"] >= 20, ">=20 questions"),
             (counts["dev"] >= 15, ">=15 dev"),
             (counts["test"] >= 5, ">=5 test"),
             (counts["with_filter"] >= 3, ">=3 filter questions"),
             (counts["with_filter_test"] >= 1, ">=1 filter question in test")]
    errors += [f"rule failed: {name}" for ok, name in rules if not ok]
    leak = ({q["disease_no"] for q in test if q["disease_no"]}
            & {q["disease_no"] for q in dev if q["disease_no"]})
    if leak:
        errors.append(f"test diseases also in dev: {sorted(leak)}")

    report = {"benchmark_version": bench["benchmark_version"], "counts": counts,
              "errors": errors, "warnings": warnings, "per_question": rows}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(counts, ensure_ascii=False))
    for w in warnings:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    print("OK: all checks passed" if not errors else f"{len(errors)} error(s)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
