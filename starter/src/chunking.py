"""Chunking strategies A (fixed-size) and B (structure-aware); C reuses B.

Every chunk is a contiguous slice of the normalized page text produced by
src/ingestion.py, so its page list and section/subsection can be derived from
character offsets. Front-matter special sections (Quyết định, title page,
editorial board, table of contents, abbreviations) are ONE unsplit chunk in
every strategy.

Run from starter/ after `python -m src.ingestion`:
    python -m src.chunking
Writes results/chunks_A.jsonl, results/chunks_B.jsonl (git-ignored: full text)
and results/chunk_stats.json.
"""
from __future__ import annotations

import hashlib
import json
import re
import statistics
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import Chunk, DocumentPage

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "starter" / "results"
PAGES_FILE = RESULTS / "pages_normalized.jsonl"
QUESTIONS = ROOT / "benchmark" / "questions.json"

# Stable namespace so the same chunk always gets the same Qdrant point id.
NAMESPACE = uuid.UUID("6f0d5b8e-3c1a-4f7e-9a51-2a2b3c4d5e6f")

STRATEGY_PARAMS = {
    "A": {"size": 500, "overlap": 50},
    "B": {"max_size": 1000, "min_size": 150},
}

# A line opens a new paragraph if it starts with a list/heading marker.
PARA_START = re.compile(r"^(?:[-+•]\s|[a-zđ]\.\s?|\d{1,2}(?:\.\d{1,2})*\.?\s|"
                        r"(?:I|II|III|IV|V|VI|VII|VIII|IX|X)\.\s)")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def chunk_id(strategy: str, document_id: str, start_key: str, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return str(uuid.uuid5(NAMESPACE, f"{strategy}|{document_id}|{start_key}|{digest}"))


class Span:
    """Several consecutive pages joined with '\\n', with offset -> page mapping."""

    def __init__(self, pages: list[DocumentPage]):
        self.pages = pages
        self.starts: list[int] = []
        parts, pos = [], 0
        for p in pages:
            self.starts.append(pos)
            parts.append(p.text)
            pos += len(p.text) + 1
        self.text = "\n".join(parts)
        # Global heading list (offset, level, text), in document order.
        self.headings = [(self.starts[i] + h["offset"], h["level"], h["text"])
                         for i, p in enumerate(pages) for h in p.metadata["headings"]]

    def page_index(self, offset: int) -> int:
        idx = 0
        for i, s in enumerate(self.starts):
            if s <= offset:
                idx = i
        return idx

    def pages_between(self, start: int, end: int) -> list[int]:
        return [self.pages[i].page for i in range(self.page_index(start), self.page_index(max(start, end - 1)) + 1)]

    def section_at(self, offset: int) -> tuple[str | None, str | None]:
        first = self.pages[0].metadata
        section, subsection = first["section_at_start"], first["subsection_at_start"]
        for off, level, text in self.headings:
            if off > offset:
                break
            if level == "section":
                section, subsection = text, None
            elif level == "subsection":
                subsection = text
        return section, subsection

    def sections_touched(self, start: int, end: int) -> list[str]:
        out = [self.section_at(start)[0]]
        out += [t for off, lvl, t in self.headings if lvl == "section" and start < off < end]
        return [s for s in dict.fromkeys(out) if s]


def make_chunk(strategy: str, span: Span, start: int, end: int, index: int) -> Chunk | None:
    raw = span.text[start:end]
    lead = len(raw) - len(raw.lstrip())
    text = raw.strip()
    if not text:
        return None
    start += lead
    end = start + len(text)
    first = span.pages[span.page_index(start)]
    section, subsection = span.section_at(start)
    pages = span.pages_between(start, end)
    md = first.metadata
    return Chunk(
        chunk_id=chunk_id(strategy, first.document_id, f"{first.page}:{start - span.starts[span.page_index(start)]}", text),
        document_id=first.document_id, source=first.source, page=pages[0], text=text,
        metadata={
            "strategy": strategy, "chunk_index": index,
            "page_end": pages[-1], "pages": pages,
            "printed_page": md["printed_page"],
            "disease_no": md["disease_no"], "disease": md["disease"],
            "special_id": md["special_id"],
            "section": section, "subsection": subsection,
            "sections": span.sections_touched(start, end),
            "char_start": start - span.starts[span.page_index(start)],
            "chars": len(text),
        })


# --------------------------------------------------------------------------- #
# Strategy A: fixed-size windows inside each page
# --------------------------------------------------------------------------- #
def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split one page into overlapping character windows (no boundary snapping)."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    span = Span([page])
    chunks, start, idx = [], 0, 0
    while start < len(span.text):
        end = min(start + size, len(span.text))
        c = make_chunk("A", span, start, end, idx)
        if c:
            chunks.append(c)
            idx += 1
        if end == len(span.text):
            break
        start = end - overlap
    return chunks


# --------------------------------------------------------------------------- #
# Strategy B: structure-aware
# --------------------------------------------------------------------------- #
def _paragraph_starts(text: str, lo: int, hi: int) -> tuple[list[int], list[int]]:
    """Soft split points in (lo, hi), in two preference levels.

    Level 1: a top-level item starts ('- ', 'a.', '1.', roman). Level 2: a '+'
    sub-item starts or the previous line ends a sentence. Cutting at level 1
    keeps a '- heading' together with its '+' sub-items (fixes dev d10).
    """
    major, minor, pos, prev = [], [], 0, ""
    for line in text.split("\n"):
        if lo < pos < hi:
            if PARA_START.match(line) and not line.startswith("+"):
                major.append(pos)
            elif line.startswith("+") or prev.rstrip().endswith((".", ":", ";", "!", "?")):
                minor.append(pos)
        prev = line
        pos += len(line) + 1
    return major, minor


def _split_long(text: str, lo: int, hi: int, max_size: int) -> list[tuple[int, int]]:
    """Greedy split of [lo, hi): top-level items, then sub-items/sentence-ending
    lines, then sentence ends, then a hard cut. A cut must leave at least 30% of
    max_size behind, so preferring top-level items never yields tiny pieces."""
    pieces, start = [], lo
    major, minor = _paragraph_starts(text, lo, hi)
    while hi - start > max_size:
        limit = start + max_size
        floor = start + int(0.3 * max_size)
        cands = ([s for s in major if floor < s <= limit]
                 or [s for s in minor if floor < s <= limit]
                 or [s for s in major + minor if start < s <= limit])
        if cands:
            cut = max(cands)
        else:
            seg = text[start:limit]
            m = list(re.finditer(r"[.!?;]\s", seg))
            cut = start + m[-1].end() if m else limit
        pieces.append((start, cut))
        start = cut
    pieces.append((start, hi))
    return pieces


def by_structure(pages: list[DocumentPage], max_size: int, min_size: int) -> list[Chunk]:
    """Chunk one disease (consecutive pages) at heading boundaries."""
    span = Span(pages)
    text = span.text
    hard = sorted({0, *[off for off, lvl, _ in span.headings if lvl in ("section", "subsection")], len(text)})
    units = [(a, b) for a, b in zip(hard, hard[1:]) if text[a:b].strip()]
    # Merge tiny units (bare headings like 'IV. ĐIỀU TRỊ') forward into the next unit.
    merged: list[tuple[int, int]] = []
    carry = None
    for a, b in units:
        a = carry if carry is not None else a
        if len(text[a:b].strip()) < min_size and (a, b) != units[-1]:
            carry = a
            continue
        carry = None
        merged.append((a, b))
    chunks, idx = [], 0
    for a, b in merged:
        for s, e in _split_long(text, a, b, max_size):
            c = make_chunk("B", span, s, e, idx)
            if c:
                chunks.append(c)
                idx += 1
    return chunks


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Starter-compatible single-page variant of strategy B."""
    return by_structure([page], max_size=max_size, min_size=STRATEGY_PARAMS["B"]["min_size"])


def special_section(strategy: str, pages: list[DocumentPage]) -> Chunk:
    """One unsplit chunk for a front-matter section (may span pages)."""
    span = Span(pages)
    chunk = make_chunk(strategy, span, 0, len(span.text), 0)
    assert chunk is not None
    return chunk


# --------------------------------------------------------------------------- #
# Build all strategies
# --------------------------------------------------------------------------- #
def load_pages() -> list[DocumentPage]:
    return [DocumentPage(**json.loads(line))
            for line in PAGES_FILE.read_text(encoding="utf-8").splitlines()]


def build(strategy: str, pages: list[DocumentPage]) -> list[Chunk]:
    groups: dict[tuple[str, Any], list[DocumentPage]] = {}
    for p in pages:
        key = ("special", p.metadata["special_id"]) if p.metadata["special_id"] else \
              ("disease", p.metadata["disease_no"])
        groups.setdefault(key, []).append(p)
    chunks: list[Chunk] = []
    for (kind, _), group in groups.items():
        if kind == "special":
            chunks.append(special_section(strategy, group))
        elif strategy == "A":
            for p in group:
                chunks.extend(fixed_size(p, **STRATEGY_PARAMS["A"]))
        elif strategy == "B":
            chunks.extend(by_structure(group, **STRATEGY_PARAMS["B"]))
        else:
            raise ValueError(f"unknown strategy {strategy}")
    for i, c in enumerate(chunks):
        c.metadata["chunk_index"] = i
    ids = [c.chunk_id for c in chunks]
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"duplicate chunk ids in strategy {strategy}")
    return chunks


def _squash(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def evidence_coverage(chunks: list[Chunk], split: str = "dev") -> dict[str, Any]:
    """Upper bound for Evidence Hit@k: does ANY chunk on a gold page hold the whole
    evidence? Computed on dev only so chunking is never tuned on test."""
    qs = [q for q in json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"] if q["split"] == split]
    texts = [(set(c.metadata["pages"]), _squash(c.text)) for c in chunks]
    missing = [q["id"] for q in qs
               if not any(set(q["gold_pages"]) & pg and _squash(q["evidence_quote"]) in t for pg, t in texts)]
    return {"split": split, "covered": len(qs) - len(missing), "total": len(qs), "not_coverable": missing}


def stats(chunks: list[Chunk]) -> dict[str, Any]:
    lens = sorted(len(c.text) for c in chunks)
    q = statistics.quantiles(lens, n=10)
    return {
        "chunks": len(chunks),
        "chars_total": sum(lens),
        "len_min": lens[0], "len_p10": round(q[0]), "len_median": round(statistics.median(lens)),
        "len_mean": round(statistics.mean(lens)), "len_p90": round(q[-1]), "len_max": lens[-1],
        "multi_page_chunks": sum(1 for c in chunks if len(c.metadata["pages"]) > 1),
        "multi_section_chunks": sum(1 for c in chunks if len(c.metadata["sections"]) > 1),
        "special_chunks": sum(1 for c in chunks if c.metadata["special_id"]),
    }


def main() -> None:
    pages = load_pages()
    report = {"params": STRATEGY_PARAMS, "strategies": {}}
    for strategy in ("A", "B"):
        chunks = build(strategy, pages)
        with (RESULTS / f"chunks_{strategy}.jsonl").open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        report["strategies"][strategy] = {**stats(chunks), "dev_evidence_coverage": evidence_coverage(chunks)}
    report["strategies"]["C"] = {"same_as": "B", "note": "C reuses B's chunks and vectors; it only adds a disease_no filter at query time."}
    (RESULTS / "chunk_stats.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["strategies"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
