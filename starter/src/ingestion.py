"""Read and normalize the dental guideline PDF into traceable DocumentPage records.

Pipeline per ingested page (scope comes from data/manifest.json):
  1. Extract text with PyMuPDF (fitz); PDF page 1 is a scan, so its text comes
     from data/ocr/page_001.json plus reviewed corrections.
  2. normalize_text: fix the Word-2007 'ƣ/Ƣ' glyph to 'ư/Ư', Unicode NFC,
     unify spaces, strip blank lines.
  3. join_split_syllables: re-join syllables the PDF text layer split apart
     ('tuyế n' -> 'tuyến'), using a vocabulary built from this corpus.
  4. Drop the printed page number line; tag disease / section / subsection
     headings with character offsets so chunks can inherit them.

Run from starter/:  python -m src.ingestion
Writes results/pages_normalized.jsonl and results/ingestion_log.json.
"""
from __future__ import annotations

import collections
import json
import re
import unicodedata
from dataclasses import asdict
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from .models import DocumentPage

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "manifest.json"
OCR_DIR = ROOT / "data" / "ocr"
RESULTS = ROOT / "starter" / "results"

PUNCT = "\"'“”‘’()[]{},.;:!?…–-/*+"
# Vietnamese syllable finals; a split-off fragment equal to one of these is a
# strong sign the PDF text layer cut a syllable before its ending.
FINALS = {"c", "ch", "m", "n", "ng", "nh", "p", "t", "i", "y", "o", "u"}
# Join 'a b' when b is a syllable final and 'ab' is this many times more frequent
# than 'a' alone. 1.5 was chosen by listing every extra join it adds over 3.0
# (32 joins, all correct on manual review); see results/ingestion_log.json.
JOIN_RATIO = 1.5
MIN_JOINED_COUNT = 3
ROMAN = r"(?:I|II|III|IV|V|VI|VII|VIII|IX|X)"
SECTION_RE = re.compile(rf"^({ROMAN})\.\s*(\S.*)$")
SUBSECTION_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2})*)\.?\s+([A-ZÀ-ỸĐ].*)$")


# --------------------------------------------------------------------------- #
# Text normalization
# --------------------------------------------------------------------------- #
def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace without discarding line structure."""
    text = text.replace("ƣ", "ư").replace("Ƣ", "Ư")       # Word-2007 font glyph for ư/Ư
    text = unicodedata.normalize("NFC", text)
    text = text.replace(" ", " ").replace("\t", " ")
    lines = (re.sub(r" {2,}", " ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def _core(token: str) -> str:
    return token.strip(PUNCT).lower()


def build_vocabulary(texts: list[str]) -> collections.Counter:
    return collections.Counter(_core(w) for t in texts for w in t.split() if _core(w))


def _should_join(a: str, b: str, vocab: collections.Counter) -> bool:
    if not a or not b or a[-1] in PUNCT or b[0] in PUNCT:
        return False
    ca, cb = _core(a), _core(b)
    joined = ca + cb
    if not ca or not cb or vocab[joined] < MIN_JOINED_COUNT:
        return False
    if vocab[ca] == 0 or vocab[cb] == 0:         # one side is never a word on its own
        return True
    # 'tuyế n': the tail is a syllable final and the joined word is clearly more common.
    return cb in FINALS and vocab[joined] > JOIN_RATIO * vocab[ca]


def _fold(text: str) -> str:
    return re.sub(r"\s+", "", text).upper()


def restore_disease_heading(text: str, disease: dict) -> tuple[str, bool]:
    """Replace the disease heading line with the manifest title (verified against the
    page render) when the two differ only in whitespace, e.g. 'MẠ N TÍNH'."""
    target = f"{disease['disease_no']}. {disease['title']}"
    lines = text.split("\n")
    for i, line in enumerate(lines[:4]):
        if _fold(line).rstrip(".") == _fold(target) and line != target:
            lines[i] = target
            return "\n".join(lines), True
    return text, False


def join_split_syllables(text: str, vocab: collections.Counter) -> tuple[str, list[str]]:
    """Re-join syllables split by stray spaces. Returns new text and the joins made."""
    joins: list[str] = []
    out_lines = []
    for line in text.split("\n"):
        tokens = line.split(" ")
        merged: list[str] = []
        for tok in tokens:
            if merged and _should_join(merged[-1], tok, vocab):
                joins.append(f"{merged[-1]} {tok}")
                merged[-1] += tok
            else:
                merged.append(tok)
        out_lines.append(" ".join(merged))
    return "\n".join(out_lines), joins


# --------------------------------------------------------------------------- #
# OCR page
# --------------------------------------------------------------------------- #
def load_ocr_text(pdf_page: int) -> tuple[str, dict[str, Any]]:
    """Raw EasyOCR text + reviewed corrections; each correction must match once."""
    raw = json.loads((OCR_DIR / f"page_{pdf_page:03d}.json").read_text(encoding="utf-8"))
    fix = json.loads((OCR_DIR / f"page_{pdf_page:03d}_corrections.json").read_text(encoding="utf-8"))
    text = raw["text_raw"]
    for c in fix["corrections"]:
        n = text.count(c["wrong"])
        if n != 1:
            raise ValueError(f"OCR correction {c['wrong']!r} matched {n} times on page {pdf_page}")
        text = text.replace(c["wrong"], c["right"])
    info = {"tool": raw["tool"], "tool_version": raw["tool_version"],
            "mean_confidence": raw["mean_confidence"], "n_boxes": raw["n_boxes"],
            "n_corrections": len(fix["corrections"]),
            "correction_types": dict(collections.Counter(c["type"] for c in fix["corrections"]))}
    return text, info


# --------------------------------------------------------------------------- #
# Structure tagging
# --------------------------------------------------------------------------- #
def _strip_printed_page_number(text: str, printed: int | None) -> tuple[str, bool]:
    first, _, rest = text.partition("\n")
    if printed is not None and first.strip() == str(printed):
        return rest, True
    return text, False


def _subsection_label(number: str, rest: str) -> str:
    label = re.split(r":| - ", rest, maxsplit=1)[0].strip()
    return f"{number}. {label[:80]}".strip()


def tag_headings(text: str, disease: dict | None, state: dict) -> list[dict]:
    """Find headings in a page; update the running section/subsection state."""
    headings: list[dict] = []
    offset = 0
    for line in text.split("\n"):
        s = SECTION_RE.match(line)
        sub = SUBSECTION_RE.match(line)
        is_disease_heading = (disease is not None and sub is not None
                              and sub.group(1) == str(disease["disease_no"])
                              and line.upper() == line and state["section"] is None)
        if s and s.group(2).upper() == s.group(2) and len(line) <= 90:
            state["section"] = f"{s.group(1)}. {s.group(2).strip().rstrip(':').strip()}"
            state["subsection"] = None
            headings.append({"offset": offset, "level": "section", "text": state["section"]})
        elif is_disease_heading:
            headings.append({"offset": offset, "level": "disease", "text": disease["title"]})
        elif sub and state["section"] is not None:
            state["subsection"] = _subsection_label(sub.group(1), sub.group(2))
            headings.append({"offset": offset, "level": "subsection", "text": state["subsection"]})
        offset += len(line) + 1
    return headings


def _table_like(text: str) -> bool:
    lines = text.split("\n")
    short = sum(1 for line in lines if len(line.split()) <= 3)
    return len(lines) >= 15 and short / len(lines) > 0.5


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def load_pdf(path: Path, document_id: str, source: str,
             manifest: dict[str, Any] | None = None) -> tuple[list[DocumentPage], dict[str, Any]]:
    """Extract, normalize and tag every ingested page; return pages and a log."""
    manifest = manifest or load_manifest()
    doc = fitz.open(path)
    page_info = {p["pdf_page"]: p for p in manifest["pages"] if p["ingested"]}
    special = {p: s for s in manifest["special_sections"] for p in s["pages"]}
    disease_of = {p: d for d in manifest["diseases"]
                  for p in range(d["pdf_start"], d["pdf_end"] + 1)}

    log: dict[str, Any] = {"errors": [], "ocr": {}, "missing_printed_number": [],
                           "table_like_pages": [], "short_pages": [], "joins": {},
                           "headings_restored_from_manifest": [],
                           "source_defects": [
                               {"disease_no": 25, "pdf_page": 105,
                                "issue": "Source has no 'III. CHẨN ĐOÁN' heading (verified on the "
                                         "page render); diagnosis text is tagged with section "
                                         "'II. NGUYÊN NHÂN'. Not corrected, to avoid inventing "
                                         "structure."},
                               {"disease_no": 55, "issue": "Treatment heading reads 'IV. ĐIỀU TRỊ "
                                         "GÃY XƯƠNG HÀM TRÊN'; kept verbatim."}]}

    # Pass 1: extract + normalize (OCR for scanned pages).
    texts: dict[int, str] = {}
    sources: dict[int, str] = {}
    for p in sorted(page_info):
        if special.get(p, {}).get("text_source") == "ocr":
            raw, info = load_ocr_text(p)
            log["ocr"][p] = info
            sources[p] = "ocr"
        else:
            raw = doc[p - 1].get_text()
            sources[p] = "pdf_text"
        if not raw.strip():
            log["errors"].append({"pdf_page": p, "error": "no text extracted"})
        texts[p] = normalize_text(raw)

    # Pass 2: corpus vocabulary, then re-join split syllables.
    vocab = build_vocabulary(list(texts.values()))
    for p in texts:
        texts[p], joins = join_split_syllables(texts[p], vocab)
        if joins:
            log["joins"][p] = joins

    # Pass 3: page numbers + headings (state carries across pages of a disease).
    pages: list[DocumentPage] = []
    state = {"disease_no": None, "section": None, "subsection": None}
    for p in sorted(texts):
        info = page_info[p]
        disease = disease_of.get(p)
        text, had_number = _strip_printed_page_number(texts[p], info["printed_page"])
        if info["counted"] and not had_number:
            log["missing_printed_number"].append(p)
        if disease and disease["disease_no"] != state["disease_no"]:
            state = {"disease_no": disease["disease_no"], "section": None, "subsection": None}
            text, restored = restore_disease_heading(text, disease)
            if restored:
                log["headings_restored_from_manifest"].append(p)
        at_start = {"section": state["section"], "subsection": state["subsection"]}
        headings = tag_headings(text, disease, state) if disease else []
        if _table_like(text):
            log["table_like_pages"].append(p)
        if info["counted"] and len(text) < 200:
            log["short_pages"].append(p)
        sp = special.get(p)
        pages.append(DocumentPage(
            document_id=document_id, source=source, page=p, text=text,
            metadata={
                "printed_page": info["printed_page"],
                "counted": info["counted"],
                "text_source": sources[p],
                "special_id": sp["special_id"] if sp else None,
                "disease_no": disease["disease_no"] if disease else None,
                "disease": disease["title"] if disease else None,
                "section_at_start": sp["section"] if sp else at_start["section"],
                "subsection_at_start": at_start["subsection"],
                "headings": headings,
                "n_syllable_joins": len(log["joins"].get(p, [])),
                "chars": len(text),
            }))
    return pages, log


def section_at(page: DocumentPage, offset: int) -> tuple[str | None, str | None]:
    """Section and subsection in force at a character offset of a page."""
    section = page.metadata["section_at_start"]
    subsection = page.metadata["subsection_at_start"]
    for h in page.metadata["headings"]:
        if h["offset"] > offset:
            break
        if h["level"] == "section":
            section, subsection = h["text"], None
        elif h["level"] == "subsection":
            subsection = h["text"]
    return section, subsection


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    manifest = load_manifest()
    pdf = ROOT / manifest["source_path_or_url"]["local_path"]
    if not pdf.exists():
        raise SystemExit(f"Missing {pdf}. Download it from manifest download_url.")
    pages, log = load_pdf(pdf, manifest["source_id"], manifest["source_path_or_url"]["local_path"],
                          manifest)

    # Consistency check: every disease title must appear verbatim on its start page.
    titles_missing = [d["disease_no"] for d in manifest["diseases"]
                      if d["title"] not in next(pg for pg in pages if pg.page == d["pdf_start"]).text]

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "pages_normalized.jsonl").open("w", encoding="utf-8") as f:
        for page in pages:
            f.write(json.dumps(asdict(page), ensure_ascii=False) + "\n")

    counted = [p for p in pages if p.metadata["counted"]]
    sections = collections.Counter(h["text"].split(". ", 1)[-1]
                                   for p in pages for h in p.metadata["headings"]
                                   if h["level"] == "section")
    summary = {
        "pages_ingested": len(pages),
        "pages_counted": len(counted),
        "chars_counted_normalized": sum(len(p.text) for p in counted),
        "chars_ingested_normalized": sum(len(p.text) for p in pages),
        "syllable_joins_total": sum(len(v) for v in log["joins"].values()),
        "pages_with_joins": sorted(log["joins"]),
        "join_rule": {"min_joined_count": MIN_JOINED_COUNT, "ratio": JOIN_RATIO,
                      "finals": sorted(FINALS)},
        "disease_titles_missing_on_start_page": titles_missing,
        "headings_restored_from_manifest": log["headings_restored_from_manifest"],
        "section_heading_counts": dict(sections.most_common()),
        "diseases_without_section_headings": sorted(
            {p.metadata["disease_no"] for p in pages if p.metadata["disease_no"]}
            - {p.metadata["disease_no"] for p in pages
               if any(h["level"] == "section" for h in p.metadata["headings"])}),
    }
    out = {"summary": summary, **log}
    (RESULTS / "ingestion_log.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"errors: {log['errors'] or 'none'} | missing printed number: "
          f"{log['missing_printed_number'] or 'none'}")
    print(f"table-like pages: {log['table_like_pages']} | short pages: {log['short_pages']}")


if __name__ == "__main__":
    main()
