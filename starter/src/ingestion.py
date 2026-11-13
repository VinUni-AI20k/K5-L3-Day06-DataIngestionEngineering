"""Read and normalize source data.

Corpus: World Bank "Điểm lại" (Taking Stock) PDFs listed in data/manifest.json.
Each physical PDF page becomes one DocumentPage; only pages counted in the manifest
are kept for the corpus, and every skipped/empty/failed page is logged.
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from .models import DocumentPage

log = logging.getLogger(__name__)

# Running header printed at the top of every page of these reports.
RUNNING_HEADER = "ĐIỂM LẠI Cập nhật tình hình phát triển kinh tế Việt Nam"

_UPPER = "A-ZÀ-ỸĐ"  # Latin + Vietnamese capitals (after NFC)
# Lines that start a new block: numbered paragraph "17. Văn bản…", box/figure/table
# captions, source notes, footnotes "3 Theo các quy định…".
_BLOCK_START = re.compile(
    rf"^(\d{{1,3}}\.\s+[{_UPPER}]"
    rf"|(Hộp|Hình|Bảng)\s+[IVX\d]"
    rf"|Nguồn\s*:"
    rf"|\d{{1,2}}\s+[{_UPPER}][a-zà-ỹđ])"
)
_SENTENCE_END = re.compile(r"[.:;?!”\")]\s*$")
_HAS_LETTER = re.compile(r"[A-Za-zÀ-ỹĐđ]{2,}")


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract every physical page, keeping page number, source and raw length.

    Pages that fail to extract or have no text are logged and returned with
    metadata ``extract_error`` / ``empty`` so the caller can report them.
    """
    reader = PdfReader(path)
    pages: list[DocumentPage] = []
    for number, pdf_page in enumerate(reader.pages, start=1):
        meta: dict[str, Any] = {}
        try:
            raw = pdf_page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001 — record any extractor failure
            log.error("%s p%d: extract failed: %s", document_id, number, exc)
            raw, meta["extract_error"] = "", str(exc)
        text = normalize_text(raw)
        meta["raw_chars"] = len(raw.strip())
        if not text:
            meta["empty"] = True
            log.warning("%s p%d: no text after normalization", document_id, number)
        pages.append(DocumentPage(document_id, source, number, text, meta))
    return pages


def normalize_text(text: str) -> str:
    """Normalize Unicode/whitespace and rebuild paragraphs.

    - NFC so composed and decomposed Vietnamese diacritics compare equal.
    - Drop the running header and the bare page-number line.
    - Drop lines with no words (chart tick labels such as ``30.5 31.0 32.6``).
    - Join wrapped lines into paragraphs; paragraphs are separated by a blank line.
    """
    text = unicodedata.normalize("NFC", text).replace(" ", " ")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    if lines and lines[0] == RUNNING_HEADER:
        lines = lines[1:]
    if lines and lines[0].isdigit():
        lines = lines[1:]

    paragraphs: list[list[str]] = []
    prev = ""
    for ln in lines:
        if not _HAS_LETTER.search(ln):
            continue
        new_block = (not paragraphs or _BLOCK_START.match(ln)
                     or (_SENTENCE_END.search(prev) and ln[0].isupper() and len(prev) < 60))
        if new_block:
            paragraphs.append([ln])
        else:
            paragraphs[-1].append(ln)
        prev = ln
    return "\n\n".join(" ".join(p) for p in paragraphs).strip()


def load_corpus(manifest_path: Path, root: Path) -> tuple[list[DocumentPage], list[dict]]:
    """Load the counted pages declared in the manifest, with filter metadata.

    Returns (pages, ingest_log). Pages outside the manifest's counted list are
    skipped and logged so the page count rule stays auditable.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pages: list[DocumentPage] = []
    events: list[dict] = []
    for src in manifest["sources"]:
        counted = {p["page"]: p["part"] for p in src["pages"]}
        for page in load_pdf(root / src["local_path"], src["source_id"], src["local_path"]):
            if page.page not in counted:
                events.append({"document_id": page.document_id, "page": page.page,
                               "event": "skipped_not_counted"})
                continue
            if page.metadata.get("empty") or page.metadata.get("extract_error"):
                events.append({"document_id": page.document_id, "page": page.page,
                               "event": "empty_or_error",
                               "detail": page.metadata.get("extract_error", "")})
                continue
            page.metadata.update(edition=src["edition"], part=counted[page.page],
                                 language=src["language"])
            pages.append(page)
    return pages, events
