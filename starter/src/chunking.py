"""Fixed-size (strategy A) and paragraph-aware (strategies B/C) chunking.

Chunks never cross a page boundary, so every chunk keeps exactly one source page.
Chunk IDs are deterministic: same page text + same parameters -> same ID, which lets
the store upsert without duplicates and detect changed pages (new IDs).
"""
from __future__ import annotations

import hashlib
import re

from .models import Chunk, DocumentPage

_SENTENCE_SPLIT = re.compile(r"(?<=[.;:?!])\s+")


def _make_chunk(page: DocumentPage, kind: str, index: int, text: str, extra: dict) -> Chunk:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    chunk_id = f"{page.document_id}:p{page.page}:{kind}{index}:{digest}"
    meta = {k: v for k, v in page.metadata.items() if k in ("edition", "part", "language")}
    meta.update(extra, chunk_index=index, char_count=len(text))
    return Chunk(chunk_id, page.document_id, page.source, page.page, text, meta)


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Slide a window of ``size`` characters with ``overlap`` characters shared."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("require size > 0 and 0 <= overlap < size")
    text = page.text
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append(_make_chunk(page, "f", len(chunks), piece,
                                      {"char_start": start, "chunking": f"fixed{size}o{overlap}"}))
        if start + size >= len(text):
            break
        start += size - overlap
    return chunks


def _split_long(paragraph: str, max_size: int) -> list[str]:
    """Split an oversized paragraph at sentence ends; hard-cut a single huge sentence."""
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_SPLIT.split(paragraph):
        while len(sentence) > max_size:  # no sentence boundary available
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:max_size])
            sentence = sentence[max_size:]
        if current and len(current) + 1 + len(sentence) > max_size:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Pack whole paragraphs (blank-line separated) up to ``max_size`` characters."""
    units: list[str] = []
    for para in (p.strip() for p in page.text.split("\n\n")):
        if para:
            units.extend(_split_long(para, max_size) if len(para) > max_size else [para])

    chunks: list[Chunk] = []
    current: list[str] = []
    for unit in units:
        if current and len("\n\n".join(current + [unit])) > max_size:
            chunks.append(_make_chunk(page, "p", len(chunks), "\n\n".join(current),
                                      {"chunking": f"paragraph{max_size}"}))
            current = []
        current.append(unit)
    if current:
        chunks.append(_make_chunk(page, "p", len(chunks), "\n\n".join(current),
                                  {"chunking": f"paragraph{max_size}"}))
    return chunks
