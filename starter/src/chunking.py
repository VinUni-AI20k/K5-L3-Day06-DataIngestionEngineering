"""Implement fixed-size and structure-aware (heading/paragraph) chunking."""
import re
from .ingestion import _HEADING_RE
from .models import Chunk, DocumentPage

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def _split_oversized_paragraph(para: str, max_size: int) -> list[str]:
    """Split a paragraph larger than max_size along sentence boundaries first, then word boundaries."""
    if len(para) <= max_size:
        return [para]

    sentences = _SENTENCE_SPLIT_RE.split(para)
    pieces: list[str] = []
    current = ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        candidate = f"{current} {sent}".strip() if current else sent
        if len(candidate) <= max_size:
            current = candidate
        else:
            if current:
                pieces.append(current)
                current = ""
            if len(sent) <= max_size:
                current = sent
            else:
                # Fallback: split long sentence by words
                words = sent.split(" ")
                w_buf = ""
                for w in words:
                    w_cand = f"{w_buf} {w}".strip() if w_buf else w
                    if len(w_cand) <= max_size:
                        w_buf = w_cand
                    else:
                        if w_buf:
                            pieces.append(w_buf)
                        w_buf = w[:max_size]
                if w_buf:
                    current = w_buf

    if current:
        pieces.append(current)
    return pieces


def fixed_size(page: DocumentPage, size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Split page text into fixed-size character windows with overlap while retaining page metadata."""
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")

    # Flatten internal newlines to single spaces so character windows are uniform
    flat_text = re.sub(r"\s+", " ", page.text).strip()
    if not flat_text:
        return []

    step = size - overlap
    chunks: list[Chunk] = []
    idx = 0

    for start in range(0, len(flat_text), step):
        end = min(start + size, len(flat_text))
        window_text = flat_text[start:end].strip()
        if not window_text:
            continue

        chunk_meta = {
            **page.metadata,
            "chunk_strategy": "fixed_size",
            "chunk_index_on_page": idx,
            "char_start": start,
            "char_end": end,
            "char_length": len(window_text),
        }
        chunks.append(
            Chunk(
                chunk_id=f"{page.document_id}_p{page.page:02d}_fix_{idx:02d}",
                document_id=page.document_id,
                source=page.source,
                page=page.page,
                text=window_text,
                metadata=chunk_meta,
            )
        )
        idx += 1
        if end >= len(flat_text):
            break

    return chunks


def by_paragraph(page: DocumentPage, max_size: int = 800) -> list[Chunk]:
    """Group paragraphs and section headings up to max_size, splitting oversized paragraphs cleanly."""
    if max_size <= 0:
        raise ValueError("max_size must be positive")

    raw_paragraphs = [p.strip() for p in page.text.split("\n\n") if p.strip()]
    if not raw_paragraphs:
        return []

    chunks: list[Chunk] = []
    current_section = str(page.metadata.get("section", "Overview"))
    buffer: list[str] = []
    buffer_len = 0
    idx = 0

    def flush_buffer(sec_name: str) -> None:
        nonlocal buffer, buffer_len, idx
        if not buffer:
            return
        chunk_text = "\n\n".join(buffer).strip()
        if chunk_text:
            chunk_meta = {
                **page.metadata,
                "section": sec_name,
                "chunk_strategy": "by_paragraph",
                "chunk_index_on_page": idx,
                "char_length": len(chunk_text),
            }
            chunks.append(
                Chunk(
                    chunk_id=f"{page.document_id}_p{page.page:02d}_para_{idx:02d}",
                    document_id=page.document_id,
                    source=page.source,
                    page=page.page,
                    text=chunk_text,
                    metadata=chunk_meta,
                )
            )
            idx += 1
        buffer = []
        buffer_len = 0

    for para in raw_paragraphs:
        is_heading = bool(_HEADING_RE.match(para))
        if is_heading:
            # If we already have substantial content in buffer before a new section heading, flush first
            if buffer and buffer_len >= 120:
                flush_buffer(current_section)
            current_section = para
            buffer.append(para)
            buffer_len = len("\n\n".join(buffer))
            continue

        sub_paras = _split_oversized_paragraph(para, max_size=max_size)
        for sp in sub_paras:
            candidate_len = (buffer_len + 2 + len(sp)) if buffer else len(sp)
            if candidate_len <= max_size or (len(buffer) == 1 and _HEADING_RE.match(buffer[0])):
                # If buffer only has a heading and heading + sp slightly exceeds max_size, trim or keep together
                if candidate_len > max_size and len(buffer) == 1 and _HEADING_RE.match(buffer[0]):
                    heading_prefix = buffer[0]
                    buffer = []
                    buffer_len = 0
                    avail = max(200, max_size - len(heading_prefix) - 2)
                    resplit = _split_oversized_paragraph(sp, max_size=avail)
                    buffer.append(f"{heading_prefix}\n\n{resplit[0]}")
                    flush_buffer(current_section)
                    for rem in resplit[1:]:
                        buffer.append(rem)
                        flush_buffer(current_section)
                else:
                    buffer.append(sp)
                    buffer_len = candidate_len
            else:
                flush_buffer(current_section)
                buffer.append(sp)
                buffer_len = len(sp)

    if buffer:
        flush_buffer(current_section)

    return chunks
