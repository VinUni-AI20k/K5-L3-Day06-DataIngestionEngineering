"""Fixed windows and section-aware chunks."""
import re

from .models import Chunk, DocumentPage


def _chunk(page: DocumentPage, start: int, text: str) -> Chunk:
    return Chunk(f"{page.document_id}:{page.page}:{start}", page.document_id,
                 page.source, page.page, text, {**page.metadata, "start": start})


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    if not 0 <= overlap < size:
        raise ValueError("Expected 0 <= overlap < size.")
    chunks = []
    for start in range(0, len(page.text), size - overlap):
        text = page.text[start:start + size]
        if text.strip():
            chunks.append(_chunk(page, start, text))
        if start + size >= len(page.text):
            break
    return chunks


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    if max_size < 1:
        raise ValueError("max_size must be positive.")
    boundaries = [0]
    boundaries.extend(match.end() for match in re.finditer(
        r"\n\s*\n|\n(?=(?:Article \d+\.|CHAPTER |\d+\.\s|[a-zđ]\)\s))", page.text
    ))
    boundaries.append(len(page.text))
    chunks = []
    for start, end in zip(boundaries, boundaries[1:]):
        while start < end:
            stop = min(start + max_size, end)
            if stop < end and not page.text[stop].isspace():
                candidates = list(re.finditer(r"\s+", page.text[start:stop]))
                if candidates and candidates[-1].start() > 0:
                    stop = start + candidates[-1].start()
            text = page.text[start:stop]
            if text.strip():
                chunks.append(_chunk(page, start, text))
            start = stop
    return chunks
