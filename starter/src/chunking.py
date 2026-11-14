"""Implement fixed-size and structure-aware chunking."""
import hashlib
import re

from .models import Chunk, DocumentPage


ARTICLE_HEADING = re.compile(r"(?m)^Điều\s+\d+[a-zA-Z]?\.")
CLAUSE_OR_POINT = re.compile(r"(?mi)^(?:\d{1,3}\.\s|[a-zđ]\)\s)")


def _make_chunk(
    page: DocumentPage,
    strategy: str,
    index: int,
    text: str,
) -> Chunk:
    chunk_id_input = (
        f"{strategy}\0{page.document_id}\0{page.page}\0{index}\0{text}"
    )
    digest = hashlib.sha256(chunk_id_input.encode("utf-8")).hexdigest()[:16]

    metadata = dict(page.metadata)
    metadata["chunking_strategy"] = strategy
    metadata["chunk_index"] = index

    return Chunk(
        chunk_id=f"{strategy}-{digest}",
        document_id=page.document_id,
        source=page.source,
        page=page.page,
        text=text,
        metadata=metadata,
    )


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split a page into fixed character windows with overlap."""
    if size <= 0:
        raise ValueError("size phải lớn hơn 0")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap phải từ 0 đến nhỏ hơn size")

    text = page.text.strip()
    if not text:
        return []

    chunks = []
    step = size - overlap
    start = 0

    while start < len(text):
        end = min(start + size, len(text))
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(
                _make_chunk(page, "fixed", len(chunks), chunk_text)
            )

        if end == len(text):
            break
        start += step

    return chunks


def _split_long_paragraph(
    text: str,
    max_size: int,
    overlap: int = 300,
) -> list[str]:
    """Split oversized sections near whitespace with context overlap."""
    pieces = []
    text = text.strip()
    overlap = min(overlap, max_size // 2)
    start = 0

    while len(text) - start > max_size:
        split_at = start + max_size
        while split_at > start and not text[split_at].isspace():
            split_at -= 1

        if split_at == start:
            split_at = start + max_size

        piece = text[start:split_at].strip()
        if piece:
            pieces.append(piece)

        next_start = max(start + 1, split_at - overlap)
        while next_start < split_at and not text[next_start].isspace():
            next_start += 1
        while next_start < len(text) and text[next_start].isspace():
            next_start += 1
        start = next_start

    remaining = text[start:].strip()
    if remaining:
        pieces.append(remaining)

    return pieces


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Keep legal article boundaries where possible and exclude PDF page numbers."""
    if max_size <= 0:
        raise ValueError("max_size phải lớn hơn 0")

    text = page.text.strip()
    if not text:
        return []
    if text == str(page.page):
        return []

    page_number_prefix = re.compile(rf"^{page.page}[ \t]*\n(?:[ \t]*\n)?")
    text = page_number_prefix.sub("", text, count=1)
    article_starts = [match.start() for match in ARTICLE_HEADING.finditer(text)]
    if article_starts:
        sections = []
        if article_starts[0] > 0:
            sections.append(text[:article_starts[0]].strip())
        sections.extend(
            text[start:end].strip()
            for start, end in zip(article_starts, article_starts[1:] + [len(text)])
        )
    else:
        sections = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", text)
            if paragraph.strip()
        ]

    segments = []
    for section in sections:
        if not section:
            continue
        if len(section) <= max_size:
            segments.append(section)
        else:
            clause_starts = [
                match.start() for match in CLAUSE_OR_POINT.finditer(section)
            ]
            if clause_starts:
                boundaries = [0, *clause_starts, len(section)]
                section_parts = [
                    section[start:end].strip()
                    for start, end in zip(boundaries, boundaries[1:])
                    if section[start:end].strip()
                ]
                for part in section_parts:
                    if len(part) <= max_size:
                        segments.append(part)
                    else:
                        segments.extend(_split_long_paragraph(part, max_size))
            else:
                segments.extend(_split_long_paragraph(section, max_size))

    chunk_texts = []
    current_parts = []
    current_length = 0

    for segment in segments:
        added_length = len(segment) + (2 if current_parts else 0)

        if current_parts and current_length + added_length > max_size:
            chunk_texts.append("\n\n".join(current_parts))
            current_parts = []
            current_length = 0
            added_length = len(segment)

        current_parts.append(segment)
        current_length += added_length

    if current_parts:
        chunk_texts.append("\n\n".join(current_parts))

    return [
        _make_chunk(page, "paragraph", index, chunk_text)
        for index, chunk_text in enumerate(chunk_texts)
    ]