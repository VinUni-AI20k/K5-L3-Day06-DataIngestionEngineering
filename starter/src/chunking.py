"""Chunking strategies. Chunks never cross page boundaries, so every chunk keeps its page."""
import re

from .models import Chunk, DocumentPage


def _meta(page: DocumentPage) -> dict:
    return {'page_id': page.metadata.get('page_id'), 'chapter': page.metadata.get('chapter')}


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Strategy A: character windows of `size` with `overlap`, inside one page."""
    if overlap >= size:
        raise ValueError('overlap must be smaller than size')
    text = page.text
    step = size - overlap
    chunks: list[Chunk] = []
    if not text:
        return chunks
    starts = range(0, max(len(text) - overlap, 1), step)
    for i, start in enumerate(starts):
        piece = text[start:start + size].strip()
        if not piece:
            continue
        chunk_id = f'{page.document_id}:p{page.page:03d}:fx{size}o{overlap}:{i:02d}'
        chunks.append(Chunk(chunk_id, page.document_id, page.source, page.page, piece,
                            {**_meta(page), 'strategy_params': f'fixed {size}/{overlap}', 'chunk_index': i}))
    return chunks


def _split_long(paragraph: str, max_size: int) -> list[str]:
    """Split an oversized paragraph at sentence ends, then by words if a sentence is still too long."""
    # Do not end a sentence after common abbreviations such as "Mr. Bingley" or "Mrs. Bennet".
    sentences = re.split(
        r'(?<!Mr\.)(?<!Mrs\.)(?<!Dr\.)(?<!St\.)(?<!Jr\.)(?<!Sr\.)(?<!Messrs\.)(?<=[.!?”"])\s+', paragraph)
    pieces, current = [], ''
    for sentence in sentences:
        while len(sentence) > max_size:
            cut = sentence.rfind(' ', 0, max_size)
            cut = cut if cut > 0 else max_size
            if current:
                pieces.append(current)
                current = ''
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        candidate = f'{current} {sentence}'.strip() if current else sentence
        if len(candidate) <= max_size:
            current = candidate
        else:
            pieces.append(current)
            current = sentence
    if current:
        pieces.append(current)
    return pieces


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Strategy B: group whole paragraphs up to `max_size` characters; split only oversized paragraphs."""
    paragraphs = [re.sub(r'\s+', ' ', p).strip() for p in page.text.split('\n\n')]
    paragraphs = [p for p in paragraphs if p]
    units: list[str] = []
    for paragraph in paragraphs:
        units.extend(_split_long(paragraph, max_size) if len(paragraph) > max_size else [paragraph])

    groups, current = [], ''
    for unit in units:
        candidate = f'{current}\n\n{unit}' if current else unit
        if len(candidate) <= max_size:
            current = candidate
        else:
            if current:
                groups.append(current)
            current = unit
    if current:
        groups.append(current)

    return [
        Chunk(f'{page.document_id}:p{page.page:03d}:pg{max_size}:{i:02d}', page.document_id, page.source,
              page.page, group, {**_meta(page), 'strategy_params': f'paragraph max {max_size}', 'chunk_index': i})
        for i, group in enumerate(groups)
    ]
