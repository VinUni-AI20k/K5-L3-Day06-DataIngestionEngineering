"""Read and normalize source data from the generated PDF."""
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from .models import DocumentPage

_ROMAN = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100}
_HEADING = re.compile(r'^Chapter ([IVXLC]+)$')


def _roman_to_int(numeral: str) -> int:
    total, prev = 0, 0
    for ch in reversed(numeral):
        value = _ROMAN[ch]
        total = total - value if value < prev else total + value
        prev = max(prev, value)
    return total


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace; keep paragraph breaks ('\\n\\n') and drop page furniture.

    Removes lines that are only a stray ']' (left over from removed [Illustration] markers).
    Page-number footers are removed by load_pdf, which knows the page number.
    """
    text = unicodedata.normalize('NFC', text).replace(' ', ' ').replace(' ', ' ')
    lines = []
    for raw in text.split('\n'):
        line = re.sub(r'[ \t]+', ' ', raw).strip()
        if line == ']':
            continue
        lines.append(line)
    text = '\n'.join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract each physical page with its number, chapter (from 'Chapter N' headings) and empty flag.

    Layout mode keeps paragraph gaps as blank lines. Pages with no text are kept and flagged
    with metadata['empty'] = True. OCR is not performed.
    """
    reader = PdfReader(str(path))
    pages: list[DocumentPage] = []
    chapter = None
    for number, page in enumerate(reader.pages, start=1):
        raw = page.extract_text(extraction_mode='layout') or ''
        lines = [l.strip() for l in raw.split('\n')]
        # Drop the footer page number that equals this page's index.
        while lines and (not lines[-1] or lines[-1] == str(number)):
            lines.pop()
        for line in lines[:3]:
            match = _HEADING.match(line)
            if match:
                chapter = _roman_to_int(match.group(1))
                break
        text = normalize_text('\n'.join(lines))
        pages.append(DocumentPage(
            document_id=document_id,
            source=source,
            page=number,
            text=text,
            metadata={'chapter': chapter, 'empty': not text, 'page_id': f'{document_id}:p{number:03d}'},
        ))
    return pages
