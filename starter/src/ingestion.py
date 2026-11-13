"""Extract and normalize PDF pages without losing physical page numbers."""
import logging
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from .models import DocumentPage


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    pages = []
    for number, page in enumerate(PdfReader(path).pages, start=1):
        text = normalize_text(page.extract_text() or "")
        if text:
            pages.append(DocumentPage(document_id, source, number, text))
        else:
            logging.warning("%s: PDF page %d has no extractable text", path.name, number)
    return pages


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = "\n".join(" ".join(line.split()) for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", text).strip()
