"""Read and normalize source data."""
import json
import logging
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from .models import DocumentPage


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while preserving paragraph breaks."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\x00", " ").replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = [" ".join(line.split()) for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def load_jsonl(path: Path) -> list[DocumentPage]:
    """Load one source-PDF page per JSONL record."""
    pages = []
    seen_pages = set()

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}, dòng {line_number}: JSON không hợp lệ"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"{path}, dòng {line_number}: mỗi dòng phải là JSON object"
                )

            document_id = record.get("document_id")
            source = record.get("source")
            page_number = record.get("page")
            raw_text = record.get("text")

            if not isinstance(document_id, str) or not document_id.strip():
                raise ValueError(f"{path}, dòng {line_number}: thiếu document_id")
            if not isinstance(source, str) or not source.strip():
                raise ValueError(f"{path}, dòng {line_number}: thiếu source")
            if (
                isinstance(page_number, bool)
                or not isinstance(page_number, int)
                or page_number < 1
            ):
                raise ValueError(
                    f"{path}, dòng {line_number}: page phải là số nguyên dương"
                )
            if not isinstance(raw_text, str):
                raise ValueError(f"{path}, dòng {line_number}: text phải là chuỗi")

            text = normalize_text(raw_text)
            if not text:
                raise ValueError(
                    f"{path}, dòng {line_number}: trang {page_number} không có text"
                )

            page_key = (document_id, page_number)
            if page_key in seen_pages:
                raise ValueError(
                    f"{path}, dòng {line_number}: trang nguồn bị lặp "
                    f"({document_id}, {page_number})"
                )
            seen_pages.add(page_key)

            metadata = {
                key: value
                for key, value in record.items()
                if key not in {"document_id", "source", "page", "text"}
            }

            pages.append(
                DocumentPage(
                    document_id=document_id,
                    source=source,
                    page=page_number,
                    text=text,
                    metadata=metadata,
                )
            )

    if not pages:
        raise ValueError(f"{path}: không tìm thấy bản ghi nào")

    return pages


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract text from a PDF and retain its physical page numbers."""
    reader = PdfReader(str(path), strict=False)

    if reader.is_encrypted:
        raise ValueError(f"PDF bị mã hóa: {path}")

    pages = []
    for page_number, pdf_page in enumerate(reader.pages, start=1):
        text = normalize_text(pdf_page.extract_text() or "")

        if not text:
            logging.getLogger(__name__).warning(
                "PDF page %d has no extractable text: %s",
                page_number,
                path,
            )

        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=page_number,
                text=text,
            )
        )

    return pages