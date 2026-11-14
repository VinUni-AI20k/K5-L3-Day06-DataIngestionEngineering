"""Read and normalize source data while preserving lineage and logging extraction issues."""
import re
import unicodedata
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from .models import DocumentPage

# Control characters (excluding \t, \n, \r) that indicate broken PDF figure CMap encoding
_CONTROL_CHAR_RE = re.compile(r"[\x01-\x08\x0b\x0c\x0e-\x1f]")

# Section heading detector for academic / technical documents
_HEADING_RE = re.compile(
    r"^(?:"
    r"Abstract|ABSTRACT|"
    r"Introduction|Background|Conclusion|Acknowledgements|Acknowledgments|Broader Impact|"
    r"Reproducibility Statement|Ethics Statement|REPRODUCIBILITY STATEMENT|ETHICS STATEMENT|"
    r"(?:\d+(?:\.\d+)*|[A-I](?:\.\d+)*)\s+[A-Z][A-Za-z0-9 \-:,/()]{2,75}"
    r")$"
)

# Normalize spaced-out ICLR headings (e.g., "1 I NTRODUCTION" -> "1 Introduction")
_ICLR_SPACED_HEADING_MAP = {
    "1 I NTRODUCTION": "1 Introduction",
    "2 REAC T: S YNERGIZING REASONING + AC TING": "2 ReAct: Synergizing Reasoning + Acting",
    "3 K NOWLEDGE -I NTENSIVE REASONING TASKS": "3 Knowledge-Intensive Reasoning Tasks",
    "3.1 S ETUP": "3.1 Setup",
    "3.2 M ETHODS": "3.2 Methods",
    "3.3 R ESULTS AND OBSERVATIONS": "3.3 Results and Observations",
    "4 D ECISION MAKING TASKS": "4 Decision Making Tasks",
    "5 R ELATED WORK": "5 Related Work",
    "6 C ONCLUSION": "6 Conclusion",
    "A A DDITIONAL RESULTS": "A Additional Results",
    "A.1 GPT-3 E XPERIMENTS": "A.1 GPT-3 Experiments",
    "A.2 REAC T OBTAINS UP -TO-DATE KNOWLEDGE ON HOTPOT QA": "A.2 ReAct Obtains Up-to-Date Knowledge on HotpotQA",
    "A.3 H UMAN -IN-THE -LOOP BEHAVIOR CORRECTION ON ALFWORLD": "A.3 Human-in-the-Loop Behavior Correction on ALFWorld",
    "B E XPERIMENT DETAILS": "B Experiment Details",
    "B.1 H OTPOT QA F INETUNING DETAILS": "B.1 HotpotQA Finetuning Details",
    "B.2 A LFWORLD IM-S TYLE DETAILS": "B.2 ALFWorld IM-Style Details",
    "E M ORE ANALYSIS": "E More Analysis",
    "E.1 S UCCESS AND FAILURE MODES ANALYSIS": "E.1 Success and Failure Modes Analysis",
}


def normalize_text(text: str) -> str:
    """Normalize Unicode, clean PDF boilerplate/garbled figure tokens, and preserve paragraph structure."""
    if not text:
        return ""

    # 1. Unicode NFKC (resolves typographic ligatures like ﬁ -> fi, ﬂ -> fl) followed by canonical NFC
    text = unicodedata.normalize("NFKC", text)
    text = unicodedata.normalize("NFC", text)

    # 2. Remove null bytes, soft hyphens, and normalize non-breaking spaces and line endings
    text = text.replace("\x00", "").replace("\u00a0", " ").replace("\u00ad", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Strip lines polluted by broken PDF figure CMap control characters (\x01..\x1f)
    clean_raw_lines: list[str] = []
    for raw_line in text.split("\n"):
        if _CONTROL_CHAR_RE.search(raw_line):
            continue
        clean_raw_lines.append(raw_line)
    text = "\n".join(clean_raw_lines)

    # 4. Remove recurring conference / arXiv headers and permission boilerplates
    text = re.sub(r"Published as a conference paper at ICLR 2023\s*", "", text)
    text = re.sub(
        r"Provided proper attribution is provided, Google hereby grants permission to\s*"
        r"reproduce the tables and figures in this paper solely for use in journalistic or\s*"
        r"scholarly works\.\s*",
        "",
        text,
    )
    text = re.sub(r"arXiv:\d{4}\.\d{4,5}v\d+\s*\[[^\]]+\]\s*\d+\s+[A-Za-z]+\s+\d{4}\s*", "", text)

    # 5. Rejoin hyphenated words broken across line wraps (e.g. "transduc-\ntion" -> "transduction")
    text = re.sub(r"([A-Za-z]{2,})-\s*\n\s*([a-z]{2,})", r"\1\2", text)

    # 6. Clean horizontal whitespace and standalone page number lines
    lines: list[str] = []
    for raw_line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if re.fullmatch(r"\d{1,2}", line):
            continue
        if line in _ICLR_SPACED_HEADING_MAP:
            line = _ICLR_SPACED_HEADING_MAP[line]
        lines.append(line)

    # 7. Reconstruct paragraph boundaries around headings and sentence-terminal short lines
    rebuilt: list[str] = []
    for line in lines:
        if not line:
            if rebuilt and rebuilt[-1] != "":
                rebuilt.append("")
            continue
        if _HEADING_RE.match(line) or line.startswith(("Table ", "Figure ")):
            if rebuilt and rebuilt[-1] != "":
                rebuilt.append("")
            rebuilt.append(line)
            if _HEADING_RE.match(line):
                rebuilt.append("")
        else:
            rebuilt.append(line)

    paragraphs: list[str] = []
    buf: list[str] = []
    for line in rebuilt:
        if line == "":
            if buf:
                paragraphs.append(" ".join(buf))
                buf = []
        else:
            buf.append(line)
            if len(line) < 68 and line.endswith((".", ":", "?")):
                paragraphs.append(" ".join(buf))
                buf = []
    if buf:
        paragraphs.append(" ".join(buf))

    normalized = "\n\n".join(p.strip() for p in paragraphs if p.strip())
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def extract_page_section(normalized_text: str, default_section: str = "Overview") -> str:
    """Detect the primary section heading present on a page, or keep the previous active section."""
    for para in normalized_text.split("\n\n"):
        line = para.strip()
        if _HEADING_RE.match(line):
            return line
    return default_section


def load_pdf(
    path: Path,
    document_id: str,
    source: str,
    valid_pages: set[int] | None = None,
    base_metadata: dict[str, Any] | None = None,
    error_log: list[dict[str, Any]] | None = None,
) -> list[DocumentPage]:
    """Extract each physical page, keep its page number and source, and log empty/excluded/noisy pages."""
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    reader = PdfReader(str(path))
    pages: list[DocumentPage] = []
    active_section = "Overview & Abstract"
    extra_meta = dict(base_metadata) if base_metadata else {}

    for page_idx, pdf_page in enumerate(reader.pages, start=1):
        try:
            raw_text = pdf_page.extract_text() or ""
        except Exception as exc:
            if error_log is not None:
                error_log.append({
                    "document_id": document_id,
                    "source": source,
                    "page": page_idx,
                    "event": "EXTRACTION_ERROR",
                    "detail": str(exc),
                })
            continue

        if not raw_text.strip():
            if error_log is not None:
                error_log.append({
                    "document_id": document_id,
                    "source": source,
                    "page": page_idx,
                    "event": "EMPTY_PAGE",
                    "detail": "Physical page yielded 0 extractable characters.",
                })
            continue

        if valid_pages is not None and page_idx not in valid_pages:
            if error_log is not None:
                error_log.append({
                    "document_id": document_id,
                    "source": source,
                    "page": page_idx,
                    "event": "EXCLUDED_NON_CONTENT_PAGE",
                    "detail": f"Excluded per manifest rules (raw_chars={len(raw_text)}).",
                })
            continue

        # Detect and log if embedded vector figures contained broken CMap control characters
        garbled_lines = [ln for ln in raw_text.splitlines() if _CONTROL_CHAR_RE.search(ln)]
        if garbled_lines and error_log is not None:
            error_log.append({
                "document_id": document_id,
                "source": source,
                "page": page_idx,
                "event": "GARBLED_FIGURE_CMAP_CLEANED",
                "detail": f"Stripped {len(garbled_lines)} embedded figure lines containing ASCII control bytes.",
            })

        norm_text = normalize_text(raw_text)
        if not norm_text:
            if error_log is not None:
                error_log.append({
                    "document_id": document_id,
                    "source": source,
                    "page": page_idx,
                    "event": "EMPTY_AFTER_NORMALIZATION",
                    "detail": "Page became empty after normalization.",
                })
            continue

        active_section = extract_page_section(norm_text, default_section=active_section)
        page_meta = {
            **extra_meta,
            "page_id": f"{document_id}_p{page_idx:02d}",
            "section": active_section,
            "raw_chars": len(raw_text),
            "normalized_chars": len(norm_text),
        }

        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=page_idx,
                text=norm_text,
                metadata=page_meta,
            )
        )

    return pages
