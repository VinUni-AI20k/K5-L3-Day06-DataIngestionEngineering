"""Build data/manifest.json for the dental guideline PDF (reproducible).

Records: source/permission fields, physical page count, which pages count as
content and why, raw character count per page (PyMuPDF/fitz extraction), the
57-disease table of contents mapped to PDF page ranges, and a SHA-256 snapshot.

Permission fields are NOT guessed: they stay null until the author fills them.
Run from starter/:  python scripts/build_manifest.py
"""
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[2]
PDF = ROOT / "data" / "tai_lieu_huong_dan_kham_rang_ham_mat.pdf"
OUT = ROOT / "data" / "manifest.json"

SOURCE_ID = "byt-rhm-3108-2015"
PRINTED_OFFSET = 1          # printed page number = PDF page - 1 (verified on pages 8, 24, 37, 201)
CONTENT_FIRST, CONTENT_LAST = 8, 201
TOC_PAGES = (5, 6)

EXCLUDED = {
    1: "blank cover (image only, no text)",
    2: "title page",
    3: "editorial board",
    4: "editorial board",
    5: "table of contents",
    6: "table of contents",
    7: "list of abbreviations",
}
EXCLUDED.update({p: "references" for p in range(202, 209)})

# Front-matter parts that are ingested (not counted as content pages). Each part
# becomes exactly ONE chunk in every strategy (A/B/C) and is never split, so the
# table of contents (2 pages) or editorial board (2 pages) stays whole.
SPECIAL_SECTIONS = [
    {"special_id": "quyet_dinh", "section": "QUYẾT ĐỊNH BAN HÀNH", "pages": [2],
     "note": "Title page citing Quyết định 3108/QĐ-BYT ngày 28/7/2015; the PDF has no full decision text."},
    {"special_id": "ban_bien_soan", "section": "CHỦ BIÊN VÀ BAN BIÊN SOẠN", "pages": [3, 4],
     "note": "Editors, authors/reviewers and secretaries (public publication credits)."},
    {"special_id": "muc_luc", "section": "MỤC LỤC", "pages": [5, 6],
     "note": "Table of contents: 57 diseases with printed start pages."},
    {"special_id": "tu_viet_tat", "section": "DANH MỤC CÁC TỪ VIẾT TẮT", "pages": [7],
     "note": "List of abbreviations."},
]
SPECIAL_PAGES = {p: s["special_id"] for s in SPECIAL_SECTIONS for p in s["pages"]}


def fold(text: str) -> str:
    """Comparison-only form: fix the Word-2007 'ƣ' glyph, drop spaces, NFC, upper."""
    text = text.replace("ƣ", "ư").replace("Ƣ", "Ư")
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", text)).upper()


# The PDF text layer splits syllables in these five headings (e.g. 'LỢ I', 'TUYẾ N'),
# both in the TOC and on the page itself; the rendered page shows them intact.
# Corrected by hand against a render of each heading (PDF pages 37, 153, 156,
# 159, 164). parse_toc() asserts each correction only removes whitespace.
TITLE_CORRECTIONS = {
    8: "VIÊM LỢI LOÉT HOẠI TỬ CẤP TÍNH",
    43: "VIÊM TUYẾN NƯỚC BỌT MANG TAI MẠN TÍNH",
    44: "VIÊM TUYẾN NƯỚC BỌT DƯỚI HÀM DO SỎI",
    45: "U HỖN HỢP TUYẾN NƯỚC BỌT MANG TAI",
    47: "U TUYẾN NƯỚC BỌT VÒM MIỆNG",
}


def parse_toc(doc: fitz.Document) -> list[dict]:
    toc_text = "\n".join(doc[p - 1].get_text() for p in TOC_PAGES)
    # fitz puts the page number either on the title line or on the next line,
    # so match "N. TITLE <ws/newline> PAGE" across the whole TOC text.
    entries = []
    for m in re.finditer(r"(?m)^[ \t]*(\d{1,2})\.[ \t]*([^\n]+?)\s+(\d{1,3})[ \t]*$", toc_text):
        entries.append({"disease_no": int(m[1]), "title_raw": m[2].strip(),
                        "printed_start": int(m[3])})
    entries.sort(key=lambda e: e["disease_no"])
    if [e["disease_no"] for e in entries] != list(range(1, 58)):
        sys.exit(f"TOC parse failed: got {[e['disease_no'] for e in entries]}")
    for e in entries:
        e["pdf_start"] = e["printed_start"] + PRINTED_OFFSET
    for i, e in enumerate(entries):
        e["pdf_end"] = (entries[i + 1]["pdf_start"] - 1) if i + 1 < len(entries) else CONTENT_LAST
        # Title in the TOC may contain stray spaces/ƣ; keep raw and a cleaned display form.
        title = e["title_raw"].replace("ƣ", "ư").replace("Ƣ", "Ư")
        title = re.sub(r"\s+", " ", unicodedata.normalize("NFC", title)).strip(" .")
        fixed = TITLE_CORRECTIONS.get(e["disease_no"])
        if fixed is not None and fold(fixed) != fold(title):
            sys.exit(f"Title correction for {e['disease_no']} changes more than whitespace")
        e["title"] = fixed or title
        e["title_corrected_manually"] = fixed is not None
        # Verify: the heading "N. TITLE" really appears on the mapped start page.
        page_fold = fold(doc[e["pdf_start"] - 1].get_text())
        e["heading_verified"] = fold(f"{e['disease_no']}.{e['title_raw']}").rstrip(".") in page_fold
    return entries


def main() -> None:
    doc = fitz.open(PDF)
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text()
        counted = CONTENT_FIRST <= i <= CONTENT_LAST
        pages.append({
            "pdf_page": i,
            "printed_page": i - PRINTED_OFFSET if i > 1 else None,
            "counted": counted,
            "exclusion_reason": None if counted else EXCLUDED.get(i, "outside content range"),
            "ingested": counted or i in SPECIAL_PAGES,
            "special_id": SPECIAL_PAGES.get(i),
            "raw_chars": len(text),
            "raw_chars_no_ws": len(re.sub(r"\s+", "", text)),
            "has_images": bool(page.get_images()),
        })
    diseases = parse_toc(doc)
    content = [p for p in pages if p["counted"]]
    manifest = {
        "source_id": SOURCE_ID,
        "title": "Hướng dẫn chẩn đoán và điều trị một số bệnh về Răng Hàm Mặt",
        "issuer": "Bộ Y tế (Ban hành kèm theo Quyết định số 3108/QĐ-BYT ngày 28/7/2015 của Bộ trưởng Bộ Y tế)",
        "language": "vi",
        "source_path_or_url": {
            "local_path": "data/tai_lieu_huong_dan_kham_rang_ham_mat.pdf",
            "download_url": "https://benhvienlaokhoa.vn/media/lib/05-04-2021/2hdrnghmmt.pdf",
            "download_verified": "2026-11-14: HTTP 200, SHA-256 identical to the local file",
        },
        "license_or_permission": {
            "status": "no explicit license found",
            "basis": ("Publicly downloadable professional guideline issued by the Ministry of "
                      "Health (Quyết định 3108/QĐ-BYT), hosted by a third party "
                      "(benhvienlaokhoa.vn). No license or reuse terms are stated in the PDF "
                      "or on the download URL."),
            "use_in_this_lab": ("Processed for non-commercial coursework; text chunks are sent "
                                "to the OpenRouter embedding API. Contains no personal data."),
            "redistribution": ("Not verified. The PDF is not committed; obtain it from "
                               "download_url and check the SHA-256 in 'snapshot'."),
        },
        "accessed_at": "2026-11-14",
        "snapshot": {
            "sha256": hashlib.sha256(PDF.read_bytes()).hexdigest(),
            "bytes": PDF.stat().st_size,
            "pdf_metadata": {k: v for k, v in doc.metadata.items() if v},
        },
        "extraction": {"tool": "PyMuPDF (fitz)", "version": fitz.VersionBind,
                       "method": "page.get_text() default 'text' mode, per physical page"},
        "page_counting_rule": (
            f"Count PDF pages {CONTENT_FIRST}-{CONTENT_LAST} (printed pages "
            f"{CONTENT_FIRST - PRINTED_OFFSET}-{CONTENT_LAST - PRINTED_OFFSET}): the 57 disease "
            "guidelines. Exclude cover, title, editorial board, table of contents, abbreviation "
            "list and reference pages. gold_page in the benchmark uses the PDF page number."
        ),
        "physical_pages": doc.page_count,
        "content_pages": len(content),
        "content_chars_raw": sum(p["raw_chars"] for p in content),
        "known_issues": [
            "Font maps 'ư/Ư' to 'ƣ/Ƣ' (U+01A3/U+01A2) throughout; fixed during normalization.",
            "Some words are split by stray spaces (e.g. 'b ệnh'); worse in diseases 8 and 43-47.",
            "Comparison tables (e.g. PDF pages 23-26) are linearized cell by cell.",
            "Short content pages (<200 chars) are section tails, not extraction failures.",
        ],
        "ingestion_scope": (
            "Ingest the 194 content pages plus the front-matter parts in special_sections "
            "(PDF pages 2-7). Front matter is ingested but NOT counted toward the 30-page "
            "content minimum. Each special section is one unsplit chunk in every strategy, "
            "with disease=null. Page 1 (blank) and references (202-208) are not ingested."
        ),
        "special_sections": [
            {**s, "chars_raw": sum(pages[p - 1]["raw_chars"] for p in s["pages"])}
            for s in SPECIAL_SECTIONS
        ],
        "diseases": diseases,
        "pages": pages,
    }
    OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    bad = [d["disease_no"] for d in diseases if not d["heading_verified"]]
    print(f"Wrote {OUT.relative_to(ROOT)}: {doc.page_count} physical, {len(content)} content pages, "
          f"{manifest['content_chars_raw']} raw chars")
    print(f"Diseases: {len(diseases)} | heading not found on mapped page: {bad or 'none'}")
    print(f"Short content pages (<200 chars): {[p['pdf_page'] for p in content if p['raw_chars'] < 200]}")
    print(f"Empty pages: {[p['pdf_page'] for p in pages if p['raw_chars_no_ws'] == 0]}")


if __name__ == "__main__":
    main()
