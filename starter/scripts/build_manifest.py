"""Build data/manifest.json from the raw PDFs.

Counted pages are declared in SOURCES below (physical PDF page numbers, 1-based).
Character counts are measured from the extracted, NFC-normalized text, not estimated.

Usage (from starter/):  python scripts/build_manifest.py
"""
from __future__ import annotations

import json
import unicodedata
from datetime import date
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

LICENSE_NOTE = (
    "World Bank Documents & Reports, marked 'Public Disclosure Authorized'. "
    "The PDF itself states no Creative Commons licence; used here for non-commercial "
    "educational processing only. Raw PDFs are not committed; download from the URL."
)

# part: overview = TỔNG QUAN, I = Phần I (diễn biến kinh tế), II = Phần II (chuyên đề)
SOURCES = [
    {
        "source_id": "ts-2018-06",
        "file": "ts-2018-06-vi.pdf",
        "title": "Điểm lại: Cập nhật tình hình phát triển kinh tế Việt Nam, Tháng 6/2018. "
                 "Chuyên đề: Ưu tiên cải cách nhằm giảm chi phí thương mại và tăng cường "
                 "năng lực cạnh tranh của Việt Nam",
        "url": "http://documents.worldbank.org/curated/en/119991529094262839/pdf/"
               "127168-VIETNAMESE-TakingStockVNfinal.pdf",
        "edition": "2018-06",
        "language": "vi",
        "parts": {"overview": (9, 12), "I": (14, 33), "II": (36, 58)},
        "excluded": "1-8 bìa/lời cảm ơn/mục lục/danh mục hình; 13, 35 trang ngăn cách; "
                    "34 trang trống; 59-60 tham khảo; 61-62 bìa sau",
    },
    {
        "source_id": "ts-2018-12",
        "file": "ts-2018-12-vi.pdf",
        "title": "Điểm lại: Cập nhật tình hình phát triển kinh tế Việt Nam, Tháng 12/2018. "
                 "Chuyên đề: Tạo thuận lợi thương mại bằng cách hợp lý hóa và cải thiện "
                 "tính minh bạch của các biện pháp phi thuế quan",
        "url": "http://documents.worldbank.org/curated/en/188661544471831249/pdf/"
               "132828-VIETNAMESE-REVISED-Taking-Stock-December-2018-Vietnamese.pdf",
        "edition": "2018-12",
        "language": "vi",
        "parts": {"overview": (10, 12), "I": (14, 34), "II": (36, 53)},
        "excluded": "1-9 bìa/lời cảm ơn/mục lục/từ viết tắt; 13, 35 trang ngăn cách; "
                    "54 tham khảo; 55-56 bìa sau",
    },
]


def main() -> None:
    manifest = {"accessed_at": date.today().isoformat(), "license_note": LICENSE_NOTE,
                "page_rule": "Trang PDF vật lý có nội dung phân tích; loại bìa, mục lục, "
                             "trang ngăn cách, trang trống, tài liệu tham khảo.",
                "sources": []}
    total = 0
    for s in SOURCES:
        reader = PdfReader(RAW / s["file"])
        pages = []
        for part, (start, end) in s["parts"].items():
            for p in range(start, end + 1):
                text = unicodedata.normalize("NFC", reader.pages[p - 1].extract_text() or "")
                pages.append({"page": p, "part": part, "chars": len(text.strip())})
        total += len(pages)
        manifest["sources"].append({
            "source_id": s["source_id"],
            "title": s["title"],
            "source_path_or_url": s["url"],
            "local_path": f"data/raw/{s['file']}",
            "license_or_permission": LICENSE_NOTE,
            "edition": s["edition"],
            "language": s["language"],
            "physical_pages": len(reader.pages),
            "counted_page_count": len(pages),
            "excluded_pages": s["excluded"],
            "char_count": sum(p["chars"] for p in pages),
            "pages": pages,
        })
    manifest["total_counted_pages"] = total
    out = ROOT / "data" / "manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out} — {total} counted pages")
    for s in manifest["sources"]:
        low = [p["page"] for p in s["pages"] if p["chars"] < 500]
        print(f"  {s['source_id']}: {s['counted_page_count']} pages, {s['char_count']} chars, "
              f"short pages (<500 chars): {low}")


if __name__ == "__main__":
    main()
