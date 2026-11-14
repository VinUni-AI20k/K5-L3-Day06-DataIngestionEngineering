"""OCR the scanned page(s) of the source PDF and save raw, reproducible output.

PDF page 1 is a scan of Quyết định 3108/QĐ-BYT with no text layer. This script
renders it, runs EasyOCR (Vietnamese), groups word boxes into lines, and writes
data/ocr/page_001.json with every box, its confidence and the assembled raw text.

Post-OCR fixes live separately in data/ocr/page_001_corrections.json (reviewed
by hand against the page image) and are applied by src/ingestion.py, so the raw
OCR output is never overwritten.

Run from starter/:  python scripts/ocr_scanned_pages.py
Hardware note: a 4 GB GPU / limited RAM needs canvas_size=1700, mag_ratio=1.0.
"""
import json
import time
from pathlib import Path

import easyocr
import fitz
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PDF = ROOT / "data" / "tai_lieu_huong_dan_kham_rang_ham_mat.pdf"
OUT_DIR = ROOT / "data" / "ocr"
SCANNED_PAGES = [1]
DPI = 200
PARAMS = {"canvas_size": 1700, "mag_ratio": 1.0, "paragraph": False}
LINE_TOLERANCE_PX = 22  # boxes whose vertical centres differ less than this share a line


def group_lines(boxes: list[dict]) -> list[str]:
    rows: list[list[dict]] = []
    for b in sorted(boxes, key=lambda b: b["y_center"]):
        if rows and abs(rows[-1][-1]["y_center"] - b["y_center"]) < LINE_TOLERANCE_PX:
            rows[-1].append(b)
        else:
            rows.append([b])
    return [" ".join(b["text"] for b in sorted(r, key=lambda b: b["x_left"])) for r in rows]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(PDF)
    reader = easyocr.Reader(["vi"], verbose=False)
    for page_no in SCANNED_PAGES:
        pix = doc[page_no - 1].get_pixmap(dpi=DPI)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3]
        start = time.perf_counter()
        result = reader.readtext(img, **PARAMS)
        seconds = time.perf_counter() - start
        boxes = [{
            "text": text,
            "confidence": round(float(conf), 4),
            "x_left": int(min(p[0] for p in box)),
            "y_center": round(sum(p[1] for p in box) / 4, 1),
            "bbox": [[int(x), int(y)] for x, y in box],
        } for box, text, conf in result]
        lines = group_lines(boxes)
        out = {
            "pdf_page": page_no,
            "tool": "EasyOCR",
            "tool_version": easyocr.__version__,
            "languages": ["vi"],
            "render_dpi": DPI,
            "params": PARAMS,
            "line_tolerance_px": LINE_TOLERANCE_PX,
            "seconds": round(seconds, 1),
            "n_boxes": len(boxes),
            "mean_confidence": round(sum(b["confidence"] for b in boxes) / len(boxes), 4),
            "low_confidence_boxes": [b["text"] for b in boxes if b["confidence"] < 0.6],
            "lines_raw": lines,
            "text_raw": "\n".join(lines),
            "boxes": boxes,
        }
        path = OUT_DIR / f"page_{page_no:03d}.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"page {page_no}: {len(boxes)} boxes, {len(lines)} lines, mean conf "
              f"{out['mean_confidence']}, {seconds:.1f}s -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
