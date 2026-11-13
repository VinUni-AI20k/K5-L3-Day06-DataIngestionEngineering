"""Render data/corpus.jsonl into a paginated PDF, then re-extract it page by page.

Output:
  data/pride-and-prejudice.pdf  generated PDF (A4, Times 11pt, page number footer)
  data/pages.jsonl              one record per PDF page, text taken back from the PDF with pypdf

Pagination rule: each chapter starts on a new page, so every page belongs to exactly one chapter.
Run from starter/:  .venv/bin/python scripts/build_pdf.py
"""
import hashlib
import json
import re
from pathlib import Path

import pypdf
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / 'data' / 'corpus.jsonl'
PDF = ROOT / 'data' / 'pride-and-prejudice.pdf'
PAGES = ROOT / 'data' / 'pages.jsonl'
DOC_ID = 'pride-and-prejudice'

FONT, BOLD, SIZE, LEAD = 'Times-Roman', 'Times-Bold', 11, 14
W, H = A4
MARGIN = 72
TEXT_W = W - 2 * MARGIN
BOTTOM = MARGIN


def norm(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def render(units: list[dict]) -> list[dict]:
    c = canvas.Canvas(str(PDF), pagesize=A4)
    c.setTitle('Pride and Prejudice (Project Gutenberg #1342, generated pagination)')
    meta: list[dict] = []

    def begin(unit: dict) -> float:
        meta.append({'chapter': unit['chapter'], 'unit_id': unit['unit_id']})
        c.setFont(FONT, SIZE)
        return H - MARGIN

    def end() -> None:
        c.setFont(FONT, 9)
        c.drawCentredString(W / 2, 40, str(len(meta)))
        c.showPage()

    for unit in units:
        y = begin(unit)
        c.setFont(BOLD, 13)
        c.drawCentredString(W / 2, y, unit['heading'])
        y -= 28
        c.setFont(FONT, SIZE)
        for para in unit['text'].split('\n\n'):
            para = norm(para)
            if not para:
                continue
            for line in simpleSplit(para, FONT, SIZE, TEXT_W):
                if y - LEAD < BOTTOM:
                    end()
                    y = begin(unit)
                c.drawString(MARGIN, y, line)
                y -= LEAD
            y -= LEAD * 0.6
        end()
    c.save()
    return meta


def extract(meta: list[dict]) -> list[dict]:
    reader = pypdf.PdfReader(str(PDF))
    assert len(reader.pages) == len(meta), 'page count mismatch between render and PDF'
    records = []
    for i, page in enumerate(reader.pages, start=1):
        text = norm(page.extract_text() or '')
        records.append({
            'page_id': f'{DOC_ID}:p{i:03d}',
            'document_id': DOC_ID,
            'source': 'data/pride-and-prejudice.pdf',
            'page': i,
            'chapter': meta[i - 1]['chapter'],
            'unit_id': meta[i - 1]['unit_id'],
            'chars': len(text),
            'empty': len(text) == 0,
            'text': text,
        })
    return records


def main() -> None:
    units = [json.loads(line) for line in CORPUS.read_text(encoding='utf-8').splitlines() if line.strip()]
    meta = render(units)
    records = extract(meta)
    with PAGES.open('w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    empty = [r['page_id'] for r in records if r['empty']]
    sha = hashlib.sha256(PDF.read_bytes()).hexdigest()
    chars = [r['chars'] for r in records]
    print(f'pdf: {PDF.name} sha256={sha}')
    print(f'pages: {len(records)} | empty: {len(empty)} {empty[:5]} | chars min/max: {min(chars)}/{max(chars)}')
    print(f'chapters: {len({r["chapter"] for r in records})} | wrote {PAGES.name}')


if __name__ == '__main__':
    main()
