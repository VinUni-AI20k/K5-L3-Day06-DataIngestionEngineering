"""Validate benchmark/questions.jsonl against the rules in ../../BENCHMARK.md and data/pages.jsonl.

Checks: required fields, unique IDs, split counts (>=15 dev, >=5 test, >=20 total),
>=3 filtered questions with >=1 in test, >=30 content pages, filter chapter equals the gold
page's chapter, and every evidence_quote appears verbatim (whitespace-normalised) in its gold page.
Run from starter/:  .venv/bin/python scripts/check_benchmark.py
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / 'benchmark' / 'questions.jsonl'
PAGES = ROOT / 'data' / 'pages.jsonl'
REQUIRED = ['id', 'split', 'type', 'question', 'expected_answer', 'gold_source_id',
            'gold_page', 'evidence_quote', 'filter']


def norm(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def main() -> int:
    pages = {}
    for line in PAGES.read_text(encoding='utf-8').splitlines():
        page = json.loads(line)
        pages[page['page_id']] = page
    content_pages = [p for p in pages.values() if not p['empty']]
    questions = [json.loads(l) for l in QUESTIONS.read_text(encoding='utf-8').splitlines() if l.strip()]

    errors, warnings = [], []
    ids = Counter(q.get('id') for q in questions)
    for qid, n in ids.items():
        if n > 1:
            errors.append(f'duplicate id {qid}')

    for q in questions:
        qid = q.get('id', '?')
        missing = [k for k in REQUIRED if k not in q]
        if missing:
            errors.append(f'{qid}: missing fields {missing}')
            continue
        if q['split'] not in ('dev', 'test'):
            errors.append(f'{qid}: split must be dev or test, got {q["split"]!r}')
        page = pages.get(q['gold_page'])
        if page is None:
            errors.append(f'{qid}: gold_page {q["gold_page"]!r} not in pages.jsonl')
            continue
        if not norm(q['evidence_quote']) or norm(q['evidence_quote']) not in page['text']:
            errors.append(f'{qid}: evidence_quote NOT found verbatim in {q["gold_page"]}')
        if q['filter'] is not None and q['filter'].get('chapter') != page['chapter']:
            errors.append(f'{qid}: filter chapter {q["filter"]} excludes gold chapter {page["chapter"]}')
        other = [p['page_id'] for p in pages.values() if p['page_id'] != q['gold_page']
                 and norm(q['evidence_quote']) in p['text']]
        if other:
            warnings.append(f'{qid}: evidence also appears in {other}')

    splits = Counter(q.get('split') for q in questions)
    filtered = [q for q in questions if q.get('filter')]
    filtered_test = [q for q in filtered if q.get('split') == 'test']
    print(f'questions: {len(questions)} | dev: {splits["dev"]} | test: {splits["test"]} | '
          f'filtered: {len(filtered)} (test: {len(filtered_test)})')
    print(f'content pages: {len(content_pages)} (empty: {len(pages) - len(content_pages)})')
    if len(content_pages) < 30:
        errors.append(f'need >=30 content pages, have {len(content_pages)}')
    if len(questions) < 20:
        errors.append(f'need >=20 questions, have {len(questions)}')
    if splits['dev'] < 15:
        errors.append(f'need >=15 dev, have {splits["dev"]}')
    if splits['test'] < 5:
        errors.append(f'need >=5 test, have {splits["test"]}')
    if len(filtered) < 3:
        errors.append(f'need >=3 filter questions, have {len(filtered)}')
    if not filtered_test:
        errors.append('need >=1 filter question in test')

    for w in warnings:
        print('WARN', w)
    for e in errors:
        print('ERROR', e)
    print('OK' if not errors else f'{len(errors)} error(s)')
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
