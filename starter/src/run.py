"""Day 06 pipeline: PDF -> chunks (A, B, C) -> local embeddings -> pgvector -> dev/test retrieval metrics.

Run from starter/:  .venv/bin/python -m src.run
"""
import csv
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import SentenceTransformerEmbedder
from .evaluation import _norm, evaluate_query, summarize
from .ingestion import load_pdf
from .store import PgVectorStore

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / 'data' / 'pride-and-prejudice.pdf'
SOURCE = 'data/pride-and-prejudice.pdf'
DOC_ID = 'pride-and-prejudice'
QUESTIONS = ROOT / 'benchmark' / 'questions.jsonl'
RESULTS = ROOT / 'results'
AUDIT = RESULTS / 'audit'

FIXED = dict(size=500, overlap=50)     # strategy A
PARAGRAPH_MAX = 800                    # strategy B; C = B + metadata filter
TOP_K = 3
MIN_CONTENT_PAGES = 30
COHERENCE_PAGES = range(1, 11)         # pre-registered: first 10 pages
GROUNDING_TEST_IDS = ['q16', 'q17', 'q18', 'q19', 'q20']  # pre-registered: first 5 test questions
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'


def load_questions() -> list[dict]:
    return [json.loads(l) for l in QUESTIONS.read_text(encoding='utf-8').splitlines() if l.strip()]


def chunk_stats(chunks, token_counts: list[int]) -> dict:
    lengths = [len(c.text) for c in chunks]
    return {'count': len(chunks), 'chars_total': sum(lengths),
            'len_min': min(lengths), 'len_median': statistics.median(lengths), 'len_max': max(lengths),
            'tokens_total': sum(token_counts), 'tokens_max': max(token_counts),
            'truncated_at_256_tokens': sum(t > 256 for t in token_counts)}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    # 0. The benchmark must be valid before anything runs.
    check = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'check_benchmark.py')],
                           capture_output=True, text=True, cwd=ROOT)
    if check.returncode != 0:
        raise SystemExit('benchmark check failed:\n' + check.stdout + check.stderr)

    load_dotenv(ROOT / '.env')
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise SystemExit('DATABASE_URL missing in starter/.env')
    RESULTS.mkdir(exist_ok=True)
    AUDIT.mkdir(exist_ok=True)

    # 1. Read the PDF page by page.
    pages = load_pdf(PDF, DOC_ID, SOURCE)
    content_pages = [p for p in pages if not p.metadata['empty']]
    empty_pages = [p.metadata['page_id'] for p in pages if p.metadata['empty']]
    if len(content_pages) < MIN_CONTENT_PAGES:
        raise SystemExit(f'only {len(content_pages)} content pages, need {MIN_CONTENT_PAGES}')
    page_by_id = {p.metadata['page_id']: p for p in pages}

    # 2. Chunk: A = fixed windows, B = paragraph groups, C = same chunks as B (filter applied at search).
    t0 = time.perf_counter()
    chunks_a = [c for p in content_pages for c in fixed_size(p, FIXED['size'], FIXED['overlap'])]
    chunks_b = [c for p in content_pages for c in by_paragraph(p, PARAGRAPH_MAX)]
    t_chunk = time.perf_counter() - t0
    # 3. Embed (one model, same for all strategies). Strategy C reuses B's vectors.
    t0 = time.perf_counter()
    embedder = SentenceTransformerEmbedder()
    tok_a = embedder.count_tokens([c.text for c in chunks_a])
    tok_b = embedder.count_tokens([c.text for c in chunks_b])
    stats = {'A': chunk_stats(chunks_a, tok_a), 'B': chunk_stats(chunks_b, tok_b),
             'C': chunk_stats(chunks_b, tok_b)}
    vec_a = embedder.embed_documents([c.text for c in chunks_a])
    vec_b = embedder.embed_documents([c.text for c in chunks_b])
    t_embed = time.perf_counter() - t0
    dims = embedder.dimensions

    # 4. Store in pgvector; re-ingest must not duplicate rows.
    store = PgVectorStore(database_url, dimensions=dims)
    store.create_schema()
    t0 = time.perf_counter()
    store.upsert('A', chunks_a, vec_a)
    store.upsert('B', chunks_b, vec_b)
    store.upsert('C', chunks_b, vec_b)
    t_store = time.perf_counter() - t0
    counts_first = {s: store.count(s) for s in 'ABC'}
    store.upsert('A', chunks_a, vec_a)
    store.upsert('B', chunks_b, vec_b)
    store.upsert('C', chunks_b, vec_b)
    counts_second = {s: store.count(s) for s in 'ABC'}
    expected = {'A': len(chunks_a), 'B': len(chunks_b), 'C': len(chunks_b)}
    if counts_first != expected or counts_second != expected:
        raise SystemExit(f'row count mismatch: expected {expected}, got {counts_first} then {counts_second}')

    # 5. Benchmark: evidence must be present in the pipeline's own page text.
    questions = load_questions()
    for q in questions:
        page = page_by_id[q['gold_page']]
        if _norm(q['evidence_quote']) not in _norm(page.text):
            raise SystemExit(f'{q["id"]}: evidence not in pipeline page text')

    # 6. Retrieval for every question and strategy.
    query_vectors = {q['id']: embedder.embed_query(q['question']) for q in questions}
    traces: dict[str, list[dict]] = {s: [] for s in 'ABC'}
    meta_checks: list[dict] = []
    for strategy in 'ABC':
        for q in questions:
            filters = q['filter'] if strategy == 'C' else None
            hits = store.search(strategy, query_vectors[q['id']], top_k=TOP_K, filters=filters)
            trace = evaluate_query(q, hits)
            for entry, hit in zip(trace['top3'], hits):
                entry['text'] = hit['text']
                entry['filter_used'] = filters
                entry['document_id'] = hit['document_id']
                page = page_by_id.get(f"{hit['document_id']}:p{hit['page']:03d}")
                text_ok = page is not None and _norm(hit['text']) in _norm(page.text)
                chapter_ok = page is not None and page.metadata['chapter'] == hit['chapter']
                entry['auto_text_in_page'] = text_ok
                entry['auto_chapter_match'] = chapter_ok
            traces[strategy].append(trace)
            if q['split'] == 'test':
                for rank, (entry, hit) in enumerate(zip(trace['top3'], hits), start=1):
                    meta_checks.append({'strategy': strategy, 'id': q['id'], 'rank': rank,
                                        'chunk_id': hit['id'], 'page': hit['page'], 'chapter': hit['chapter'],
                                        'source': hit['source'], 'auto_text_in_page': entry['auto_text_in_page'],
                                        'auto_chapter_match': entry['auto_chapter_match'],
                                        'filter': q['filter']['chapter'] if q['filter'] else ''})

    # 7. Summaries and files.
    summary = {s: summarize(traces[s]) for s in 'ABC'}
    filter_compliance = {}
    for s in 'C':
        rows = [t for t in traces[s] if t['filtered'] and t['split'] == 'test']
        hits_total = sum(len(t['top3']) for t in rows)
        ok_total = sum(1 for m in meta_checks if m['strategy'] == s and m['filter'] != '' and m['chapter'] == m['filter'])
        filter_compliance[s] = {'test_filtered_questions': len(rows), 'top3_chunks': hits_total,
                                'top3_matching_filter': ok_total}
    meta_auto = {s: {'records': sum(1 for m in meta_checks if m['strategy'] == s),
                     'text_in_page': sum(1 for m in meta_checks if m['strategy'] == s and m['auto_text_in_page']),
                     'chapter_match': sum(1 for m in meta_checks if m['strategy'] == s and m['auto_chapter_match'])}
                 for s in 'ABC'}

    for s in 'ABC':
        with (RESULTS / f'traces_{s}.jsonl').open('w', encoding='utf-8') as f:
            for t in traces[s]:
                f.write(json.dumps(t, ensure_ascii=False) + '\n')

    report = {
        'pdf_sha256': hashlib.sha256(PDF.read_bytes()).hexdigest(),
        'model': MODEL_NAME,
        'dimensions': dims,
        'max_seq_length': 256,
        'pages_total': len(pages),
        'pages_content': len(content_pages),
        'pages_empty': empty_pages,
        'chunking': {'A': f'fixed {FIXED["size"]}/{FIXED["overlap"]} chars, per page',
                     'B': f'paragraph groups, max {PARAGRAPH_MAX} chars, per page',
                     'C': 'same chunks as B; metadata filter {chapter} applied only to filtered questions'},
        'chunk_stats': stats,
        'rows_after_upsert': counts_first,
        'rows_after_reupsert': counts_second,
        'timing_seconds': {'chunk': round(t_chunk, 3), 'embed': round(t_embed, 3), 'store': round(t_store, 3)},
        'cost_usd': 0,
        'summary': summary,
        'filter_compliance_test': filter_compliance,
        'metadata_auto_check_test_top3': meta_auto,
    }
    (RESULTS / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    # Experiment log (header unchanged).
    log_rows = []
    for s, chunking, size, overlap, meta_desc, flt in [
        ('A', 'fixed', FIXED['size'], FIXED['overlap'], 'document_id,source,page,chapter', 'none'),
        ('B', 'paragraph', PARAGRAPH_MAX, '', 'document_id,source,page,chapter', 'none'),
        ('C', 'paragraph', PARAGRAPH_MAX, '', 'document_id,source,page,chapter', 'chapter (only when question has filter)'),
    ]:
        d, t = summary[s]['dev'], summary[s]['test']
        log_rows.append({
            'strategy': s, 'chunking': chunking, 'chunk_size': size, 'overlap': overlap,
            'metadata': meta_desc, 'filter': flt, 'vectorizer': f'{MODEL_NAME} ({dims}d, cosine)',
            'hypothesis': {'A': 'baseline', 'B': 'H1', 'C': 'H2'}[s],
            'dev_hit1': f'{d["hit1"]}/{d["n"]}', 'dev_hit3': f'{d["hit3"]}/{d["n"]}',
            'test_hit1': f'{t["hit1"]}/{t["n"]}', 'test_hit3': f'{t["hit3"]}/{t["n"]}',
            'notes': f'chunks={stats[s]["count"]}; local model, no API cost; ingest_s={round(t_chunk + t_embed + t_store, 1)}',
        })
    write_csv(ROOT.parent / 'artifacts' / 'experiment_log.csv', log_rows)

    # Audit sheets: automatic columns filled, manual label columns left blank for the author.
    # Exactly 10 positions, fixed before scoring: the first paragraph of pages 1-10.
    # For each strategy, list the chunk(s) that contain that paragraph's opening text.
    coherence = []
    for page_obj in content_pages[:10]:
        opening = _norm(next(x for x in page_obj.text.split('\n\n') if x.strip()))[:40]
        for strategy, chunks in (('A', chunks_a), ('B', chunks_b), ('C', chunks_b)):
            matches = [c for c in chunks if c.page == page_obj.page and opening in _norm(c.text)]
            if not matches:
                raise SystemExit(f'coherence position on page {page_obj.page} not found in strategy {strategy}')
            for c in matches:
                coherence.append({'position': page_obj.metadata['page_id'], 'opening': opening,
                                  'strategy': strategy, 'chunk_id': c.chunk_id, 'page': c.page,
                                  'chapter': c.metadata.get('chapter'), 'text': c.text,
                                  'label_0_2': '', 'reason': ''})
    write_csv(AUDIT / 'coherence_positions.csv', coherence)

    metadata_sheet = [dict(m, manual_correct='') for m in meta_checks]
    write_csv(AUDIT / 'metadata_test_top3.csv', metadata_sheet)

    grounding = []
    for s in 'ABC':
        for t in traces[s]:
            if t['id'] in GROUNDING_TEST_IDS:
                for entry in t['top3']:
                    grounding.append({'strategy': s, 'id': t['id'], 'rank': t['top3'].index(entry) + 1,
                                      'chunk_id': entry['chunk_id'], 'auto_is_gold': entry['is_gold'],
                                      'text': entry['text'], 'label_0_2': '', 'reason': ''})
    write_csv(AUDIT / 'grounding_test.csv', grounding)

    # Console report.
    print(f'pages: {len(pages)} (content {len(content_pages)}, empty {len(empty_pages)})')
    print(f'chunks: A={stats["A"]["count"]} B={stats["B"]["count"]} C=same as B')
    print(f'rows after upsert: {counts_first}; after re-upsert: {counts_second}')
    for s in 'ABC':
        d, t = summary[s]['dev'], summary[s]['test']
        print(f'{s}: dev Hit@1 {d["hit1"]}/{d["n"]}, Hit@3 {d["hit3"]}/{d["n"]} | '
              f'test Hit@1 {t["hit1"]}/{t["n"]}, Hit@3 {t["hit3"]}/{t["n"]}')
    print('filter compliance (test, strategy C):', filter_compliance['C'])


if __name__ == '__main__':
    main()
