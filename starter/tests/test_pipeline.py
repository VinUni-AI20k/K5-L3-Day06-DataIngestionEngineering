"""Checks for the Day 06 pipeline. Run from starter/:  .venv/bin/python -m unittest tests.test_pipeline -v

Unit tests need no services. Integration tests need the Docker Postgres from starter/README.md.
"""
import json
import os
import re
import unittest
from pathlib import Path

from dotenv import load_dotenv

from src.chunking import by_paragraph, fixed_size
from src.evaluation import evaluate_query, gold_page_number, is_evidence_hit, summarize
from src.ingestion import load_pdf, normalize_text, _roman_to_int
from src.models import Chunk, DocumentPage

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / 'data' / 'pride-and-prejudice.pdf'
DOC = 'pride-and-prejudice'
DIMS = 384  # must equal the embedding model's dimension used by the real table


def page(text: str, number: int = 1, chapter: int = 1) -> DocumentPage:
    return DocumentPage(DOC, 'src.pdf', number, text, {'chapter': chapter, 'page_id': f'{DOC}:p{number:03d}', 'empty': not text})


class TestNormalize(unittest.TestCase):
    def test_keeps_paragraph_breaks_and_collapses_spaces(self):
        self.assertEqual(normalize_text('a  b\n\n\n\nc d'), 'a b\n\nc d')

    def test_drops_stray_bracket_line(self):
        self.assertEqual(normalize_text('one\n]\ntwo'), 'one\ntwo')

    def test_roman_numerals(self):
        for numeral, value in [('I', 1), ('IV', 4), ('XIV', 14), ('XL', 40), ('XLIX', 49), ('LXI', 61)]:
            self.assertEqual(_roman_to_int(numeral), value, numeral)


class TestChunking(unittest.TestCase):
    TEXT = ' '.join(f'word{i}' for i in range(600))

    def test_fixed_size_limits_and_overlap(self):
        chunks = fixed_size(page(self.TEXT), 500, 50)
        self.assertTrue(all(len(c.text) <= 500 for c in chunks))
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].text[-50:].strip() in chunks[1].text, True)

    def test_fixed_size_covers_all_words(self):
        chunks = fixed_size(page(self.TEXT), 500, 50)
        joined = ' '.join(c.text for c in chunks)
        for i in range(600):
            self.assertIn(f'word{i}', joined)

    def test_fixed_size_rejects_bad_overlap(self):
        with self.assertRaises(ValueError):
            fixed_size(page('x' * 100), 50, 50)

    def test_by_paragraph_respects_max_and_keeps_paragraphs(self):
        paras = ['First paragraph. ' * 10, 'Second paragraph. ' * 5, 'Third ' * 400]
        text = '\n\n'.join(p.strip() for p in paras)
        chunks = by_paragraph(page(text), 800)
        self.assertTrue(all(len(c.text) <= 800 for c in chunks))
        self.assertTrue(all(c.text.strip() for c in chunks))
        self.assertIn('First paragraph.', chunks[0].text)
        self.assertIn('Second paragraph.', chunks[0].text)

    def test_long_paragraph_is_not_split_after_abbreviation(self):
        # Regression: "Mr." used to end a sentence, leaving chunks that stop at "hopes of Mr."
        filler = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit sed do eiusmod. ' * 9
        paragraph = filler + 'Hopes of Mr. Bingley were lively. ' + filler
        chunks = by_paragraph(page(paragraph.strip()), 800)
        for c in chunks:
            self.assertFalse(c.text.rstrip().endswith('Mr.'), c.text[-30:])
        self.assertTrue(any('Mr. Bingley' in c.text for c in chunks))

    def test_chunks_keep_page_and_chapter(self):
        for c in by_paragraph(page('hello world\n\nagain', number=7, chapter=3), 800):
            self.assertEqual(c.page, 7)
            self.assertEqual(c.metadata['chapter'], 3)


class TestEvaluation(unittest.TestCase):
    Q = {'id': 'q', 'split': 'dev', 'gold_source_id': DOC, 'gold_page': f'{DOC}:p005',
         'evidence_quote': 'the exact sentence', 'filter': None}

    def hit(self, page_no, text):
        return {'id': f'c{page_no}', 'document_id': DOC, 'page': page_no, 'text': text, 'score': 0.5, 'chapter': 1}

    def test_gold_page_number(self):
        self.assertEqual(gold_page_number(f'{DOC}:p005'), 5)
        with self.assertRaises(ValueError):
            gold_page_number('bad')

    def test_right_page_with_evidence_is_hit(self):
        self.assertTrue(is_evidence_hit(self.Q, self.hit(5, 'xx the exact   sentence yy')))

    def test_right_page_without_full_evidence_is_miss(self):
        self.assertFalse(is_evidence_hit(self.Q, self.hit(5, 'the exact')))

    def test_wrong_page_is_miss_even_with_text(self):
        self.assertFalse(is_evidence_hit(self.Q, self.hit(6, 'the exact sentence')))

    def test_hit1_and_hit3_positions(self):
        hits = [self.hit(1, ''), self.hit(5, 'the exact sentence'), self.hit(2, '')]
        t = evaluate_query(self.Q, hits)
        self.assertFalse(t['hit1'])
        self.assertTrue(t['hit3'])
        hits = [self.hit(5, 'the exact sentence'), self.hit(1, ''), self.hit(2, '')]
        t = evaluate_query(self.Q, hits)
        self.assertTrue(t['hit1'])

    def test_hit_outside_top3_does_not_count(self):
        hits = [self.hit(1, ''), self.hit(2, ''), self.hit(3, ''), self.hit(5, 'the exact sentence')]
        self.assertFalse(evaluate_query(self.Q, hits)['hit3'])

    def test_summarize_counts(self):
        traces = [{'split': 'dev', 'hit1': True, 'hit3': True},
                  {'split': 'dev', 'hit1': False, 'hit3': True},
                  {'split': 'test', 'hit1': False, 'hit3': False}]
        s = summarize(traces)
        self.assertEqual((s['dev']['n'], s['dev']['hit1'], s['dev']['hit3']), (2, 1, 2))
        self.assertEqual((s['test']['n'], s['test']['hit3']), (1, 0))
        self.assertEqual(s['all']['n'], 3)


class TestPdfIngestion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pages = load_pdf(PDF, DOC, 'data/pride-and-prejudice.pdf')

    def test_page_count_and_no_empty_pages(self):
        self.assertEqual(len(self.pages), 216)
        self.assertEqual(sum(p.metadata['empty'] for p in self.pages), 0)

    def test_every_page_has_chapter(self):
        self.assertTrue(all(isinstance(p.metadata['chapter'], int) for p in self.pages))

    def test_chapters_match_pages_jsonl(self):
        reference = {json.loads(l)['page_id']: json.loads(l)['chapter']
                     for l in (ROOT / 'data' / 'pages.jsonl').read_text(encoding='utf-8').splitlines()}
        mismatched = [p.metadata['page_id'] for p in self.pages
                      if reference[p.metadata['page_id']] != p.metadata['chapter']]
        self.assertEqual(mismatched, [])

    def test_no_page_number_footer_left_in_text(self):
        for p in self.pages[:20]:
            last = p.text.split('\n')[-1]
            self.assertFalse(last.strip().isdigit(), p.metadata['page_id'])


class TestBenchmark(unittest.TestCase):
    def setUp(self):
        self.questions = [json.loads(l) for l in (ROOT / 'benchmark' / 'questions.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
        self.pages = {json.loads(l)['page_id']: json.loads(l)
                      for l in (ROOT / 'data' / 'pages.jsonl').read_text(encoding='utf-8').splitlines()}

    def test_counts_and_splits(self):
        dev = [q for q in self.questions if q['split'] == 'dev']
        test = [q for q in self.questions if q['split'] == 'test']
        self.assertGreaterEqual(len(self.questions), 20)
        self.assertGreaterEqual(len(dev), 15)
        self.assertGreaterEqual(len(test), 5)
        self.assertGreaterEqual(sum(1 for q in self.questions if q['filter']), 3)
        self.assertGreaterEqual(sum(1 for q in test if q['filter']), 1)

    def test_evidence_verbatim_in_gold_page(self):
        for q in self.questions:
            text = re.sub(r'\s+', ' ', self.pages[q['gold_page']]['text'])
            self.assertIn(re.sub(r'\s+', ' ', q['evidence_quote']), text, q['id'])

    def test_filter_contains_gold_chapter(self):
        for q in self.questions:
            if q['filter']:
                self.assertEqual(q['filter']['chapter'], self.pages[q['gold_page']]['chapter'], q['id'])

    def test_split_fixed_before_run_hypotheses_file_exists(self):
        self.assertTrue((ROOT / 'results' / 'hypotheses.md').exists())


@unittest.skipUnless(os.environ.get('RUN_DB_TESTS') or (ROOT / '.env').exists(), 'needs Postgres')
class TestStoreIntegration(unittest.TestCase):
    """Uses a separate table-free strategy name so it never touches A/B/C rows."""

    STRATEGY = 'TEST_ONLY'

    @classmethod
    def setUpClass(cls):
        load_dotenv(ROOT / '.env')
        from src.store import PgVectorStore
        cls.store = PgVectorStore(os.environ['DATABASE_URL'], dimensions=DIMS)
        cls.store.create_schema()

    def tearDown(self):
        import psycopg
        with psycopg.connect(os.environ['DATABASE_URL'], autocommit=True) as conn:
            conn.execute('DELETE FROM chunks WHERE strategy = %s', (self.STRATEGY,))

    def chunk(self, n, chapter):
        return Chunk(f'{DOC}:t{n}', DOC, 'src.pdf', n, f'text {n}', {'chapter': chapter})

    @staticmethod
    def vec(*head):
        return [float(x) for x in head] + [0.0] * (DIMS - len(head))

    def test_upsert_idempotent_and_removes_obsolete(self):
        chunks = [self.chunk(1, 1), self.chunk(2, 2)]
        vecs = [self.vec(1), self.vec(0, 1)]
        self.store.upsert(self.STRATEGY, chunks, vecs)
        self.store.upsert(self.STRATEGY, chunks, vecs)
        self.assertEqual(self.store.count(self.STRATEGY), 2)
        self.store.upsert(self.STRATEGY, [self.chunk(1, 1)], [self.vec(1)])
        self.assertEqual(self.store.count(self.STRATEGY), 1)

    def test_search_orders_by_cosine_and_filters(self):
        self.store.upsert(self.STRATEGY, [self.chunk(1, 1), self.chunk(2, 2)],
                          [self.vec(1), self.vec(0.9, 0.1)])
        top = self.store.search(self.STRATEGY, self.vec(1), top_k=2)
        self.assertEqual([h['page'] for h in top], [1, 2])
        only2 = self.store.search(self.STRATEGY, self.vec(1), top_k=2, filters={'chapter': 2})
        self.assertEqual([h['page'] for h in only2], [2])

    def test_rejects_unknown_filter_key_and_bad_dimensions(self):
        with self.assertRaises(ValueError):
            self.store.search(self.STRATEGY, self.vec(1), filters={'chapter; DROP TABLE chunks': 1})
        with self.assertRaises(ValueError):
            self.store.upsert(self.STRATEGY, [self.chunk(1, 1)], [[1.0, 0.0]])


@unittest.skipUnless((ROOT / 'results' / 'summary.json').exists(), 'run the pipeline first')
class TestResultsConsistency(unittest.TestCase):
    def test_summary_matches_traces(self):
        summary = json.loads((ROOT / 'results' / 'summary.json').read_text(encoding='utf-8'))
        for s in 'ABC':
            traces = [json.loads(l) for l in (ROOT / 'results' / f'traces_{s}.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(summarize(traces)['all']['hit3'], summary['summary'][s]['all']['hit3'], s)
            self.assertEqual(summarize(traces)['all']['hit1'], summary['summary'][s]['all']['hit1'], s)

    def test_traces_have_top3_and_text(self):
        for s in 'ABC':
            for line in (ROOT / 'results' / f'traces_{s}.jsonl').read_text(encoding='utf-8').splitlines():
                t = json.loads(line)
                self.assertEqual(len(t['top3']), 3)
                self.assertTrue(all('text' in e for e in t['top3']))


if __name__ == '__main__':
    unittest.main()
