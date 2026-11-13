# Chạy lại toàn bộ bài Day 06

Mọi lệnh chạy từ thư mục `starter/`.

## 1. Môi trường

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt reportlab sentence-transformers
docker compose -p day6_student_starter up -d --wait
cp .env.example .env   # không cần API key: embedding chạy local
.venv/bin/python scripts/check_setup.py   # kỳ vọng: Setup OK
```

Model embedding (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, ~470 MB) được tải lần đầu từ Hugging Face.

## 2. Dữ liệu

- `data/source/pg1342.txt` không nằm trong git. Tải lại từ `source_path_or_url` trong `data/manifest.json` và đặt vào `data/source/`.
- `data/corpus.jsonl` là corpus đã làm sạch theo chương (61 đơn vị).
- Dựng PDF và trang: `.venv/bin/python scripts/build_pdf.py` → `data/pride-and-prejudice.pdf`, `data/pages.jsonl`. Kiểm tra sha256 với `derived_pdf.sha256` trong manifest.

## 3. Benchmark

```bash
.venv/bin/python scripts/check_benchmark.py   # kỳ vọng: OK (23 câu, 15 dev / 8 test, 5 có filter)
```

## 4. Pipeline và kết quả

```bash
.venv/bin/python -m src.run
```

Ghi ra:
- `results/summary.json`: số liệu tổng hợp, số chunk, token, tỷ lệ tuân thủ filter.
- `results/traces_{A,B,C}.jsonl`: top-3 đầy đủ cho từng câu hỏi.
- `results/audit/*.csv`: mẫu chấm coherence, metadata, grounding (cột nhãn).
- `artifacts/experiment_log.csv` (nằm ở thư mục gốc repo).

Pipeline dùng bảng `chunks` trong Postgres với khóa `(strategy, id)`. Nếu bảng được tạo bằng phiên bản cũ có khóa `id` đơn, chạy `DROP TABLE chunks;` rồi chạy lại.

## 5. Kiểm thử

```bash
.venv/bin/python -m unittest tests.test_pipeline -v
```

Các test tích hợp ghi vào strategy `TEST_ONLY` rồi xóa; không đụng A/B/C.

## 6. Cấu trúc chính

| File | Vai trò |
|---|---|
| `src/ingestion.py` | đọc PDF theo trang, chương, trang rỗng; chuẩn hóa text |
| `src/chunking.py` | A: cửa sổ cố định 500/50; B: đoạn tối đa 800 |
| `src/embeddings.py` | embedding local, không fallback |
| `src/store.py` | pgvector; upsert không trùng; filter có tham số |
| `src/evaluation.py` | Hit@1, Evidence Hit@3, tóm tắt theo dev/test |
| `src/run.py` | toàn bộ pipeline |
| `benchmark/` | câu hỏi, gold, giải thích tiêu chí |
| `report/individual/` | báo cáo cá nhân |
