# Pipeline · Qdrant + Gemini Embedding 2

Pipeline cá nhân cho lab Day 06. Bản starter gốc dùng PostgreSQL + pgvector; bài này **thay bằng Qdrant tự host** và **`google/gemini-embedding-2` qua OpenRouter** (GUIDE cho phép vector store tương đương). `docker-compose.yml` và `init.sql` của starter không còn dùng.

| Thành phần | Lựa chọn |
|---|---|
| Corpus | *Hướng dẫn chẩn đoán và điều trị một số bệnh về Răng Hàm Mặt* (Bộ Y tế, QĐ 3108/QĐ-BYT, 2015), 208 trang PDF, 194 trang nội dung |
| Đọc PDF | PyMuPDF (`fitz`) 1.26.4; trang 1 (bản scan) dùng EasyOCR 1.7.2 |
| Embedding | `google/gemini-embedding-2` qua OpenRouter, 3072 chiều |
| Vector store | Qdrant server 1.19.1 (tự host, HTTPS), `qdrant-client` 1.15.1, metric cosine |
| Python | 3.11.9, môi trường mặc định (không dùng venv) |

## Cấu trúc

```text
data/
├── manifest.json                 # nguồn, quyền, quy tắc đếm trang, 57 bệnh → trang PDF
├── ocr/page_001.json             # kết quả OCR thô trang Quyết định (box + độ tin cậy)
├── ocr/page_001_corrections.json # 26 sửa lỗi OCR đã đối chiếu ảnh scan
└── tai_lieu_huong_dan_kham_rang_ham_mat.pdf   # KHÔNG commit, tải theo manifest
starter/
├── scripts/check_setup.py        # kiểm tra Qdrant + 1 lần gọi embedding thật
├── scripts/build_manifest.py     # sinh data/manifest.json
├── scripts/ocr_scanned_pages.py  # OCR trang scan (chỉ cần chạy lại khi đổi OCR)
├── scripts/verify_benchmark.py   # kiểm chứng evidence + ràng buộc benchmark
├── src/ingestion.py              # ✅ đọc, chuẩn hóa, gắn bệnh/mục/tiểu mục
├── src/chunking.py               # ✅ A fixed-size, B theo mục (C dùng lại B)
├── src/embeddings.py             # TODO: OpenRouter adapter
├── src/store.py                  # TODO: Qdrant upsert/search/filter
├── src/evaluation.py             # TODO: Hit@1, Evidence Hit@3, trace
├── src/run.py                    # TODO: điều phối A/B/C
└── results/
    ├── ingestion_log.json        # thống kê + lỗi + mọi chỗ ghép chữ
    ├── chunk_stats.json          # số chunk, phân bố độ dài, độ phủ evidence trên dev
    ├── pages_normalized.jsonl    # text đã chuẩn hóa (không commit, sinh lại được)
    └── chunks_A.jsonl, chunks_B.jsonl   # chunk + metadata (không commit, sinh lại được)
```

## 1. Setup

Chạy từ thư mục `starter/`:

```bash
python -m pip install -r requirements.txt      # easyocr/torch chỉ cần nếu chạy lại OCR
# Windows PowerShell: Copy-Item .env.example .env ; macOS/Linux: cp .env.example .env
# điền QDRANT_URL, QDRANT_API_KEY, OPENROUTER_API_KEY vào .env (không commit .env)
python scripts/check_setup.py
```

Kỳ vọng: dòng `Qdrant: ... | server 1.19.1`, dòng `Embedding: google/gemini-embedding-2 ... | dim 3072`, rồi `Setup OK`. Một lần chạy tốn khoảng USD 0.000001. Client 1.15.1 với server 1.19.1 sinh cảnh báo lệch phiên bản; các lệnh cơ bản vẫn chạy.

> `qdrant-client` mặc định dùng cổng 6333 khi URL không ghi cổng. Server sau HTTPS reverse proxy cần `port=443`; `check_setup.py` tự suy ra từ URL.

## 2. Dữ liệu

Tải PDF về `data/tai_lieu_huong_dan_kham_rang_ham_mat.pdf` từ `download_url` trong [manifest](../data/manifest.json) và kiểm tra SHA-256 khớp `snapshot.sha256`. File không được commit vì chưa xác minh quyền phát tán lại.

```bash
python scripts/build_manifest.py   # sinh lại manifest (không cần mạng)
python scripts/ocr_scanned_pages.py   # tùy chọn: OCR lại trang 1 (~20 s, GPU 4 GB hoặc CPU)
```

**Quy tắc đếm trang:** tính trang PDF 8–201 (57 bệnh, số in 7–200). Không tính trang 1–7 (Quyết định, trang tên sách, ban biên soạn, mục lục, từ viết tắt) và 202–208 (tài liệu tham khảo). `gold_page` trong benchmark dùng **số trang PDF**, tức số in + 1.

**Phạm vi ingest:** 194 trang nội dung + 7 trang phần mở đầu (trang 1–7, gồm 5 phần). Mỗi phần mở đầu là **một chunk nguyên vẹn ở mọi cấu hình**: `quyet_dinh` (tr.1), `trang_ten_sach` (tr.2), `ban_bien_soan` (tr.3–4), `muc_luc` (tr.5–6), `tu_viet_tat` (tr.7). Phần mở đầu được ingest nhưng **không tính** vào 30 trang tối thiểu.

## 3. Đọc và chuẩn hóa (bước 2)

```bash
python -m src.ingestion
```

Kết quả lần chạy 2026-11-14: 201 trang ingest, 194 trang tính, 205.742 ký tự nội dung sau chuẩn hóa, 0 lỗi trích xuất, 339 chỗ ghép chữ trên 22 trang, đủ 57/57 bệnh có tiêu đề mục.

| Bước | Vấn đề trong PDF | Cách xử lý |
|---|---|---|
| Ký tự | Font Word 2007 mã hóa `ư/Ư` thành `ƣ/Ƣ` (≈2.800 lần) | Thay ký tự, rồi Unicode NFC |
| Khoảng trắng | Dòng trống, khoảng trắng kép, NBSP | Gộp khoảng trắng, bỏ dòng trống, **giữ ngắt dòng** để nhận tiêu đề |
| Chữ bị tách | Lớp text tách âm tiết: `tuyế n`, `b ệnh` (nặng ở bệnh 8, 43–47) | Ghép theo **từ vựng của chính corpus** (xem dưới) |
| Số trang in | Dòng đầu mỗi trang là số in | Bỏ nếu khớp `printed_page`; cả 194 trang đều khớp |
| Tiêu đề bệnh | 5 tên bị tách chữ trong cả mục lục lẫn trang | Thay dòng tiêu đề bằng tên trong manifest khi **chỉ khác khoảng trắng** |
| Trang scan | Trang 1 không có lớp text | EasyOCR + 26 sửa lỗi đã đối chiếu ảnh, mỗi sửa lỗi phải khớp đúng 1 lần |

**Quy tắc ghép chữ:** ghép hai token `a b` khi `ab` xuất hiện ≥3 lần trong corpus và một trong hai: (1) `a` hoặc `b` chưa từng đứng riêng, hoặc (2) `b` là âm cuối tiếng Việt (`c ch m n ng nh p t i y o u`) và `ab` phổ biến hơn `a` trên 1,5 lần. Ngưỡng 1,5 được chọn sau khi liệt kê đủ 32 chỗ ghép thêm so với ngưỡng 3 và kiểm tra tay đều đúng. Quy tắc giữ nguyên `là u` (khối u) vì `làu` không tồn tại. Mọi chỗ ghép được ghi trong `results/ingestion_log.json` để kiểm tra lại.

**Metadata mỗi trang** (chunk kế thừa ở bước 5): `document_id`, `source`, `page` (PDF), `printed_page`, `disease_no`, `disease`, `special_id`, `section_at_start`, `subsection_at_start`, và `headings` = danh sách tiêu đề trong trang kèm vị trí ký tự (`disease` / `section` / `subsection`). Hàm `section_at(page, offset)` trả về mục và tiểu mục đang có hiệu lực tại một vị trí, nên chunk vắt qua ranh giới mục vẫn được gán đúng.

**Hạn chế đã biết** (ghi trong log):
- Bệnh 25 (tr.105) **thiếu tiêu đề "III. CHẨN ĐOÁN" trong bản gốc** (đã kiểm tra ảnh render). Phần chẩn đoán mang `section = II. NGUYÊN NHÂN`; không tự thêm cấu trúc.
- Bảng so sánh (trang 21, 24, 26, 42, 172, 176, 180; trang 6 là mục lục) bị trải thành từng ô một dòng.
- Vẫn còn một số chữ bị tách mà quy tắc không ghép được an toàn, ví dụ `đầ y`, `là m`, `mụ n`, `bù ng` (trang 30–39) và `HỖ N` trong mục lục (trang 6). Ở `là m` và `đầ y`, mảnh đầu cũng là một từ có nghĩa, nên ghép sẽ rủi ro.
- 7 trang nội dung ngắn (<200 ký tự) là phần cuối của một mục, không phải lỗi trích xuất.

## 4. Benchmark (bước 3)

```bash
python scripts/verify_benchmark.py   # cần chạy python -m src.ingestion trước
```

24 câu trong [benchmark/questions.json](../benchmark/questions.json): 18 dev / 6 test, tách theo bệnh, 5 câu có filter `disease_no` (2 câu trong test). Script kiểm tra evidence nằm đúng trang gold sau chuẩn hóa, cảnh báo nếu evidence xuất hiện ở trang khác, và kiểm tra các ràng buộc số lượng cùng việc không rò rỉ bệnh giữa dev và test. Thiết kế và các quyết định về gold: [benchmark/README.md](../benchmark/README.md). Benchmark được commit **trước** khi chạy bất kỳ retrieval nào.

## 5. Cấu hình và chunking (bước 4–5)

Tiêu chí chọn cấu hình được chốt **trước** khi chạy: [artifacts/selection_criteria.md](../artifacts/selection_criteria.md) (dev Evidence Hit@3 → dev Hit@1 nếu chênh ≤1 câu → ít chunk hơn). Tham số và giả thuyết: [artifacts/experiment_log.csv](../artifacts/experiment_log.csv).

```bash
python -m src.chunking   # cần results/pages_normalized.jsonl
```

| | A | B | C |
|---|---|---|---|
| Cách cắt | Cửa sổ 500 ký tự, overlap 50, trong từng trang, không căn theo từ | Cắt tại tiêu đề mục/tiểu mục; gộp mục <150 ký tự vào mục sau; mục >1000 ký tự tách ở ý lớn (`-`, `a.`, `1.`), rồi ý con (`+`), rồi cuối câu | Giống B |
| Truy xuất | Không lọc | Không lọc | Lọc `disease_no` khi câu hỏi có filter |
| Số chunk | 547 | 526 | (dùng chung với B) |
| Độ dài trung vị / trung bình | 499 / 419 | 304 / 402 | |
| Chunk vắt qua 2 trang | 2 (chỉ phần mở đầu) | 121 | |
| Chunk chạm >1 mục | 228 | 69 | |
| Độ phủ evidence dev | 17/18 (d04 bị cắt đôi) | 18/18 | |

Cả hai cấu hình đều giữ 5 phần mở đầu nguyên vẹn (mỗi phần 1 chunk). "Độ phủ evidence" là cận trên của Hit@k: có chunk nào trên trang gold chứa **trọn** evidence hay không. Chỉ số này chỉ tính trên **dev**, để không tinh chỉnh chunking theo test.

**Một thay đổi dựa trên dev:** bản đầu của B cắt mục dài ở bất kỳ dòng nào, nên tách "- Gãy mỏm khẩu cái…" khỏi các ý con "+" của nó (d10 không thể hit). Bản hiện tại ưu tiên cắt ở ý lớn trước, và độ phủ dev tăng từ 17 lên 18/18.

**Metadata mỗi chunk:** `document_id`, `source`, `page` (trang đầu), `page_end`, `pages`, `printed_page`, `disease_no`, `disease`, `special_id`, `section` và `subsection` tại đầu chunk, `sections` (mọi mục chunk chạm tới), `char_start`, `chars`, `strategy`, `chunk_index`. `chunk_id` là UUID5 tính từ (cấu hình, tài liệu, trang:vị trí bắt đầu, hash nội dung): chạy lại cho ra đúng ID cũ, còn nội dung đổi thì ID đổi.

**Ảnh hưởng tới so sánh:** độ dài trung bình của A và B gần bằng nhau (419 so với 402), nên khác biệt A–B chủ yếu do **ranh giới cắt**. Phân bố độ dài thì khác: B có nhiều chunk ngắn hơn và vài chunk gần 1000 ký tự.

## 6. Các bước tiếp theo

Embedding, Qdrant và đánh giá: xem [GUIDE.md](../GUIDE.md).
