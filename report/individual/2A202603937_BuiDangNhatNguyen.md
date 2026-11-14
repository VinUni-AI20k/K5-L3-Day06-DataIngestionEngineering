# Báo cáo cá nhân — Day 6: Data Ingestion Engineering

- **Họ và tên:** Bùi Đặng Nhật Nguyên
- **MSSV:** 2A202603937
- **Lớp:** L3B

---

## Dữ liệu và pipeline

### 1. Chủ đề, nguồn/quyền, số trang và quy tắc tính trang (Dataset Overview)
- **Chủ đề Corpus**: *Foundational Architectures for Modern AI Systems: Transformer, Retrieval-Augmented Generation (RAG), and ReAct Agents*.
- **Danh sách tài liệu và quyền sử dụng** (chi tiết trong [data/manifest.json](../../data/manifest.json) và [data/manifest.csv](../../data/manifest.csv)):
  1. `vaswani-2017-transformer`: *Attention Is All You Need* (Vaswani et al., 2017) — Nguồn: `https://arxiv.org/pdf/1706.03762.pdf` (`arXiv:1706.03762v7`). Giấy phép: *arXiv.org perpetual, non-exclusive license to distribute (Open Access)*. Số trang vật lý: **15** -> Số trang nội dung hợp lệ: **10 trang** (trang `1..10`; loại trang `11..12` chỉ chứa References và trang `13..15` chỉ chứa hình vẽ Attention Visualizations rời rạc). Số ký tự sau chuẩn hóa: **30.710 ký tự**.
  2. `lewis-2020-rag`: *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (Lewis et al., 2020) — Nguồn: `https://arxiv.org/pdf/2005.11401.pdf` (`arXiv:2005.11401v4`). Giấy phép: *arXiv.org perpetual, non-exclusive license to distribute (Open Access)*. Số trang vật lý: **19** -> Số trang nội dung hợp lệ: **13 trang** (trang `1..10` nội dung chính và trang `17..19` Appendices A–I mô tả chi tiết kỹ thuật; loại trang `11..16` chỉ chứa References). Số ký tự sau chuẩn hóa: **48.096 ký tự**.
  3. `yao-2022-react`: *ReAct: Synergizing Reasoning and Acting in Language Models* (Yao et al., 2022) — Nguồn: `https://arxiv.org/pdf/2210.03629.pdf` (`arXiv:2210.03629v3`). Giấy phép: *Creative Commons Attribution 4.0 International (CC-BY 4.0)*. Số trang vật lý: **33** -> Số trang nội dung hợp lệ: **14 trang** (trang `1..10` nội dung chính, trang `14..15` Appendices A–B về thực nghiệm bổ sung & fine-tuning, trang `32..33` Appendix E về phân tích lỗi; loại trang `11..13` chỉ chứa References và trang `16..31` Appendices C–D chỉ chứa verbatim few-shot prompt/trajectory dumps). Số ký tự sau chuẩn hóa: **57.798 ký tự**.
- **Tổng quy mô Corpus**: **67 trang vật lý** -> **37 trang nội dung hợp lệ** (vượt ngưỡng tối thiểu 30 trang theo `GUIDE.md`), tổng cộng **136.604 ký tự** sau chuẩn hóa.
- **Quy tắc tính trang (`page_counting_rule`)**: Chỉ tính các trang PDF vật lý có chứa nội dung văn bản kỹ thuật, công thức toán học, mô tả kiến trúc hoặc phân tích thực nghiệm. Loại bỏ hoàn toàn các trang chỉ chứa danh mục tài liệu tham khảo (`References`), trang chỉ có hình heatmap rời rạc không có văn bản luận giải, và các trang chỉ dump log prompt mẫu để tránh hiện tượng nhiễu từ khóa (keyword magnet) khi truy xuất.

### 2. Kiến trúc Pipeline, Chuẩn hóa, Lỗi và Giới hạn (Pipeline Architecture & Data Processing Strategy)
- **Kiến trúc Pipeline** ([starter/src/](../../starter/src/)):
  `PDF Files (data/raw/) -> load_pdf() & Error Logging -> normalize_text() -> Chunking (fixed_size vs by_paragraph) -> MyEmbedder (google/embeddinggemma-2) -> Persistent Vector Store (Idempotent Upsert) -> Top-3 Retrieval (with/without Metadata Filter) -> Evaluation & Audit`.
- **Quy trình chuẩn hóa văn bản (`normalize_text` trong [starter/src/ingestion.py](../../starter/src/ingestion.py))**:
  1. Chuẩn hóa Unicode `NFKC` tiếp nối bởi `NFC` nhằm phân rã các ký tự ligature đặc thù của LaTeX PDF (`ﬁ` -> `fi`, `ﬂ` -> `fl`, `ﬀ` -> `ff`) về ký tự chuẩn, đồng thời bảo toàn dấu tiếng Việt/ký tự chuẩn.
  2. Loại bỏ ký tự null (`\x00`), soft-hyphen (`\u00ad`), khoảng trắng không ngắt (`\u00a0` -> `" "`).
  3. Phát hiện và lọc bỏ các dòng văn bản rác sinh ra từ lỗi bảng mã font CMap (`[\x01-\x08\x0b\x0c\x0e-\x1f]`) trong các hình vẽ vector nhúng của bài báo `yao-2022-react`.
  4. Xóa các dòng header/watermark lặp lại ở đầu trang (`Published as a conference paper at ICLR 2023`, `arXiv:...`, câu bản quyền của Google) và số trang đứng độc lập.
  5. Nối lại các từ tiếng Anh bị ngắt dòng bằng dấu gạch ngang (`([A-Za-z]{2,})-\s*\n\s*([a-z]{2,})` -> `\1\2`).
  6. Chuẩn hóa các tiêu đề mục bị giãn chữ của style ICLR (`1 I NTRODUCTION` -> `1 Introduction`) và khôi phục dấu ngắt đoạn `\n\n` quanh các tiêu đề mục.
- **Ghi nhận lỗi và trang bị loại ([results/ingestion_errors.log](../../results/ingestion_errors.log) & [results/ingestion_errors.json](../../results/ingestion_errors.json))**:
  - Ghi nhận tổng cộng **33 sự kiện** trong quá trình đọc 67 trang vật lý: **30 sự kiện `EXCLUDED_NON_CONTENT_PAGE`** (các trang References, Visualizations, Prompt dumps bị loại theo manifest) và **3 sự kiện `GARBLED_FIGURE_CMAP_CLEANED`** tại trang `2` (156 dòng), trang `14` (96 dòng) và trang `15` (109 dòng) của `yao-2022-react` do hình vẽ nhúng Figure 1, 4, 5 trong PDF gốc bị lệch bảng mã CMap +3 ASCII.
  - Không có trang nào bị rỗng hoàn toàn (`EMPTY_PAGE = 0`).

### 3. Embedding Model, Vector Store, Metric, Lệnh chạy và Chi phí
- **Embedding Model ([starter/src/embeddings.py](../../starter/src/embeddings.py))**:
  - Provider & SDK: Local HuggingFace qua `sentence-transformers==6.3.0` (`torch==2.15.0`, `pypdf==6.11.0`, `Python 3.13.3`).
  - Tên model: `google/embeddinggemma-2` (phiên bản local cache, `768` chiều).
  - Xử lý vector: Chuẩn hóa độ dài $L_2$ (`normalize_embeddings=True`) cho cả document chunks và queries. Không sử dụng bất kỳ vector mock/random nào.
- **Vector Store ([starter/src/store.py](../../starter/src/store.py))**:
  - Mã nguồn hiện thực đầy đủ cả 2 adapter: `PgVectorStore` (`psycopg[binary]`, PostgreSQL 16 + `pgvector`, toán tử `<=>` HNSW cosine) và `QdrantPersistentStore` (`qdrant-client==1.20.0`).
  - Trong phiên chạy thực nghiệm trên máy local (do Docker Desktop daemon không bật), pipeline sử dụng **`Qdrant Local Persistent`** (`qdrant-client==1.20.0`), lưu trữ bền vững trên đĩa tại `results/qdrant_db` (collection `day06_document_chunks`).
  - Độ đo tương đồng: **Cosine Similarity** (`Distance.COSINE`, tìm kiếm chính xác exact search kết hợp lọc cấu trúc `models.Filter` / `FieldCondition`).
  - **Kiểm chứng nạp lại không nhân đôi (Idempotency Verification)**: Mỗi chunk có ID xác định (`{doc_id}_p{page:02d}_{fix|para}_{idx:02d}`) được ánh xạ sang `UUIDv5(NAMESPACE_DNS, f"{strategy}::{chunk_id}")` kèm bước dọn dẹp các chunk cũ của cùng `(strategy, document_id)` trước khi `upsert`. Kiểm tra thực tế khi chạy nạp lại Strategy B lần 2: số bản ghi trước khi nạp lại = **232**, sau khi nạp lại = **232** (`idempotent_passed = True`).
- **Chi phí và Thời gian**:
  - Chi phí API: **`$0.00`** (chạy local 100%).
  - Thời gian load model: `10.31s`; đọc và chuẩn hóa 37 trang PDF: `2.04s`; tạo embedding Strategy A (300 chunks): `168.25s`; tạo embedding Strategy B/C (232 chunks): `144.67s`; truy vấn 20 câu hỏi: `0.16s–0.19s` mỗi chiến lược.
- **Lệnh chạy tái lập**:
  ```powershell
  cd starter
  E:\Python_download\python.exe -m src.run
  ```

### 4. Metadata bắt buộc, trường lọc bổ sung, nguồn giá trị và lý do chọn
- **Metadata bắt buộc trên mọi chunk**:
  - `document_id` (chuỗi định danh tài liệu: `vaswani-2017-transformer`, `lewis-2020-rag`, `yao-2022-react`).
  - `source` (đường dẫn file nguồn: `data/raw/*.pdf`).
  - `page` (số nguyên chỉ số trang vật lý `1..33`).
- **Trường lọc và truy vết bổ sung**:
  1. `category` (`string`: `"architecture"`, `"rag"`, `"agent"`): Lấy từ phân loại miền kỹ thuật của tài liệu trong `data/manifest.json`. Giúp lọc phạm vi kiến trúc khi câu hỏi đề cập đến các benchmark hoặc khái niệm dùng chung giữa nhiều bài báo.
  2. `year` (`integer`: `2017`, `2020`, `2022`): Lấy từ năm công bố chính thức trên arXiv. Giúp phân biệt mốc thời gian công bố mô hình.
  3. `section` (`string`): Trích xuất tự động từ dòng tiêu đề mục trong văn bản (`1 Introduction`, `3 Model Architecture`, `4 Experiments`...) để hỗ trợ người đọc kiểm tra ngữ cảnh.

---

## Cấu hình và benchmark

### 1. Ba cấu hình A / B / C (Tham số cụ thể & tính kiểm soát biến số)
- **Configuration A (`fixed_size`, không filter)**: Chia mỗi trang thành các cửa sổ trượt cố định `size = 500` ký tự, `overlap = 50` ký tự (bước nhảy 450 ký tự). Lưu đầy đủ metadata nhưng **tắt metadata filter** khi truy xuất (`filters = None`). Sinh ra **300 chunks** (độ dài trung bình `471.8` ký tự).
- **Configuration B (`by_paragraph`, không filter)**: Chia theo cấu trúc đoạn văn (`\n\n`) và tiêu đề mục (`_HEADING_RE`), gom các đoạn liên tiếp thuộc cùng một mục đến giới hạn `max_size = 800` ký tự (nếu một đoạn đơn lẻ vượt `800` ký tự thì tách theo ranh giới câu). Lưu cùng schema metadata như A và **tắt metadata filter** khi truy xuất (`filters = None`). Sinh ra **232 chunks** (độ dài trung bình `553.6` ký tự). *(So sánh A vs B chỉ thay đổi duy nhất thuật toán chunking).*
- **Configuration C (`by_paragraph` + Metadata Filter)**: Dùng **chính xác cùng 232 chunks và cùng vector embedding của Configuration B**, chỉ khác duy nhất ở bước truy xuất: **bật bộ lọc metadata (`category`, `year`)** đối với các câu hỏi có trường `filter` khác `null` (với các câu hỏi không có `filter`, C chạy không lọc giống B). *(So sánh B vs C chỉ thay đổi duy nhất biến số Metadata Filtering).*

### 2. Bộ câu hỏi Benchmark (≥20 câu) và quy trình kiểm chứng Evidence
- Chi tiết lưu tại [benchmark/questions.json](../../benchmark/questions.json):
  - **Tổng số câu**: **20 câu hỏi** được viết và khóa trước khi chạy thử nghiệm, chia thành **15 câu `dev`** (`q01`–`q15`) và **5 câu `test`** (`q16`–`q20`).
  - **Độ đa dạng loại câu hỏi**: Gồm **9 câu trực tiếp (`truc_tiep`)**, **4 câu diễn đạt lại (`dien_dat_lai`)** và **7 câu điều kiện/ngoại lệ (`dieu_kien_ngoai_le`)**.
  - **Câu hỏi kiểm tra Metadata Filter**: Có **5 câu đi kèm `filter`** (vượt yêu cầu tối thiểu 3 câu), gồm **3 câu ở `dev`** (`q13`, `q14`, `q15`) và **2 câu ở `test`** (`q19`, `q20`). Các câu này hỏi về các tác vụ/khái niệm giao thoa giữa nhiều bài báo (như kết quả trên tập `FEVER`, `HotpotQA`, cấu hình `beam search` và `checkpoint averaging`), và giá trị filter chỉ dùng `{"category": ..., "year": ...}` giả định từ ngữ cảnh người dùng, tuyệt đối không đưa `gold_source_id` hay đáp án vào câu hỏi/filter.
  - **Kiểm chứng Evidence**: Toàn bộ 20 câu hỏi đã được kiểm tra tự động bằng script đối chiếu với văn bản trang nguồn sau chuẩn hóa (`normalize_text`), đảm bảo 100% `evidence_quote` và `evidence_anchors` tồn tại nguyên văn tại `(gold_source_id, gold_page)`.

### 3. Tiêu chí chọn chiến lược đã chốt trước khi xem Test
Trước khi xem kết quả trên tập `test`, quy tắc chọn cấu hình trên tập `dev` (15 câu) được khóa cố định theo thứ tự ưu tiên:
1. Đạt `Evidence Hit@3` trên tập `dev` cao nhất, tiếp theo là `Hit@1` trên tập `dev` cao nhất.
2. Điểm trung bình `Chunk Coherence` (trên 10 vị trí cố định) đạt $\ge 1.5 / 2.0$ và tỷ lệ tuân thủ filter (`Filter Compliance`) đạt $100\%$.
3. Tối ưu số lượng chunk và thời gian tính toán embedding.

---

## Kết quả

### Bảng 1: Độ chính xác truy xuất (Retrieval Accuracy) và Tài nguyên theo từng Split

| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Chunk count | Thời gian (Embed + Upsert + Search) | Chi phí |
|---|---|---:|---:|---:|---:|---:|
| **A (`fixed_size` 500/50, no filter)** | `dev` | `5/15` (33.33%) | `7/15` (46.67%) | 300 | 181.47s (Embed: 168.25s, Upsert: 13.03s, Search: 0.19s) | $0.00 |
| **A (`fixed_size` 500/50, no filter)** | `test` | `2/5` (40.00%) | `2/5` (40.00%) | 300 | (dùng chung index A) | $0.00 |
| **A (`fixed_size` 500/50, no filter)** | `all` | `7/20` (35.00%) | `9/20` (45.00%) | 300 | 181.47s | $0.00 |
| **B (`by_paragraph` 800, no filter)** | `dev` | `10/15` (66.67%) | `12/15` (80.00%) | 232 | 150.36s (Embed: 144.67s, Upsert: 5.52s, Search: 0.17s) | $0.00 |
| **B (`by_paragraph` 800, no filter)** | `test` | `3/5` (60.00%) | `3/5` (60.00%) | 232 | (dùng chung index B) | $0.00 |
| **B (`by_paragraph` 800, no filter)** | `all` | `13/20` (65.00%) | `15/20` (75.00%) | 232 | 150.36s | $0.00 |
| **C (`by_paragraph` 800 + filter)** | `dev` | `10/15` (66.67%) | `12/15` (80.00%) | 232 | 148.43s (Dùng chung Embed B: 144.67s, Upsert: 3.60s, Search: 0.16s) | $0.00 |
| **C (`by_paragraph` 800 + filter)** | `test` | `3/5` (60.00%) | `3/5` (60.00%) | 232 | (dùng chung index C) | $0.00 |
| **C (`by_paragraph` 800 + filter)** | `all` | `13/20` (65.00%) | `15/20` (75.00%) | 232 | 148.43s | $0.00 |

*(Ghi chú bổ sung về lỗi cắt bằng chứng `Split Evidence Error`: Ở Strategy A có tới **7/20 câu** (`5/15 dev`, `2/5 test`) lấy đúng trang vàng vào Top-3 nhưng bằng chứng bị cắt đôi qua 2 chunk liên tiếp; nếu gộp đa chunk trong Top-3 thì A tăng thêm +4 hit nhưng vẫn thấp hơn B và C).*

### Bảng 2: Đánh giá Thủ công (Coherence, Metadata Usefulness & Grounding Quality)

| Strategy | Coherence trung bình /2 (n) | Metadata đủ và đúng (n/N) | Filter tuân thủ (n/N hoặc N/A) | Grounding 0/1/2 (số câu) và trung bình /2 |
|---|---|---|---|---|
| **A (`fixed_size` 500/50)** | **0.90 / 2.0** (`n=10`, phân bố 0/1/2: `1/9/0`) | Core (`source`,`page`,`year`,`cat`): `15/15` (100%)<br>Strict (kể cả `section`): `10/15` (66.67%) | Dev: `2/3` (66.67%)<br>Test: `1/2` (50.00%)<br>All: `3/5` (60.00%) | Phân bố 0/1/2: `1 / 1 / 3`<br>Trung bình: **1.40 / 2.0** (`n=5` câu test) |
| **B (`by_paragraph` 800, no filter)** | **1.80 / 2.0** (`n=10`, phân bố 0/1/2: `0/2/8`) | Core (`source`,`page`,`year`,`cat`): `15/15` (100%)<br>Strict (kể cả `section`): `12/15` (80.00%) | Dev: `2/3` (66.67%)<br>Test: `1/2` (50.00%)<br>All: `3/5` (60.00%) | Phân bố 0/1/2: `1 / 0 / 4`<br>Trung bình: **1.60 / 2.0** (`n=5` câu test) |
| **C (`by_paragraph` 800 + filter)** | **1.80 / 2.0** (`n=10`, phân bố 0/1/2: `0/2/8`) | Core (`source`,`page`,`year`,`cat`): `15/15` (100%)<br>Strict (kể cả `section`): `12/15` (80.00%) | Dev: **`3/3` (100.0%)**<br>Test: **`2/2` (100.0%)**<br>All: **`5/5` (100.0%)** | Phân bố 0/1/2: `0 / 1 / 4`<br>Trung bình: **1.80 / 2.0** (`n=5` câu test) |

### Đường dẫn trace, nhãn thủ công và cách chọn mẫu
- **Traces đầy đủ Top-3 của 20 câu hỏi**:
  - Strategy A: [results/traces_A.json](../../results/traces_A.json)
  - Strategy B: [results/traces_B.json](../../results/traces_B.json)
  - Strategy C: [results/traces_C.json](../../results/traces_C.json)
  - Tổng hợp chỉ số & thống kê: [results/metrics_summary.json](../../results/metrics_summary.json), [results/ingestion_stats.json](../../results/ingestion_stats.json), [artifacts/experiment_log.csv](../../artifacts/experiment_log.csv).
- **Nhãn chấm thủ công Chunk Coherence (10 vị trí)**:
  - File: [results/manual_audit_coherence.csv](../../results/manual_audit_coherence.csv) và [results/manual_audit_coherence.json](../../results/manual_audit_coherence.json).
  - Cách chọn mẫu: Chọn cố định trước khi xem điểm dựa theo 10 `page_id` đầu tiên đã sắp xếp theo thứ tự từ điển (`lewis-2020-rag_p01` đến `lewis-2020-rag_p10`) tại vị trí neo khoảng 1/3 trang (character offset ~350).
- **Nhãn chấm thủ công Metadata Usefulness (15 bản ghi test/chiến lược) & Grounding Quality (5 câu test/chiến lược)**:
  - File: [results/manual_audit_metadata_grounding.csv](../../results/manual_audit_metadata_grounding.csv) và [results/manual_audit_metadata_grounding.json](../../results/manual_audit_metadata_grounding.json).
  - Cách chọn mẫu: Kiểm tra toàn bộ 5 câu thuộc tập holdout `test` (`q16`–`q20`) x Top-3 chunks = 15 bản ghi mỗi chiến lược (tổng cộng 45 bản ghi). Người chấm là tác giả bài làm (độc lập với công thức tính Hit@3 tự động).

---

## Phân tích và kết luận

### 1. So sánh A–B và B–C; 2 ví dụ sai cụ thể có ID câu và nguồn

- **So sánh A vs B (Tác động của Chunking)**:
  - Việc chuyển từ cửa sổ cố định `fixed_size` (500 ký tự, overlap 50) ở Strategy A sang chia theo đoạn và tiêu đề `by_paragraph` (`max_size = 800`) ở Strategy B giúp **tăng gấp đôi `Hit@1` trên tập dev** (từ `5/15 = 33.33%` lên `10/15 = 66.67%`), tăng `Evidence Hit@3` trên tập dev từ `7/15 (46.67%)` lên `12/15 (80.00%)`, tăng `Hit@3` trên tập test từ `2/5 (40.0%)` lên `3/5 (60.0%)`, và **tăng gấp đôi `Chunk Coherence`** từ `0.90/2.0` lên `1.80/2.0`.
  - Đồng thời, số lượng chunk giảm từ **300 xuống 232 chunks (-22.7%)**, tiết kiệm 23.58 giây thời gian tính embedding. Lý do cốt lõi là `fixed_size` cắt ngang qua câu và mệnh đề điều kiện (gây ra 7/20 câu bị lỗi `Split Evidence Error`, ví dụ câu `q05`, `q08`, `q13`, `q14`, `q20`), trong khi `by_paragraph` giữ nguyên vẹn các đoạn lập luận hoàn chỉnh.
- **So sánh B vs C (Tác động của Metadata Filtering)**:
  - Vì B và C dùng chung bộ chunk `by_paragraph`, các câu hỏi không có filter cho kết quả giống nhau. Tuy nhiên, trên nhóm 5 câu hỏi có filter (`q13`, `q14`, `q15`, `q19`, `q20`), Strategy B chỉ đạt tỷ lệ tuân thủ filter `3/5 (60.0%)` do hiện tượng **nhiễu liên tài liệu (cross-document intrusion)**: ở `q15` (hỏi về số bước fine-tune HotpotQA của ReAct 2022), chunk `vaswani-2017-transformer_p07_para_04` (nói về 100.000 bước huấn luyện của Transformer 2017) lọt vào Rank #3; nghiêm trọng hơn, ở `q19` (Test), chunk `lewis-2020-rag_p02_para_05` của bài RAG 2020 chiếm luôn vị trí **Rank #1** (`score=0.817`) của bài ReAct 2022.
  - Khi bật Metadata Filter ở **Strategy C**, tỷ lệ tuân thủ filter tăng lên **`5/5 (100.0%)`**, loại bỏ hoàn toàn các chunk xâm nhập từ sai bài báo và nâng điểm **Grounding Quality** trên tập test từ `1.60/2.0` lên **`1.80/2.0`**.

- **Hai ví dụ lỗi điển hình có ID câu và nguồn**:
  1. **Ví dụ sai số 1 — Câu `q04` (Dev, nguồn `vaswani-2017-transformer`, trang 7, bị MISS ở cả A, B, C do câu văn vắt qua ranh giới 2 trang vật lý)**:
     - Câu hỏi `q04` hỏi: *"In terms of computational complexity per layer, under what condition is a self-attention layer faster than a recurrent layer, and how can self-attention be modified when sequences are very long?"*
     - Trong PDF gốc của bài *Attention Is All You Need*, câu trả lời bắt đầu ở dòng cuối cùng của **trang 6** (`"In terms of computational complexity, self-attention layers are faster than recurrent layers when the sequence"`) và bị ngắt sang dòng đầu tiên của **trang 7** (`"length n is smaller than the representation dimensionality d..."`).
     - Khi truy xuất ở Strategy B và C, mô hình embedding xếp hạng chunk cuối trang 6 (`vaswani-2017-transformer_p06_para_06`) ở **Rank #1 với điểm rất cao (`0.893`)** và chunk `Table 1` ở trang 6 (`p06_para_00`) ở **Rank #2 (`0.836`)**, trong khi đoạn đầu trang 7 chứa mệnh đề điều kiện `"length n is smaller than d"` lại bị tách thành một chunk riêng biệt ở trang 7 và không lọt vào Top-3 (Rank #3 là `p07_para_01` nói về convolutional layers). Đây là hạn chế kinh điển của việc chia chunk cô lập theo từng trang vật lý (`page-bounded chunking`).
  2. **Ví dụ sai số 2 — Câu `q19` (Test, nguồn `yao-2022-react`, trang 5, bị MISS ở cả A, B, C do lệch độ tương đồng giữa văn xuôi và bảng số liệu thô)**:
     - Câu hỏi `q19` hỏi về điểm số cụ thể trên `FEVER` và `HotpotQA` của các phương pháp prompting so với Supervised SoTA (`Table 1` tại trang 5 bài `yao-2022-react`).
     - Ở Strategy B (không filter), Rank #1 bị lấy nhầm sang `lewis-2020-rag_p02_para_05` (`score=0.817`) vì bài RAG cũng đánh giá trên `FEVER` và so sánh với `state-of-the-art`.
     - Khi bật filter `{"category": "agent", "year": 2022}` ở Strategy C, lỗi lấy nhầm tài liệu được khắc phục triệt để: Rank #1 là `yao-2022-react_p05_para_05` (`score=0.793`, đúng trang 5 bài ReAct, chứa đoạn văn xuôi *"3.3 Results and Observations: ReAct outperforms Act consistently Table 1 shows HotpotQA and Fever results using PaLM-540B..."*). Tuy nhiên, bản thân chunk chứa các dòng số của `Table 1` (`yao-2022-react_p05_para_00`: `"Prompt Methoda HotpotQA Fever (EM) (Acc) Standard 28.7 57.1..."`) chỉ gồm tên viết tắt và các chữ số, nên dense vector của nó có Cosine Similarity với câu hỏi tự nhiên thấp hơn các đoạn văn xuôi ở trang 5, trang 3 và trang 1, dẫn đến không lọt vào Top-3.

### 2. Chiến lược chọn, đánh đổi và điều kiện áp dụng
- **Chiến lược được chọn**: **Configuration C (`by_paragraph` với `max_size=800` kết hợp Metadata Filtering khi có filter)**.
- **Các đánh đổi kỹ thuật (Trade-offs)**:
  - *Chunking (`by_paragraph` vs `fixed_size`)*: `by_paragraph` đạt độ mạch lạc (`1.80/2.0`) và độ chính xác truy xuất (`Hit@3 = 75.0%` toàn tập) cao vượt trội đồng thời tạo ít chunk hơn (`232` vs `300`), nhưng kích thước chunk có phương sai lớn hơn (`47` đến `799` ký tự) và phụ thuộc chặt vào chất lượng bộ khôi phục ngắt đoạn trong `normalize_text`.
  - *Metadata Filtering (Strategy C vs Strategy B)*: Lọc cứng theo `category` và `year` đảm bảo `100%` kết quả nằm đúng miền tài liệu và tăng chất lượng Grounding (`1.80/2.0`), nhưng đánh đổi bằng rủi ro **Zero Recall** nếu tầng ứng dụng phía trước phân loại sai `category`/`year` của câu hỏi người dùng.
- **Điều kiện áp dụng**: Phù hợp cho kho tài liệu kỹ thuật/học thuật có cấu trúc đoạn và tiêu đề rõ ràng, nơi ngữ cảnh phiên làm việc (hoặc bộ Router/Query Analyzer) cung cấp được metadata đáng tin cậy. Không tự suy rộng rằng cấu hình này tối ưu cho mọi loại văn bản (như bảng biểu tài chính dày đặc hoặc văn bản OCR nhiều lỗi).

### 3. Hạn chế của dữ liệu, nhãn, tập test nhỏ và phép đo thời gian/chi phí
- **Hạn chế của quy mô mẫu**: Corpus gồm 37 trang nội dung hợp lệ (3 tài liệu) và tập `test` chỉ gồm **5 câu hỏi** (trong đó có 2 câu có filter). Với $N_{\text{test}} = 5$, mỗi câu hỏi tương ứng tới $20\%$ độ chính xác, nên phương sai thống kê lớn; tập test chỉ mang tính minh họa kiểm chứng độc lập, không dùng để tuyên bố hiệu năng tổng quát.
- **Hạn chế của nhãn thủ công (Manual Audit Bias)**: Việc chấm `Chunk Coherence`, `Metadata Usefulness` và `Grounding Quality` do chính tác giả thực hiện. Mặc dù đã cố định cách chọn 10 vị trí (`pos_01`–`pos_10`) trước khi xem điểm và công khai lý do từng nhãn trong file CSV/JSON, vẫn tồn tại nguy cơ thiên lệch chủ quan (author bias).
- **Hạn chế của phép đo metadata `section`**: Việc trích xuất tự động `section` bằng Regex gặp hiện tượng trôi nhãn ở đầu trang mới (`Intra-page Section Drift` trên 3/15 bản ghi test của B/C) khi đoạn văn tiếp nối từ trang trước đứng trước một tiêu đề mục mới ở giữa trang.
- **Hạn chế đo thời gian/chi phí**: Thời gian tạo embedding (`144s–168s`) được đo trên CPU local với kiểu dữ liệu `bfloat16`; nếu chạy trên GPU hoặc chuyển sang `float32` trên CPU hỗ trợ AVX-512, thời gian sẽ khác biệt đáng kể.

### 4. Hướng cải tiến nếu làm lại hoặc mở rộng
1. **Cross-Page Sliding Paragraph Chunking**: Cho phép gom đoạn văn liên tục qua ranh giới hai trang vật lý và lưu metadata dạng `page_range: [p_start, p_end]`, đồng thời truyền trạng thái `active_section` chính xác tới từng đoạn văn thay vì cấp độ trang để khắc phục hoàn toàn lỗi ở câu `q04` và lỗi trôi nhãn `section`.
2. **Table-Context Enrichment & Hybrid Retrieval (BM25 + Dense)**: Tự động gắn câu tiêu đề bảng (`Table Caption`) vào đầu mỗi chunk bảng số liệu và kết hợp điểm từ khóa **BM25** với **Dense Cosine Similarity** (Reciprocal Rank Fusion) để kéo các bảng số liệu như `Table 1` ở câu `q19` vào Top-3.
3. **Mở rộng quy mô Benchmark**: Tăng tập `test` lên $\ge 30$ câu hỏi đa dạng và sử dụng đánh giá chéo độc lập (blind double-annotation) giữa nhiều người chấm.
