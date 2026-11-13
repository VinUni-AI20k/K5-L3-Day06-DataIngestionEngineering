# Báo cáo cá nhân — Day 6

## Dữ liệu và pipeline
- Chủ đề, nguồn/quyền, số trang và quy tắc tính trang: *Pride and Prejudice* (Jane Austen), Project Gutenberg eBook #1342, public domain theo giấy phép Gutenberg trong file (xem `data/manifest.json`). Bản `.txt` không có trang nên đã **tự dựng PDF** (`starter/scripts/build_pdf.py`, A4, Times 11pt, có số trang): 216 trang, 61 chương, mỗi chương bắt đầu trang mới. Đây là **trang do PDF sinh ra, không phải trang ấn bản in**. Số trang nội dung: 216/216, trang rỗng: 0. Bốn trang cuối chương có dưới 100 ký tự nhưng vẫn là nội dung truyện, nên được tính.
- Chuẩn hóa, lỗi và giới hạn: `load_pdf` dùng pypdf ở chế độ `layout` để giữ ranh giới đoạn; lấy chương từ heading "Chapter N" (61/61 trang có heading), bỏ số trang ở chân trang và dòng `]` còn sót. Không làm OCR (PDF có text). Chunk không vượt ranh giới trang nên evidence nằm giữa hai trang sẽ bị cắt.
- Lỗi đã sửa trong quá trình làm: bộ tách câu coi "Mr." là hết câu, làm chunk B kết thúc bằng "hopes of Mr." (phát hiện qua audit coherence). Đã sửa và thêm test hồi quy; pipeline chạy lại. Hệ quả: số chunk B và điểm B/C thay đổi (xem phần Kết quả).
- Embedding model/provider/version/số chiều, vector store/version, metric, lệnh chạy và chi phí:
  - Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (local, 384 chiều, cosine, vector đã chuẩn hóa), `max_seq_length = 256`. Không cần API key, chi phí 0.
  - Vector store: PostgreSQL 16 + pgvector 0.8.7 (Docker `pgvector/pgvector:0.8.7-pg16`), tìm kiếm exact (không có index ANN).
  - Lệnh chạy: xem `starter/RUN.md`. Tóm tắt: `docker compose -p day6_student_starter up -d --wait`, rồi `.venv/bin/python -m src.run`.
  - Thời gian ingest (chunk + embed + store): khoảng 107 giây trên CPU, dùng chung cho cả ba cấu hình.
- Metadata bắt buộc, trường lọc bổ sung, nguồn giá trị và lý do chọn: `document_id`, `source`, `page`, `chapter` (lấy từ heading trong PDF, không tự đoán). Trường lọc là `chapter` (1–61) vì chương phân biệt rõ và có trong nguồn.

## Cấu hình và benchmark
- A / B / C:
  - **A**: chia cửa sổ cố định 500 ký tự, overlap 50, trong từng trang; 1.610 chunk, 191.091 token (không chunk nào vượt 256 token nên không bị cắt).
  - **B**: gom đoạn theo ranh giới đoạn, tối đa 800 ký tự, tách đoạn quá dài theo câu; 1.166 chunk, 171.719 token.
  - **C**: cùng chunk với B; chỉ khác ở việc áp filter `chapter` cho các câu có filter (5 câu). Các câu không có filter cho kết quả giống hệt B (18/18 câu đã kiểm tra).
- ≥20 câu: 23 câu, phân chia **15 dev / 8 test**; 5 câu có filter (2 dev, 3 test). Cách kiểm chứng: `scripts/check_benchmark.py` xác nhận mọi evidence nằm nguyên văn trong đúng trang gold và filter không loại mất trang gold. Bộ câu hỏi do AI soạn, chưa được tác giả đối chiếu từng câu với sách.
- Tiêu chí chọn chiến lược đã chốt trước khi xem test (`benchmark/README.md`, giả thuyết ghi trong `results/hypotheses.md`): Hit@3 dev → Hit@1 dev → grounding thủ công dev → chi phí/thời gian.
- Trung thực về thứ tự: lỗi tách câu được phát hiện khi chấm coherence, tức là khi đọc văn bản chunk, chứ không phải khi xem kết quả retrieval. Sau khi sửa, số liệu B và C thay đổi nhưng thứ hạng theo tiêu chí đã chốt không đổi (A vẫn đứng đầu).

## Kết quả

| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Chunk count | Thời gian ingest | Chi phí |
|---|---|---|---|---|---|---|
| A | dev | 3/15 (20.0%) | 5/15 (33.3%) | 1.610 | ~107 s (chung) | 0 |
| A | test | 2/8 (25.0%) | 3/8 (37.5%) | | | |
| B | dev | 2/15 (13.3%) | 3/15 (20.0%) | 1.166 | ~107 s (chung) | 0 |
| B | test | 2/8 (25.0%) | 2/8 (25.0%) | | | |
| C | dev | 2/15 (13.3%) | 4/15 (26.7%) | 1.166 | dùng lại vector B | 0 |
| C | test | 2/8 (25.0%) | 3/8 (37.5%) | | | |

Kiểm tra tự động và nhãn audit (nhãn do AI gán, tác giả cần duyệt; xem `results/audit/README.md`):

| Strategy | Coherence trung bình /2 (n = 10 vị trí) | Metadata đúng, kiểm tra tự động + nhãn (n/N) | Filter tuân thủ (n/N) | Grounding 0/1/2 (5 câu test) và trung bình /2 |
|---|---|---|---|---|
| A | 0,8 (8 điểm 1, 2 điểm 0) | 24/24 | N/A (không lọc) | 1 câu 2, 2 câu 1, 2 câu 0 → trung bình 0,8 |
| B | 1,6 (6 điểm 2, 4 điểm 1) | 24/24 | N/A | 1 câu 2, 2 câu 1, 2 câu 0 → trung bình 0,8 |
| C | 1,6 (6 điểm 2, 4 điểm 1) | 24/24 | 9/9 chunk top-3 của câu có filter đúng chương | 1 câu 2, 2 câu 1, 2 câu 0 → trung bình 0,8 |

Chi tiết từng nhãn và lý do: `results/audit/coherence_positions.csv`, `grounding_test.csv`, `metadata_test_top3.csv`.

Đường dẫn trace và cách chọn mẫu:
- Trace đầy đủ top-3 (ID chunk, trang, chương, điểm, text, cờ gold): `results/traces_A.jsonl`, `results/traces_B.jsonl`, `results/traces_C.jsonl`.
- Tóm tắt: `results/summary.json`; nhật ký: `artifacts/experiment_log.csv`.
- Mẫu chấm được chọn **trước khi xem số**: coherence cho đoạn đầu của 10 trang đầu (p001–p010); grounding cho 5 câu test đầu (q16–q20).

Lỗi phân tích (trang gold có trong top-3 hay không):
- A: 2 câu đúng trang nhưng evidence bị cắt (q02, q08); 13 câu trang gold không nằm trong top-3.
- B và C: không có câu nào đúng trang nhưng evidence bị cắt; 18 (B) và 16 (C) câu có trang gold ngoài top-3.

## Phân tích và kết luận
- **So sánh A–B (H1, giả thuyết: B > A)**: **không được xác nhận**. Dev Hit@3: B 3/15 so với A 5/15; dev Hit@1: 2/15 so với 3/15; test Hit@3: 2/8 so với 3/8. Ví dụ sai ở B: q01 (gold p001, top-3 gồm các trang 40, 40, 26); q02 (gold p001, top-3 gồm các trang 162, 47, 58). Trang gold không nằm trong top-3 nên đây là lỗi tìm trang, không phải lỗi cắt chunk. Điều này không có nghĩa là chunk theo đoạn tệ hơn về mặt lý thuyết; trong dữ liệu này, lỗi chính nằm ở truy xuất.
- **So sánh B–C (H2, giả thuyết: C không kém B và tốt hơn trên câu có filter)**: đúng hướng nhưng khác biệt nhỏ. Dev Hit@3: C 4/15 so với B 3/15; test Hit@3: C 3/8 so với B 2/8. Đây là chênh lệch 1 câu mỗi bên, không đủ để kết luận. Filter được tuân thủ 9/9; các câu không có filter cho kết quả giống B.
- **Chất lượng tuyệt đối thấp**: Hit@3 cao nhất chỉ 37,5% (test, A và C). Kiểm tra đã loại trừ lỗi lưu trữ và lỗi ngôn ngữ:
  - Vector trong DB khớp với vector tính trực tiếp (cosine 1,0000).
  - Viết lại 6 câu hỏi bằng tiếng Anh (chỉ để chẩn đoán, không đưa vào benchmark) trên chiến lược B: cả 6 câu vẫn không có trang gold trong top-3, nên lỗi không nằm ở ngôn ngữ. Script chẩn đoán nằm trong scratchpad của phiên làm việc, không nằm trong repo.
  - Nguyên nhân chủ yếu là mô hình paraphrase được huấn luyện để so sánh các câu tương đương, không phải để tìm đoạn trả lời câu hỏi. Ví dụ: điểm similarity của một chunk không liên quan lên tới ~0,43, trong khi chunk đúng chỉ ~0,18.
- **Chiến lược được chọn theo tiêu chí đã chốt**: A (Hit@3 dev cao nhất, 5/15; C là 4/15; B là 3/15). Đánh đổi: A chia cắt câu theo ký tự nên dễ cắt evidence (2 câu trong dữ liệu này), nhưng vẫn tìm đúng trang nhiều hơn. Điều kiện áp dụng: corpus này, model này, benchmark nhỏ này.
- **Chất lượng chunk (coherence)**: B và C giữ trọn đoạn (điểm 2 ở 6/10 vị trí), A cắt giữa câu ở phần lớn vị trí (điểm trung bình 0,8). Đây là điểm mạnh của chunk theo đoạn, nhưng không chuyển thành Hit@3 cao hơn trong benchmark này.
- **Hạn chế**:
  - Benchmark nhỏ (15 dev, 8 test, 3 test có filter). Chênh lệch 1 câu không có ý nghĩa thống kê; không nên kết luận chiến lược nào tốt nhất tổng quát.
  - Trang do PDF tự sinh, không phải trang ấn bản in.
  - Bộ câu hỏi do AI soạn, chưa được tác giả đối chiếu từng câu với sách.
  - Nhãn coherence, grounding và metadata do AI gán theo thang GUIDE, chưa do tác giả chấm tay. Số liệu ở trên chỉ có giá trị sau khi tác giả duyệt lại.
  - Grounding chỉ 5 câu × 3 cấu hình; trung bình 0,8 có độ tin cậy thấp.
  - Chưa thử model retrieval chuyên dụng; kết quả hiện tại phản ánh giới hạn của model paraphrase.
- **Nếu làm lại hoặc mở rộng**: thay model bằng một model retrieval đa ngôn ngữ (ví dụ họ multilingual-e5) và so sánh lại A/B/C; tăng số câu test; thêm index ANN để kiểm chứng; tác giả tự chấm lại các mẫu đã chuẩn bị.
