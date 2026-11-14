# Báo cáo cá nhân — Day 6


## Dữ liệu và pipeline

- **Chủ đề:** Luật Sở hữu trí tuệ, Văn bản hợp nhất số 155/VBHN-VPQH ngày 09-09-2025.
- **Nguồn/quyền:** nguồn công khai tại [Cổng văn bản Chính phủ](https://vanban.chinhphu.vn/?pageid=27160&docid=215309); tình trạng công khai.
- **Số trang:** 30 trang nội dung, trang vật lý 2–31 của PDF gốc; loại trang vật lý 1. Mỗi trang được lưu thành một record, không gộp trang. Tổng 73.902 ký tự đã chuẩn hóa; manifest khớp số trang và số ký tự từng record.
- **Chuẩn hóa/lỗi/giới hạn:** trích xuất text PDF, chuẩn hóa Unicode NFC và khoảng trắng; lần chạy có 30 trang không rỗng. PDF có xuống dòng giữa câu và số trang/dấu chú thích, có thể ảnh hưởng ranh giới chunk. Pipeline không thực hiện OCR.
- **Embedding:** OpenRouter, `openai/text-embedding-3-large`, 3.072 chiều; dùng cùng model cho document và query. 275 chunk văn bản duy nhất được embed cùng 20 query; 79.722 input tokens; thời gian embed document 17,96 s, query 23,10 s; chi phí do provider báo cáo là **$0,01036386**.
- **Vector store:** PostgreSQL 16.16 + pgvector 0.8.7, database `day6_lab`, cosine distance (`<=>`), tìm kiếm exact scan.
- **Metadata:** `document_id`, `source`, `page`, cùng `section`, `year`, `source_version`, `unit_id` và thông tin truy vết khác. `section` được lấy từ corpus/manifest; hai giá trị là “Phần thứ nhất” và “Phần thứ hai”, dùng lọc phạm vi khi câu hỏi cung cấp section.

## Cấu hình và benchmark

- **A:** fixed-size 500 ký tự, overlap 50, không lọc; 177 chunks.
- **B:** chunk theo ranh giới Điều/khoản/điểm khi có thể, giới hạn 1.000 ký tự; phần quá dài được chia gần ranh giới từ với overlap tối đa 300 ký tự; không lọc; 98 chunks.
- **C:** dùng đúng chunks của B, chỉ thêm lọc metadata khi câu hỏi có filter; 98 chunks. B–C giữ nguyên embedding, corpus và benchmark.
- **Benchmark:** 20 câu đã có evidence, 15 dev / 5 test; 3 câu có filter, trong đó 2 câu test. Evidence được kiểm tra với text trang nguồn; filter được kiểm tra với metadata trang gold.
- **Tiêu chí chọn chiến lược trước khi xem test:** không tìm thấy ghi nhận tiêu chí được chốt trước. Không suy diễn tiêu chí sau khi đã xem test; cần ghi lại cho lần thử tiếp theo.

## Kết quả



| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Chunk count | Upsert / search (s) | Chi phí |
|---|---|---:|---:|---:|---:|---|
| A | dev | 3/15 (20,0%) | 5/15 (33,3%) | 177 | 0,400 / 0,524 | Chung: $0,01036386 |
| A | test | 2/5 (40,0%) | 2/5 (40,0%) | 177 | 0,400 / 0,524 | Chung: $0,01036386 |
| B | dev | 6/15 (40,0%) | 10/15 (66,7%) | 98 | 0,633 / 0,485 | Chung: $0,01036386 |
| B | test | 2/5 (40,0%) | 3/5 (60,0%) | 98 | 0,633 / 0,485 | Chung: $0,01036386 |
| C | dev | 6/15 (40,0%) | 10/15 (66,7%) | 98 | 0,512 / 0,478 | Chung: $0,01036386 |
| C | test | 2/5 (40,0%) | 3/5 (60,0%) | 98 | 0,512 / 0,478 | Chung: $0,01036386 |

| Strategy | Coherence trung bình /2 (n) | Metadata đủ/đúng (n/N) | Filter tuân thủ | Grounding thủ công 0/1/2 |
|---|---|---:|---|---|
| A | 0,8/2 (n=10; 0:2, 1:8, 2:0) | 15/15 | 9/9 | 1,4/2 (n=5; 0:1, 1:1, 2:3) |
| B | 1,4/2 (n=10; 0:1, 1:4, 2:5) | 15/15 | 8/9 | 1,4/2 (n=5; 0:1, 1:1, 2:3) |
| C | 1,4/2 (n=10; 0:1, 1:4, 2:5) | 15/15 | 9/9 | 1,4/2 (n=5; 0:1, 1:1, 2:3) |


### Trace và nhãn

- Kết quả được báo cáo: [run-20261114T043724Z]. Lần chạy cũng tạo [run-20261114T043639Z] với cùng summary; chỉ dùng run mới nhất ở đây, không gộp traces.
- [Trace top-3](../../starter/results/run-20261114T043724Z/retrieval_traces.jsonl), [summary](../../starter/results/run-20261114T043724Z/summary.json), [metadata lần chạy](../../starter/results/run-20261114T043724Z/run_metadata.json).
- Mẫu chấm: [coherence](../../starter/results/run-20261114T043724Z/manual_coherence_review.jsonl), [grounding](../../starter/results/run-20261114T043724Z/manual_grounding_review.jsonl), [metadata](../../starter/results/run-20261114T043724Z/manual_metadata_review.jsonl). 
- Coherence lấy chunk đầu mỗi trang trong 10 trang đầu corpus (trang vật lý 2–11). Đây là chấm mẫu theo vị trí, không phải đánh giá độc lập/mù; cũng chưa ghi nhận tiêu chí chọn strategy trước khi xem test.
- Metadata được đối chiếu trên 5 câu test × top-3 cho mỗi strategy (15 bản ghi/strategy): `document_id`, `source`, `page`, `unit_id`, `section` và `source_version` khớp trace và record trang nguồn ở cả 45/45 bản ghi. `source`/`page` giúp truy vết về văn bản/trang; `section` hỗ trợ lọc phạm vi nhưng chỉ có hai nhóm thô.

## Phân tích và kết luận

- **A–B:** trên dev, B tăng Hit@1 từ 3/15 lên 6/15 và Evidence Hit@3 từ 5/15 lên 10/15. Trên test, B có Hit@1 bằng A (2/5) và Evidence Hit@3 cao hơn (3/5 so với 2/5). B/C cùng có 2/5 Hit@1 và 3/5 Evidence Hit@3 trên test.
- **B–C:** cùng chunk và model; C chỉ lọc khi câu hỏi có filter. C tuân thủ filter 9/9 hit, trong khi B tuân thủ tình cờ 8/9; tuy vậy câu q20 vẫn trượt dù filter đúng, nên filter giúp giới hạn phạm vi chứ không đảm bảo tìm đúng bằng chứng.
- **Ví dụ lỗi 1 — q19:** gold trang 21 hỏi thời hạn bảo hộ khi có đồng tác giả; B/C không có hit top-3 (trả về trang 20, 28, 20). Evidence nằm nguyên trong chunk theo kiểm tra cục bộ nhưng không được xếp đủ cao.
- **Ví dụ lỗi 2 — q20:** gold trang 27 hỏi thời hạn quyền tổ chức phát sóng. C áp filter “Phần thứ hai” nhưng top-3 là trang 24, 24, 13; không có evidence gold. Lọc đúng section không đảm bảo truy xuất đúng điều khoản.
- **Lựa chọn:** chưa chốt strategy cuối cùng. B/C cải thiện Evidence Hit@3 dev và test so với A ở lần chạy này, nhưng còn trượt q19/q20, mẫu test chỉ 5 câu và thiếu nhãn thủ công/tiêu chí chọn trước.
.
