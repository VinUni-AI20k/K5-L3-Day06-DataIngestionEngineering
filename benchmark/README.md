# Benchmark · 24 câu hỏi (18 dev / 6 test)

Bộ câu hỏi cho [BENCHMARK.md](../BENCHMARK.md), viết **trước khi chạy bất kỳ truy xuất nào** và đóng băng ngày 2026-11-14 (`benchmark_version` 1.0). Thời điểm commit file này trên git là bằng chứng cho thứ tự đó. Không sửa tập test sau khi xem kết quả; nếu buộc phải sửa, tăng version và ghi lý do bên dưới.

| File | Nội dung |
|---|---|
| `questions.json` | 24 câu: `id`, `split`, `type`, `question`, `expected_answer`, `gold_source_id`, `gold_page`, `gold_pages`, `evidence_quote`, `filter`, ghi chú |
| `verification.json` | Kết quả kiểm chứng tự động (sinh bởi script) |

Kiểm chứng: chạy `python -m src.ingestion` rồi `python scripts/verify_benchmark.py` từ thư mục `starter/`.

## Cách thiết kế

**Phân bố**

| Loại (`type`) | Số câu | Ý nghĩa |
|---|---:|---|
| `truc_tiep` | 5 | Hỏi gần với câu chữ trong tài liệu |
| `dien_dat_lai` | 5 | Diễn đạt khác tài liệu ("hay để lại biến chứng gì nhất") |
| `dieu_kien` / `ngoai_le` | 5 / 1 | Có điều kiện, ngưỡng hoặc ngoại lệ (≥6 kg, >½ chiều dài Implant, thử tủy ở trẻ em) |
| `metadata_filter` | 5 | Cần filter `disease_no` mới phân biệt được (3 dev, 2 test) |
| `phan_mo_dau` | 3 | Phần mở đầu: từ viết tắt, chủ biên, Quyết định (trang OCR) |

**Tách dev/test theo bệnh:** tập test dùng bệnh 18, 30, 43, 52, 57 và trang Quyết định. Không bệnh nào trong số này có câu hỏi ở dev; script kiểm tra tự động điều này.

**Nguy cơ rò rỉ (ghi nhận, không loại bỏ được):** một số bệnh test có "bệnh anh em" trong dev với văn bản theo cùng khuôn: 30 (áp xe má) với 31 (áp xe dưới hàm), 18 với 14 (sai khớp cắn loại II), 43 với 42 (viêm tuyến mang tai), trang Quyết định với các trang mở đầu khác. Nếu chỉnh tham số theo dev, kết quả có thể "học" theo khuôn này, nên test không hoàn toàn độc lập.

**Câu có filter:** `filter = {"disease_no": N}` là **thông tin ngữ cảnh giả định có sẵn trước khi truy xuất** (người dùng đang đọc phác đồ của bệnh N). Filter không được suy ra từ đáp án. Ba câu (d08, t01, t02) cố ý **không nêu tên bệnh**, vì các chương áp xe 29–34 và sai khớp cắn 13–22 dùng câu chữ gần như giống nhau. A và B nhận cùng câu hỏi nhưng không có filter, C áp dụng filter: đây là phép đo B–C. Với câu không có filter, C chạy không lọc, giống B.

**Evidence:** mỗi `evidence_quote` là đoạn nguyên văn trong text đã chuẩn hóa, so khớp sau khi gộp mọi khoảng trắng. Một hit (GUIDE §4) là chunk thuộc một trang trong `gold_pages` và chứa **đủ** evidence.

## Quyết định về gold

| Câu | Vấn đề | Quyết định |
|---|---|---|
| d04 | Câu về Streptococus mutans xuất hiện y hệt ở tr.40 (bệnh 9) | **Chấp nhận** tr.40: câu không nhắc răng sữa, trả lời đúng câu hỏi |
| t01 | Câu chỉ định rạch đường trong miệng xuất hiện y hệt ở tr.125 (bệnh 32) | **Không chấp nhận**: ngữ cảnh cố định bệnh 30, tr.125 là sai bệnh |
| t06 | Evidence nằm ở trang OCR | Text Điều 3 giống nhau ở cả OCR thô lẫn bản đã sửa |

## Kết quả kiểm chứng (2026-11-14)

24/24 evidence nằm trên trang gold; đủ ≥20 câu, ≥15 dev, ≥5 test, ≥3 câu filter (có câu trong test); id không trùng; không rò rỉ bệnh từ test sang dev. Cảnh báo duy nhất còn lại là t01 (đã giải thích ở trên).

## Người soạn và hạn chế

Câu hỏi, đáp án và evidence do trợ lý AI (Claude) soạn từ text đã chuẩn hóa. Script chỉ xác minh **evidence có trong trang**; script **không** xác minh evidence đủ để trả lời hay đáp án đúng về chuyên môn. Tác giả bài cần đọc lại từng câu so với PDF trước khi chạy retrieval. Benchmark nhỏ (6 câu test), nên mọi chỉ số test chỉ mang tính minh họa.
