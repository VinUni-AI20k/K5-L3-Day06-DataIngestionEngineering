# Mẫu audit

Các cột nhãn đã được điền bởi **AI (Claude)**, không phải do tác giả chấm tay. Tác giả cần đọc lại từng nhãn, sửa nếu không đồng ý, rồi ghi tên người duyệt vào báo cáo.

## coherence_positions.csv (10 vị trí × 3 cấu hình = 30 dòng)
- Vị trí: đoạn đầu tiên của trang p001–p010, chọn trước khi chấm.
- Thang 0/1/2 (GUIDE): 0 = bị cắt giữa ý, không hiểu; 1 = hiểu một phần, thiếu ngữ cảnh; 2 = tự hiểu được, giữ trọn ý.
- Quy tắc đã dùng: chunk bắt đầu hoặc kết thúc giữa câu, hoặc bắt đầu giữa lời thoại mà không rõ người nói → tối đa 1; bắt đầu bằng mảnh câu không xác định chủ thể → 0 nếu cũng cắt cuối câu, 1 nếu kết thúc đúng ranh giới đoạn; chunk trọn đoạn → 2.
- B và C có cùng chunk nên cùng nhãn.

## grounding_test.csv (5 câu test q16–q20 × 3 cấu hình × top-3)
- Nhãn đặt ở hàng rank 1 của mỗi câu (một nhãn cho cả top-3).
- Thang: 0 = không có bằng chứng trong top-3; 1 = có nhưng thiếu điều kiện cần thiết; 2 = đủ để trả lời và truy nguồn.
- Đã kiểm tra thêm: cụm từ chính của evidence có xuất hiện trong các trang top-3 hay không (xem lý do từng câu).

## metadata_test_top3.csv (8 câu test × top-3 × 3 cấu hình = 72 dòng)
- `manual_correct` = yes khi text chunk nằm trong đúng trang và chương trong metadata khớp trang gốc. Cột này cũng được kiểm tra bằng script (`auto_text_in_page`, `auto_chapter_match`).
- Lưu ý: nhãn chỉ kiểm tra nguồn/trang/chương, không phải chất lượng của trường filter.

Mọi nhãn phải được tác giả xem lại trước khi nộp.
