# Giả thuyết (ghi trước khi chạy pipeline)

Ghi ngày 2026-11-13, trước khi có bất kỳ số liệu retrieval nào.

- **H1 (A vs B, chunking):** B (chia theo đoạn, tối đa 800 ký tự) có Evidence Hit@3 cao hơn A (cửa sổ cố định 500/50) trên dev, vì A có thể cắt ngang câu evidence.
- **H2 (B vs C, filter):** C giống B ở mọi câu không có filter. Trên 5 câu có filter, C có tỷ lệ top-3 tuân thủ filter là 100%, và Hit@3 không kém B. Trên các câu không có filter, kết quả C phải bằng B.
- **Hủy bỏ:** nếu A hoặc B cho Hit@3 = 0 trên dev, pipeline có lỗi và phải sửa trước khi đọc test.

Tiêu chí chọn cấu hình (đã chốt trong `benchmark/README.md`): Hit@3 dev, rồi Hit@1 dev, rồi grounding thủ công dev, rồi chi phí/thời gian.
