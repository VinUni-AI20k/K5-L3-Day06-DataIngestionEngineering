# Benchmark đã chốt

- File: `benchmark/questions.json` (20 câu: 15 dev / 5 test; 5 câu có filter, 2 trong test)
- SHA-256 (tính trên nội dung với xuống dòng LF): `62d3d7a34403056aea65b79b7ea339269b04945f53cbb19a02be8718fa240f2c`
- Chốt lúc: 2026-11-13T10:30Z, trước khi chạy thử bất kỳ chiến lược truy xuất nào.
- Người soạn: bản nháp do AI soạn từ text đã chuẩn hóa; học viên đã đọc và xác nhận.
- Kiểm chứng: `python scripts/verify_benchmark.py` (từ `starter/`), mọi evidence có nguyên văn ở trang gold.

## Tiêu chí chọn chiến lược (chốt trước khi xem test)
1. Evidence Hit@3 trên **dev** cao hơn.
2. Nếu hòa: Hit@1 trên dev.
3. Nếu vẫn hòa: ít chunk hơn / chi phí embedding thấp hơn.

Tập test chỉ chạy **một lần** sau khi đã chọn chiến lược trên dev; không sửa câu hỏi, gold hay tham số sau khi xem test.
