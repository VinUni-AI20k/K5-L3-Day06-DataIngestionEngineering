# Chọn chiến lược (trên dev, trước khi chạy test)

Kết quả dev (15 câu), từ `summary_dev.csv`:

| Strategy | Hit@1 | Evidence Hit@3 | Filter tuân thủ top-3 | Chunk |
|---|---|---|---|---:|
| A | 8/15 | 12/15 | 1/3 | 611 |
| B | 8/15 | 12/15 | 0/3 | 444 |
| C | 8/15 | 12/15 | 3/3 | 444 (chung index B) |

Áp dụng tiêu chí đã chốt trong `benchmark/FROZEN.md`:
1. Evidence Hit@3: hòa (12/15).
2. Hit@1: hòa (8/15).
3. Ít chunk / chi phí hơn: loại A; B và C vẫn hòa (chung index).

**Phân định B–C (bổ sung, không có trong tiêu chí chốt ban đầu):** chọn **C** vì tuân thủ filter trên dev 3/3 so với 0/3 của B. Bước phân định này được quyết định sau khi xem kết quả dev, trước khi chạy test; học viên xác nhận ngày 2026-11-13.

Test chạy một lần sau quyết định này; báo cáo cả A, B, C trên test để so sánh, không đổi lựa chọn theo kết quả test.
