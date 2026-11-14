# Tiêu chí chọn cấu hình (chốt trước khi chạy)

Chốt ngày 2026-11-14, **trước mọi lần chạy retrieval** và trước khi xem bất kỳ kết quả test nào. Thời điểm commit file này trên git là bằng chứng.

Chỉ dùng **tập dev (18 câu)** để chọn. Tập test (6 câu) chạy **một lần** sau khi đã chọn xong, và chỉ để báo cáo.

1. **Evidence Hit@3 trên dev** cao nhất thắng.
2. Nếu chênh nhau **≤ 1 câu** (≈ 6% trên 18 câu, nằm trong sai số của mẫu nhỏ): chọn cấu hình có **Hit@1 trên dev** cao hơn.
3. Nếu vẫn hòa: chọn cấu hình có **ít chunk hơn**, rồi tới **thời gian ingest ngắn hơn** (chi phí embedding gần như tỷ lệ với số ký tự, và giống nhau giữa B và C).

Không dùng để chọn: coherence và grounding là nhãn chấm tay trên mẫu cố định/tập test, được báo cáo để giải thích, không để tinh chỉnh.

Không tinh chỉnh tham số (kích thước chunk, overlap, ngưỡng) sau khi xem test. Nếu phải làm vậy, ghi rõ trong báo cáo và không gọi kết quả test là holdout độc lập.
