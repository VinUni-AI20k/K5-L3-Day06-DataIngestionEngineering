# Benchmark — Pride and Prejudice

Bộ câu hỏi `questions.jsonl`: **23 câu**, **15 dev / 8 test**, trong đó **5 câu có metadata filter** (2 dev, 3 test). Kiểm tra bằng:

```bash
cd starter && .venv/bin/python scripts/check_benchmark.py
```

Script xác nhận: đủ số lượng và tỷ lệ split; đủ ≥30 trang nội dung; mọi `evidence_quote` nằm nguyên văn trong trang gold (`data/pages.jsonl`); filter không loại mất trang gold.

Corpus được dựng thành PDF bằng `scripts/build_pdf.py` (216 trang, mỗi chương bắt đầu một trang mới). Số trang là do PDF sinh ra, **không phải số trang của ấn bản in**; xem `derived_pdf` trong `data/manifest.json`.

## Trường dữ liệu

| Trường | Ý nghĩa |
|---|---|
| `id`, `split` (`dev`/`test`), `type` | Mã câu; dev hoặc test; loại câu: `truc_tiep`, `dien_dat_lai`, `dieu_kien` |
| `question`, `expected_answer` | Câu hỏi tiếng Việt và đáp án tham chiếu |
| `gold_source_id` | `pride-and-prejudice` (xem `data/manifest.json`) |
| `gold_page` | ID trang trong PDF đã dựng: `pride-and-prejudice:pNNN` (xem `data/pages.jsonl`) |
| `evidence_quote` | Đoạn tiếng Anh nguyên văn trong đơn vị gold, đủ để trả lời |
| `filter` | `{"chapter": N}` hoặc `null`. Giá trị coi là **đầu vào người dùng cung cấp trước khi truy xuất**, không lấy từ đáp án |

## Quy tắc áp dụng

- Cấu hình C chỉ áp filter cho câu có `filter`; câu còn lại chạy không lọc.
- Filter chỉ được chọn bằng chương chứa gold, nên nó kiểm tra khả năng lọc chứ không làm câu dễ hơn bằng cách loại gold.
- Câu hỏi và đáp án viết tiếng Việt, evidence giữ tiếng Anh để khớp với corpus. Đây là một thử thách truy xuất đa ngôn ngữ; ghi lại trong báo cáo.
- Dùng cùng bộ câu hỏi, cùng nhãn cho cả ba cấu hình.

## Tiêu chí chọn cấu hình (chốt trước khi chạy test)

1. Evidence Hit@3 trên **dev**.
2. Nếu hòa: Hit@1 trên dev.
3. Nếu vẫn hòa: grounding chấm thủ công trên dev.
4. Cuối cùng: thời gian và chi phí ingest.

Test chỉ được chạy sau khi cấu hình đã chọn trên dev. Nếu phải xem test để sửa, ghi rõ trong báo cáo.

## Giới hạn

- Test chỉ 8 câu, filter test 3 câu: kết quả chỉ minh họa, không đủ kết luận tổng quát.
- Bộ câu hỏi do AI soạn, sau đó được kiểm tra bằng script (evidence, split, filter). Người làm bài cần đọc lại từng câu và hiểu đáp án trước khi dùng.
- Câu hỏi chưa được đối chiếu với các đoạn khác của sách ngoài evidence đã ghi; có thể có đáp án thứ hai hợp lệ. Hiện script không phát hiện evidence trùng ở trang khác.
- Trang do PDF tự dựng, không phải trang in gốc; chương không đổi nên filter theo chương vẫn có nghĩa.
