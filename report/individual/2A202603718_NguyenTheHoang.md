# Báo cáo cá nhân — Day 6

## Dữ liệu và pipeline
- Corpus là Quy chế đào tạo HUST 2025 bản tiếng Anh, truy cập ngày 13/11/2026 từ cổng thông tin HUST. Tính 32/36 trang vật lý, từ trang PDF 5 đến 36. Loại bìa, hai trang mục lục và một trang trống. Tài liệu được công bố công khai. 
- Pipeline trích văn bản bằng pypdf, chuẩn hóa Unicode NFC và khoảng trắng, giữ ngắt dòng và số trang vật lý. Trang PDF 4 không có văn bản trích xuất. Text trích từ PDF có thể chứa lỗi ngắt từ hoặc lệch thứ tự đọc ở bảng.
- Dùng `text-embedding-3-small` qua API tương thích OpenAI, 1.536 chiều. Cùng một model mã hóa câu hỏi và chunk. PostgreSQL 16.16 và pgvector 0.8.7 thực hiện tìm kiếm cosine chính xác. Chạy `python -m src.run --split dev` và `python -m src.run --split test` từ thư mục `starter/`. Tổng chi phí API theo số liệu học viên cung cấp là **0,001 USD** cho thí nghiệm. Chi phí theo cấu hình, theo tập và số token chưa được ghi nhận.
- Mỗi chunk giữ `document_id`, `source`, `page` và vị trí bắt đầu `start`. Trường `program_level` là chuỗi gán theo phạm vi chương Đại học, Kỹ sư và Thạc sĩ. Trường này hỗ trợ giới hạn truy xuất khi câu hỏi chỉ định một trong ba chương trình.

## Cấu hình và benchmark
- A chia từng trang theo cửa sổ 500 ký tự, chồng lấn 50 ký tự. B chia theo cấu trúc văn bản, tối đa 500 ký tự mỗi chunk. C dùng lại chunk và vector của B và áp dụng `program_level` khi câu hỏi có bộ lọc. Cả ba dùng cùng corpus, model và benchmark.
- Benchmark gồm 20 câu dev và 5 câu test. Ba câu có bộ lọc là q10, q17 và q21. Đã đối chiếu từng trích đoạn với trang PDF được dẫn sau khi chuẩn hóa khoảng trắng. 
- Giả thuyết là C có chất lượng truy xuất tốt nhất, A thấp nhất. 
- Chọn cấu hình C theo Evidence Hit@3 trên dev, dùng Hit@1 trên dev nếu đồng hạng. Tiêu chí này được ghi sau khi đã có kết quả dev và test.

## Kết quả
| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Chunk count | Thời gian | Chi phí |
|---|---|---|---|---|---|---|
| A | dev | 2/20 | 7/20 | 179 | 4,317 s | Chi phí chung |
| B | dev | 12/20 | 16/20 | 445 | 8,883 s | Chi phí chung |
| C | dev | 13/20 | 17/20 | 445 | 1,255 s | Chi phí chung |
| A | test | 2/5 | 2/5 | 179 | 3,127 s | Chi phí chung |
| B | test | 3/5 | 5/5 | 445 | 7,415 s | Chi phí chung |
| C | test | 3/5 | 5/5 | 445 | 0,680 s | Chi phí chung |

Thời gian trong bảng là thời gian ingest. C tái sử dụng embedding của B. Tổng chi phí chung là 0,001 USD. 
| Strategy | Coherence trung bình /2 (n) | Metadata đủ và đúng (n/N) | Filter tuân thủ (n/N hoặc N/A) | Grounding 0/1/2 (số câu) và trung bình /2 |
|---|---|---|---|---|
| A | 1,1 (10) | 15/15 | 0/1 | 2 / 1 / 2 · 1,0 |
| B | 1,7 (10) | 15/15 | 0/1 | 0 / 0 / 5 · 2,0 |
| C | 1,7 (10) | 15/15 | 1/1 | 0 / 0 / 5 · 2,0 |

Tỷ lệ tuân thủ bộ lọc tính từ top-3 của q21, câu test duy nhất yêu cầu bộ lọc. Ba kết quả của C đều thuộc chương Thạc sĩ. A và B đều trả về ít nhất một kết quả ngoài phạm vi. Mẫu coherence dùng cùng một vị trí trên mỗi trang PDF 5–14 cho cả ba cấu hình. Metadata được đối chiếu với nội dung trang PDF và phạm vi gán nhãn. Grounding được chấm theo độ đủ của top-3 để trả lời từng câu test. 
## Phân tích và kết luận
- **A–B.** B tăng Evidence Hit@3 từ 7/20 lên 16/20 trên dev và từ 2/5 lên 5/5 trên test. Số chunk tăng từ 179 lên 445 và thời gian ingest của B dài hơn ở cả hai lần chạy.
- **B–C.** C tăng Evidence Hit@3 trên dev từ 16/20 lên 17/20. Trên test, hai cấu hình đều đạt 5/5. Riêng q21, bộ lọc của C giữ cả ba kết quả trong chương Thạc sĩ.
- **Hai ví dụ lỗi.** Với q24, A không trả về trang PDF 30 chứa mốc 12 tháng để hoàn thành tổng quan tài liệu. B và C tìm được bằng chứng ở hạng 1. Với q17, B có một chunk từ trang PDF 21 nhưng chunk đó thiếu trích đoạn về giới hạn 15 tín chỉ của chương trình Kỹ sư. C tìm được trích đoạn đầy đủ ở hạng 2.
- C dẫn đầu theo Evidence Hit@3 trên dev và đồng hạng với B trên test. Lợi ích lọc phụ thuộc vào trường `program_level` có giá trị đúng và câu hỏi chỉ định phạm vi.
- Đánh giá giới hạn trong một PDF, với 5 câu test và chỉ một câu test yêu cầu lọc. Điểm truy xuất tự động dựa vào trích đoạn khớp sau chuẩn hóa. Coherence và grounding có yếu tố chủ quan. Thời gian ingest phụ thuộc môi trường chạy và cách tái sử dụng vector. Chi phí theo cấu hình chưa được đo.
- Nếu mở rộng, bổ sung câu hỏi cần phân biệt các chương trình và rà soát các chunk ngắn chỉ chứa tiêu đề hoặc số trang.

