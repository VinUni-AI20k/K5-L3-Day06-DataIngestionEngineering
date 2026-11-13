# PostgreSQL + pgvector · Benchmark HUST

Pipeline đọc PDF quy chế HUST trong `data/corpus/` và 25 câu trong `../benchmark/questions.json`. A chia cửa sổ cố định, B chia theo cấu trúc văn bản, C dùng cùng chunk và vector của B rồi áp bộ lọc khi câu hỏi yêu cầu.

## Chạy trên Windows

Mở Docker Desktop. Từ thư mục `starter/`, dùng môi trường Python của bài lab và cài các thư viện cần thiết.

```powershell
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
docker compose -p day6_student_starter up -d --wait
python scripts/check_setup.py
```

Đặt API key và thông số embedding trong `.env`. Cấu hình mẫu dùng API tương thích OpenAI. Nếu chạy local, cài `sentence-transformers`, đặt `EMBEDDING_PROVIDER=local` và điền `EMBEDDING_MODEL` bằng tên model local. Chọn một model và dùng xuyên suốt cả hai tập.

```powershell
python -m src.run --validate-only
python -m src.run --split dev
```

`--validate-only` kiểm tra PDF, nguồn bằng chứng và số chunk mà không kết nối PostgreSQL hay API. Khi bỏ `--split`, lệnh lấy tập từ `EVALUATION_SPLIT` trong môi trường, mặc định là dev. Sau khi chốt cấu hình từ kết quả dev, dùng `python -m src.run --split test`. Kết quả lưu dưới `../results/` theo từng tập, gồm trace top-3, chỉ số và thống kê ingest. Dùng trace để chấm thủ công theo [GUIDE.md](../GUIDE.md).

Nếu lệnh kiểm tra PostgreSQL lỗi, kiểm tra Docker Desktop, cổng `55433` và `DATABASE_URL` trong `.env`. Có thể dừng container và giữ dữ liệu bằng `docker compose -p day6_student_starter stop`.
