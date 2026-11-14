# Audit thủ công — Day 06

Người chấm: Lê Thanh Hà (tác giả bài làm) — có nguy cơ thiên lệch vì tự chấm.
Nguồn: `starter/results/traces.json`, `run_meta.json`, tái tạo chunk từ `src/chunking.py`.
Ngày chấm: 2026-11-14.

---

## 1) Chunk coherence (thang 0–2)

**Cách chọn mẫu (trước khi nhìn điểm Hit):** 10 vị trí = điểm giữa text của 10 page ID đầu đã sắp xếp trong corpus hợp lệ: trang **3→12**. Với mỗi strategy, lấy chunk bao phủ vị trí giữa trang đó.

| Vị trí (page) | A điểm | A lý do ngắn | B/C điểm | B/C lý do ngắn |
|---|---:|---|---:|---|
| 3 | 0 | Cắt giữa câu (“và kiểm kê…”), thiếu chủ thể đoạn trước | 2 | Giữ trọn ý cam kết chứng nhận ESG / mua sắm bền vững |
| 4 | 1 | Dải KPI/ISO, đọc được một phần nhưng mất tiêu đề khung | 1 | Có mở đầu “Nơi làm việc…” nhưng vẫn dày số liệu, ít ngữ cảnh câu |
| 5 | 0 | Cắt giữa câu giới thiệu GRI | 2 | Đoạn mở đầu báo cáo ESG năm thứ 05 tự đứng được |
| 6 | 0 | Mất đầu câu (“ốn trục nội dung…”) | 2 | Có nhãn “Định hướng ESG” + đủ bốn trục |
| 7 | 1 | Có Net Zero nhưng checklist bước 02–03 hơi rời | 2 | Có baseline 15,8%/2040 trước khi liệt kê bước |
| 8 | 1 | Cam kết NCC đọc được nhưng bắt đầu giữa danh sách | 1 | Khối KPI môi trường văn phòng, thiếu câu chủ đề |
| 9 | 1 | Bullet bảo vệ dữ liệu đủ ý cục bộ | 1 | Bullet năng lượng/AI, hiểu được nhưng không có khung mục |
| 10 | 1 | Bullet ATTT đủ ý | 1 | Đoạn đầu lẫn mục lục điều hướng báo cáo → nhiễu |
| 11 | 1 | Chuỗi bullet phúc lợi, thiếu tiêu đề mục | 1 | Bullet “Thung lũng Silicon”, đủ ý cục bộ |
| 12 | 1 | Lạc sang ý cộng đồng/đào tạo giữa trang lẫn E+S | 2 | Có tiêu đề “HOẠT ĐỘNG VÌ MÔI TRƯỜNG XANH…” |

**Tổng / trung bình**

| Strategy | Tổng ( /20 ) | Trung bình /2 (n=10) |
|---|---:|---:|
| A | 7 | **0.70** |
| B | 15 | **1.50** |
| C | 15 | **1.50** (cùng chunk với B) |

---

## 2) Metadata usefulness (5 câu test × top-3)

Mỗi bản ghi kiểm tra: có đủ `document_id`, `source`, `page`, `pillar` và giá trị khớp mapping trang trong `data/manifest.json` / `get_pillar_by_page`.

| Strategy | Bản ghi kiểm tra | Đủ & đúng | Lỗi ghi nhận |
|---|---:|---:|---|
| A | 15 | **15/15** | Không |
| B | 15 | **15/15** | Không |
| C | 15 | **15/15** | Không |

**Filter tuân thủ** (câu test có filter: chỉ **q17**, `pillar=Environment`; yêu cầu cả top-3 cùng pillar):

| Strategy | Tuân thủ filter (câu) | Chi tiết top-3 pillar q17 |
|---|---|---|
| A | **0/1** | Environment, Social, Social |
| B | **0/1** | Environment, Social, Social |
| C | **1/1** | Environment, Environment, Environment |

**Hữu dụng:** `pillar` giúp Config C loại nhiễu Social khi hỏi trồng rừng (q17).  
**Hạn chế:** gán pillar theo khoảng trang — trang lẫn chủ đề (vd. tr.12) chỉ nhận một nhãn; filter không sửa được lỗi semantic (q19 vẫn miss dù metadata đúng).

---

## 3) Grounding quality — 5 câu test (thang 0–2 trên top-3)

0 = không có bằng chứng trả lời; 1 = có liên quan nhưng thiếu điều kiện/số liệu; 2 = đủ để trả lời và truy nguồn.  
**Không** đồng nhất với độ đúng câu trả lời LLM.

| QID | Gold page | A | Lý do A | B | Lý do B | C | Lý do C |
|---|---:|---:|---|---:|---|---:|---|
| q16 | 9 | 2 | Rank3 có “Top 5… ASEAN” | 0 | Top-3 lệch sang nơi làm việc / nhân sự, không có Top 5 | 0 | Giống B (không filter) |
| q17 | 41 | 2 | Rank1 đủ 4.500 cây / địa điểm | 2 | Rank1 đủ evidence | 2 | Rank1 đủ; top-3 sạch Environment |
| q18 | 51 | 2 | Rank1 có World Vision + Bếp Xanh + Sín Chải | 2 | Rank1 đủ evidence | 2 | Giống B |
| q19 | 8 | 1 | Top-3 nói mua sắm/NCC/đạo đức nhưng **thiếu** câu “Không ký hợp đồng…” | 1 | Tương tự — gần chủ đề, thiếu điều kiện cấm ký HĐ | 1 | Giống B |
| q20 | 11 | 1 | Top-3 toàn tr.31 (giờ học AI/MiniMBA), thiếu 7,1 triệu / 151 giờ | 2 | Rank3 tr.11 chứa “7,1 triệu… 151 giờ” | 2 | Giống B |

**Phân bố & trung bình /2**

| Strategy | Số câu mức 0 | mức 1 | mức 2 | Trung bình /2 |
|---|---:|---:|---:|---:|
| A | 0 | 2 | 3 | **1.60** |
| B | 1 | 1 | 3 | **1.40** |
| C | 1 | 1 | 3 | **1.40** |
