"""Đọc PDF, chuẩn hóa Unicode và gắn metadata truy vết (document/source/page/pillar)."""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from .models import DocumentPage

logger = logging.getLogger(__name__)

# Trang bìa/kết theo data/manifest.json — không tính vào 49 trang nội dung hợp lệ.
DEFAULT_EXCLUDED_PAGES = {1, 2, 52}


def get_pillar_by_page(page_num: int) -> str:
    """Gán trụ cột ESG theo nội dung chính của từng trang PDF.

    Mapping đối chiếu mục lục / nội dung Báo cáo ESG FPT 2025:
      - Environment: Net Zero (tr.7), tóm tắt môi trường (tr.12), chương môi trường (tr.37–46)
      - Social: nơi làm việc (tr.11), CBNV (tr.15), cộng đồng (tr.17),
                nhân sự/đào tạo/văn hóa (tr.30–36), CSR cộng đồng (tr.47–51)
      - Governance: phần còn lại (định hướng, quản trị, ISO, rủi ro, mua sắm…)

    Lưu ý: một số trang có nội dung lẫn (vd. tr.12 vừa môi trường vừa cộng đồng).
    Ta gán theo chủ đề thống trị / phục vụ filter benchmark — ghi rõ hạn chế này trong báo cáo.
    """
    # Environment: lộ trình Net Zero + khối “Hoạt động vì môi trường xanh”
    if page_num in {7, 12} or 37 <= page_num <= 46:
        return "Environment"

    # Social: môi trường làm việc, CBNV, cộng đồng / trách nhiệm xã hội
    if page_num in {11, 15, 17} or 30 <= page_num <= 36 or 47 <= page_num <= 51:
        return "Social"

    # Governance: thông điệp/định hướng, quản trị xuất sắc, rủi ro, chứng chỉ…
    return "Governance"


def normalize_text(text: str) -> str:
    """Chuẩn hóa text trước khi chunk/embed.

    - NFC: tiếng Việt ổn định (tránh 1 ký tự bị tách thành base + dấu).
    - Gom khoảng trắng trên từng dòng.
    - Giữ tối đa một dòng trống giữa các khối → dùng được cho chunk theo đoạn (`\\n\\n`).
    """
    if not text:
        return ""

    # 1) Unicode dựng sẵn (NFC) — yêu cầu GUIDE: chuẩn hóa Unicode
    text = unicodedata.normalize("NFC", text)

    # 2) Làm sạch từng dòng (không xóa dòng trống — dòng trống = ranh giới đoạn)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]

    # 3) Thu gọn 3+ dòng trống liên tiếp thành đúng 1 dòng trống (tức \\n\\n)
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def load_pdf(
    path: Path,
    document_id: str,
    source: str,
    excluded_pages: set[int] | None = None,
) -> list[DocumentPage]:
    """Đọc PDF thành list DocumentPage (một phần tử = một trang nội dung).

    - Bỏ trang excluded (mặc định bìa/kết trong manifest).
    - Log trang rỗng / không extract được (GUIDE: ghi lỗi, không nuốt im).
    - Mỗi trang có metadata.filterable: pillar ∈ {Environment, Social, Governance}.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy PDF: {path}")

    excluded = excluded_pages if excluded_pages is not None else DEFAULT_EXCLUDED_PAGES
    reader = PdfReader(str(path))
    pages: list[DocumentPage] = []
    empty_pages: list[int] = []
    skipped_excluded: list[int] = []

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1  # PDF 1-indexed, khớp gold_page trong benchmark

        # Loại trang không thuộc “nội dung hợp lệ” theo manifest
        if page_num in excluded:
            skipped_excluded.append(page_num)
            continue

        try:
            raw_text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001 — ghi log rồi bỏ trang lỗi, không crash cả pipeline
            logger.warning("Lỗi extract trang %s: %s", page_num, exc)
            empty_pages.append(page_num)
            continue

        clean_text = normalize_text(raw_text)
        if not clean_text:
            # Trang trắng / chỉ ảnh / OCR thất bại → ghi nhận để báo cáo
            empty_pages.append(page_num)
            logger.warning("Trang %s rỗng sau chuẩn hóa — bỏ qua", page_num)
            continue

        pillar = get_pillar_by_page(page_num)
        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=page_num,
                text=clean_text,
                # Field filter dùng cho strategy C; giá trị phải khớp nguồn (xem get_pillar_by_page)
                metadata={"pillar": pillar},
            )
        )

    logger.info(
        "load_pdf xong: %s trang nội dung | excluded=%s | rỗng/lỗi=%s",
        len(pages),
        skipped_excluded,
        empty_pages,
    )
    return pages
