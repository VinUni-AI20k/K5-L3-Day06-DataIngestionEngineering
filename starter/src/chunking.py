"""Hai chiến lược chunk: A = fixed-size + overlap; B = theo đoạn, có giới hạn kích thước."""

from __future__ import annotations

from .models import Chunk, DocumentPage


def _make_chunk(
    page: DocumentPage,
    strategy_tag: str,
    chunk_idx: int,
    text: str,
) -> Chunk:
    """Tạo Chunk với ID ổn định + metadata kế thừa từ trang (gồm pillar).

    strategy_tag:
      - "a" = fixed_size (config A)
      - "b" = by_paragraph (config B/C dùng cùng chunk, C chỉ thêm filter lúc search)
    """
    return Chunk(
        chunk_id=f"{page.document_id}_p{page.page}_{strategy_tag}_{chunk_idx}",
        document_id=page.document_id,
        source=page.source,
        page=page.page,
        text=text,
        # Copy dict để sau này sửa metadata chunk không làm đổi metadata page gốc
        metadata=dict(page.metadata),
    )


def fixed_size(page: DocumentPage, size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Strategy A — cắt theo số ký tự cố định, có overlap.

    Ví dụ GUIDE: size=500, overlap=50.
    Cố ý có thể cắt giữa câu → dùng để so sánh coherence với strategy B.
    """
    text = page.text
    if not text:
        return []

    # Bước trượt cửa sổ; overlap giúp evidence gần ranh giới không bị mất hết
    step = size - overlap
    if step <= 0:
        # Tránh vòng lặp vô hạn nếu người gọi truyền overlap >= size
        step = size

    chunks: list[Chunk] = []
    chunk_idx = 0
    for start in range(0, len(text), step):
        end = min(start + size, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(_make_chunk(page, "a", chunk_idx, chunk_text))
            chunk_idx += 1
        # Đã tới cuối trang → dừng (tránh chunk cuối trùng vì overlap)
        if end >= len(text):
            break
    return chunks


def _split_oversized(unit: str, max_size: int) -> list[str]:
    """Khi một đơn vị (đoạn/bullet) dài hơn max_size — cắt cứng theo ký tự.

    GUIDE yêu cầu chunk theo đoạn VẪN có giới hạn kích thước.
    Không overlap ở bước này để tránh nhân bản nội dung trong strategy B.
    """
    unit = unit.strip()
    if not unit:
        return []
    if len(unit) <= max_size:
        return [unit]
    parts: list[str] = []
    for start in range(0, len(unit), max_size):
        piece = unit[start : start + max_size].strip()
        if piece:
            parts.append(piece)
    return parts


def _paragraph_units(text: str) -> list[str]:
    """Tách trang thành các đơn vị ngữ nghĩa trước khi đóng gói chunk.

    PDF ESG thường:
      1) Có \\n\\n thật → dùng luôn.
      2) Không có dòng trống → thử tách theo dòng bắt đầu bằng '•' (bullet phổ biến).
      3) Vẫn chỉ 1 khối → giữ nguyên; bước đóng gói + _split_oversized sẽ giới hạn size.
    """
    text = text.strip()
    if not text:
        return []

    # Ưu tiên ranh giới đoạn chuẩn (sau normalize_text trong ingestion)
    blank_parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(blank_parts) > 1:
        return blank_parts

    # Fallback structure-aware: mỗi bullet ≈ một ý
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    units: list[str] = []
    buf: list[str] = []
    for ln in lines:
        # Gặp bullet mới thì đóng khối trước đó
        if ln.startswith("•") and buf:
            units.append("\n".join(buf))
            buf = [ln]
        else:
            buf.append(ln)
    if buf:
        units.append("\n".join(buf))

    return units


def by_paragraph(page: DocumentPage, max_size: int = 800) -> list[Chunk]:
    """Strategy B — gom theo đoạn/bullet, không vượt max_size.

    Config C tái dùng chunk của B; khác biệt chỉ ở metadata filter lúc truy xuất.
    """
    if not page.text.strip():
        return []

    units = _paragraph_units(page.text)
    # Mỗi unit dài quá → tách nhỏ trước, rồi mới đóng gói
    units = [piece for unit in units for piece in _split_oversized(unit, max_size)]

    chunks: list[Chunk] = []
    current_parts: list[str] = []
    current_len = 0
    chunk_idx = 0

    def flush() -> None:
        """Đóng chunk hiện tại (nếu có nội dung)."""
        nonlocal current_parts, current_len, chunk_idx
        if not current_parts:
            return
        chunk_text = "\n\n".join(current_parts).strip()
        if chunk_text:
            chunks.append(_make_chunk(page, "b", chunk_idx, chunk_text))
            chunk_idx += 1
        current_parts = []
        current_len = 0

    for unit in units:
        # +2 = độ dài ước lượng của "\\n\\n" khi join
        extra = len(unit) if not current_parts else len(unit) + 2
        if current_parts and current_len + extra > max_size:
            flush()
        current_parts.append(unit)
        current_len += extra

    flush()
    return chunks
