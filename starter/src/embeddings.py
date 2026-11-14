"""Embedding thật bằng Sentence Transformers — model local BAAI/bge-m3 (đã tải sẵn)."""

from __future__ import annotations

import logging
from typing import Protocol

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Model đa ngữ, dim=1024; phù hợp tiếng Việt trong báo cáo ESG.
DEFAULT_MODEL = "BAAI/bge-m3"


class Embedder(Protocol):
    """Hợp đồng chung: mọi chiến lược A/B/C dùng CÙNG một embedder."""

    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Trả về đúng 1 vector / 1 đoạn, giữ nguyên thứ tự đầu vào."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Vector câu hỏi — cùng không gian với document vectors."""
        ...


class MyEmbedder:
    """Adapter local cho BAAI/bge-m3 (không mock, không fallback vector giả).

    - Ưu tiên load từ HuggingFace cache đã tải (`local_files_only=True`).
    - Luôn L2-normalize để cosine distance trên pgvector ổn định.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        local_files_only: bool = True,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size

        logger.info(
            "Loading embedding model %s (local_files_only=%s)…",
            model_name,
            local_files_only,
        )
        try:
            # Model đã có trong ~/.cache/huggingface/hub/models--BAAI--bge-m3
            self.model = SentenceTransformer(
                model_name,
                local_files_only=local_files_only,
            )
        except Exception:
            if local_files_only:
                # Chỉ thử tải lại khi cache thiếu — vẫn là model thật, không mock
                logger.warning(
                    "Không load được %s từ cache; thử tải từ HuggingFace…",
                    model_name,
                )
                self.model = SentenceTransformer(model_name, local_files_only=False)
            else:
                raise

        # API mới: get_embedding_dimension; bản cũ: get_sentence_embedding_dimension
        if hasattr(self.model, "get_embedding_dimension"):
            self.dimensions = int(self.model.get_embedding_dimension())
        else:
            self.dimensions = int(self.model.get_sentence_embedding_dimension())

        if self.dimensions <= 0:
            raise RuntimeError(f"Model {model_name} trả về dimensions không hợp lệ: {self.dimensions}")

        logger.info("Embedder sẵn sàng: %s | dim=%s", self.model_name, self.dimensions)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Encode nhiều chunk; normalize_embeddings=True → ||v||=1 (cosine)."""
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 64,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        # Đảm bảo list[list[float]] thuần Python để ghi PostgreSQL
        return [row.tolist() for row in embeddings]

    def embed_query(self, question: str) -> list[float]:
        """Encode 1 câu hỏi — cùng normalize như documents."""
        if not question or not question.strip():
            raise ValueError("Câu hỏi rỗng — không embed được")

        vec = self.model.encode(
            question,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.tolist()
