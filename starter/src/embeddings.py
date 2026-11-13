"""Local multilingual embedding model via sentence-transformers (no API key, no mock fallback)."""
from typing import Protocol

DEFAULT_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class SentenceTransformerEmbedder:
    """Loads the model once. Raises on any failure; never returns placeholder vectors."""

    def __init__(self, model_name: str = DEFAULT_MODEL, max_seq_length: int = 256) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        # The model was trained on short inputs; 256 tokens covers every chunk this lab creates.
        self._model.max_seq_length = max_seq_length
        self.dimensions = self._model.get_sentence_embedding_dimension()
        if not self.dimensions:
            raise RuntimeError(f'could not read embedding dimension from {model_name}')

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                                     show_progress_bar=False)
        result = [list(map(float, v)) for v in vectors]
        if any(len(v) != self.dimensions for v in result):
            raise RuntimeError('embedding dimension mismatch')
        return result

    def count_tokens(self, texts: list[str]) -> list[int]:
        """Token counts under the model's own tokenizer (inputs beyond max_seq_length are truncated)."""
        encoded = self._model.tokenizer(texts, add_special_tokens=True, truncation=False, padding=False)
        return [len(ids) for ids in encoded['input_ids']]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode(texts)

    def embed_query(self, question: str) -> list[float]:
        return self._encode([question])[0]
