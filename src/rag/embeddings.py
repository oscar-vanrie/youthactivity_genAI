"""Het gratis, lokale embedding-model. multilingual-e5 verwacht prefixen."""

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from .config import EMBEDDING_MODEL


class E5Embeddings(HuggingFaceEmbeddings):
    """Voegt 'passage: ' toe bij documenten en 'query: ' bij zoekopdrachten."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents([f"passage: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(f"query: {text}")


@lru_cache(maxsize=1)
def get_embeddings() -> E5Embeddings:
    """Laadt het model maar één keer per proces."""
    return E5Embeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )