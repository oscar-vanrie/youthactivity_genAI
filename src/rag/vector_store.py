"""Stap 2: chunks.json -> ChromaDB. Dit is het trage deel (embedden)."""

import json
import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import CHROMA_DIR, CHUNKS_PATH, COLLECTION_NAME
from .embeddings import get_embeddings


def load_chunks() -> list[Document]:
    """Leest data/processed/chunks.json. Vereist dat stap 1 eerst gedraaid is."""
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError("Draai eerst: python -m src.rag.chunker")
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        payload = json.load(f)
    return [
        Document(page_content=c["page_content"], metadata=c["metadata"])
        for c in payload
    ]


def build_vector_store() -> Chroma:
    """Embedt alle chunks en slaat ze op in ChromaDB. Oude database wordt vervangen."""
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        ids = [c["id"] for c in json.load(f)]
    documents = load_chunks()

    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    return Chroma.from_documents(
        documents=documents,
        embedding=get_embeddings(),
        ids=ids,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )


def get_vector_store() -> Chroma:
    """Opent de bestaande database. Bouwt er alleen een als die nog niet bestaat."""
    if not CHROMA_DIR.exists():
        return build_vector_store()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


if __name__ == "__main__":
    build_vector_store()
    print(f"Database opgeslagen in {CHROMA_DIR}")