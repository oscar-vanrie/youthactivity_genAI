"""
Startpunt van de RAG-pipeline. De stappen draai je apart:

    python -m src.rag.chunker        # stap 1: games_raw.json -> chunks.json
    python -m src.rag.vector_store   # stap 2: chunks.json -> ChromaDB (traag)
    python -m src.rag.rag_pipeline   # stap 3: test-zoekopdracht

Import door teamgenoot:
    from src.rag.rag_pipeline import search_games
"""

from typing import Any

from .config import CHROMA_DIR, CHUNKS_PATH, USE_HYBRID
from .search import search_games  # noqa: F401  (re-export voor import door anderen)


def print_results(query: str, hits: list[dict[str, Any]]) -> None:
    print(f"Zoekopdracht: '{query}'")
    print("-" * 60)
    if not hits:
        print("  (geen resultaten)\n")
        return
    for i, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        print(f"{i}. {hit['title']}  (score: {hit['score']:.4f})")
        print(f"   URL:      {hit['source_url']}")
        print(f"   Leeftijd: {meta.get('leeftijd', '')}")
        print(f"   Terrein:  {meta.get('terrein', '')}")
        print(f"   Themas:   {meta.get('themas', '')}")
        print()


if __name__ == "__main__":
    print(f"Chunks:   {CHUNKS_PATH}")
    print(f"Database: {CHROMA_DIR}")
    print(f"Modus:    {'hybride (semantisch + BM25)' if USE_HYBRID else 'alleen semantisch'}\n")

    print_results("spel over gevoelens", search_games("spel over gevoelens", k=3))