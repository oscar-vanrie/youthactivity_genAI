"""
Schrijft de volledige tekst van de top 3 per query naar een bestand.

Gebruik vanuit de rootmap:
    python -m src.rag.export_full_text
"""

from .config import BASE_DIR
from .manual_review import QUERIES
from .search import search_games

OUT_PATH = BASE_DIR / "data" / "eval" / "full_text_top3.txt"


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for query in QUERIES:
            f.write("=" * 70 + "\n")
            f.write(f"VRAAG: {query}\n")
            f.write("=" * 70 + "\n\n")
            for i, hit in enumerate(search_games(query, k=3), start=1):
                meta = hit["metadata"]
                f.write(f"--- {i}. {hit['title']} ---\n")
                f.write(f"URL: {hit['source_url']}\n")
                f.write(f"Leeftijd: {meta.get('leeftijd', '')} | Terrein: {meta.get('terrein', '')}\n\n")
                f.write(hit["content"] + "\n\n")
    print(f"Geschreven naar {OUT_PATH}")


if __name__ == "__main__":
    main()