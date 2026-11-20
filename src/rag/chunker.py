"""Stap 1: games_raw.json -> chunks.json (één opgeschoond document per spel)."""

import json

from .config import CHUNKS_PATH
from .loader import load_games


def save_chunks() -> int:
    """Schrijft alle chunks naar data/processed/chunks.json. Geeft het aantal terug."""
    documents = load_games()
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)

    payload = [
        {"id": f"game-{i}", "page_content": doc.page_content, "metadata": doc.metadata}
        for i, doc in enumerate(documents)
    ]
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return len(payload)


if __name__ == "__main__":
    n = save_chunks()
    print(f"{n} chunks opgeslagen in {CHUNKS_PATH}")