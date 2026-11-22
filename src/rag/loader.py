"""Leest games_raw.json, maakt opgeschoonde metadata en één Document per spel."""

import json
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from .config import EMPTY_VALUES, RAW_PATH, UNKNOWN_AGE

# Labels die in de zoektekst komen, in de volgorde waarin ze staan
DETAIL_FIELDS = [
    ("Leeftijd", "leeftijd"),
    ("Thema's", "themas"),
    ("Soort", "soort"),
    ("Terrein", "terrein"),
    ("Duur", "duur"),
    ("Spelers", "spelers"),
]


def clean_value(value: Any) -> str:
    """Zet elke waarde om naar een string. None en placeholders worden leeg, lijsten worden 'a, b, c'."""
    if value is None:
        return ""
    if isinstance(value, list):
        items = [clean_value(v) for v in value]
        return ", ".join(item for item in items if item)
    text = str(value).strip()
    return "" if text.lower() in EMPTY_VALUES else text


def parse_age(text: str) -> tuple[int, int]:
    """'van 12 tot 14 jaaar' -> (12, 14). Geen getal -> (UNKNOWN_AGE, UNKNOWN_AGE)."""
    numbers = [int(n) for n in re.findall(r"\d+", text)]
    if not numbers:
        return UNKNOWN_AGE, UNKNOWN_AGE
    if len(numbers) == 1:
        return numbers[0], numbers[0]
    return min(numbers[0], numbers[1]), max(numbers[0], numbers[1])


def clean_metadata(game: dict) -> dict[str, Any]:
    """Platte metadata: strings, plus min_age en max_age als integers."""
    meta_in = game.get("metadata") or {}
    min_age, max_age = parse_age(clean_value(meta_in.get("leeftijd")))

    result: dict[str, Any] = {
        "title": clean_value(game.get("title")),
        "source_url": clean_value(game.get("source_url")),
        "min_age": min_age,
        "max_age": max_age,
    }
    for key, value in meta_in.items():
        result[key] = clean_value(value)
    return result


def build_page_content(title: str, text: str, meta: dict[str, Any]) -> str:
    """Titel en metadata vooraan (past binnen de tokenlimiet), daarna de spelbeschrijving."""
    details = [f"{label}: {meta[key]}" for label, key in DETAIL_FIELDS if meta.get(key)]
    header = " | ".join([title] + details)
    return f"{header}\n{text}".strip()


def load_games(path: Path = RAW_PATH) -> list[Document]:
    """Eén Document per spel. Lege invoer wordt overgeslagen."""
    with open(path, encoding="utf-8") as f:
        games = json.load(f)

    documents: list[Document] = []
    for game in games:
        title = clean_value(game.get("title"))
        text = clean_value(game.get("text"))
        if not title and not text:
            continue

        metadata = clean_metadata(game)
        documents.append(
            Document(
                page_content=build_page_content(title, text, metadata),
                metadata=metadata,
            )
        )
    return documents