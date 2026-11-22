"""
Handmatige beoordeling van zoekresultaten.

Per vraag toont het script de top 3 en vraagt jou of elk spel bruikbaar is als
basis voor een nieuwe activiteit: ja, deels of nee, met een korte reden.
Alles wordt opgeslagen in data/eval/manual_review.json, zodat je het later
kunt gebruiken voor het report.

Gebruik vanuit de rootmap:
    python -m src.rag.manual_review
"""

import json

from .config import BASE_DIR
from .search import search_games

REVIEW_PATH = BASE_DIR / "data" / "eval" / "manual_review.json"

QUERIES = [
    "spel over liefde",
    "spel met klakkebuizen en propjes schieten",
    "gevoel van paniek en angst in een spel",
    "spel waarbij leden geld verdienen en opdrachten kopen",
]

ANSWERS = {"j": "ja", "d": "deels", "n": "nee"}


def ask(prompt: str) -> str:
    while True:
        answer = input(prompt).strip().lower()
        if answer in ANSWERS:
            return ANSWERS[answer]
        print("  Typ j (ja), d (deels) of n (nee).")


def review_query(query: str) -> list[dict]:
    print("\n" + "=" * 70)
    print(f"Vraag: {query}")
    print("=" * 70)

    notes = []
    for i, hit in enumerate(search_games(query, k=3), start=1):
        meta = hit["metadata"]
        print(f"\n{i}. {hit['title']}")
        print(f"   Leeftijd: {meta.get('leeftijd', '')} | Terrein: {meta.get('terrein', '')}")
        print(f"   Themas:   {meta.get('themas', '')}")
        print(f"   Tekst:    {hit['content'][:400]}...")
        print(f"   URL:      {hit['source_url']}")

        verdict = ask("   Bruikbaar? (j/d/n): ")
        reason = input("   Reden (één zin): ").strip()
        notes.append({
            "title": hit["title"],
            "source_url": hit["source_url"],
            "verdict": verdict,
            "reason": reason,
        })
    return notes


def main() -> None:
    results = {}
    if REVIEW_PATH.exists():
        results = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    for query in QUERIES:
        results[query] = review_query(query)
        REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
        REVIEW_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nOpgeslagen in {REVIEW_PATH}")

    print("\nKlaar. Overzicht:")
    for query, notes in results.items():
        counts = {v: sum(n["verdict"] == v for n in notes) for v in ["ja", "deels", "nee"]}
        print(f"  {query[:50]:<50} ja={counts['ja']} deels={counts['deels']} nee={counts['nee']}")


if __name__ == "__main__":
    main()