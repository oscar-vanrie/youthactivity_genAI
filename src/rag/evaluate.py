"""
Evalueert de zoekfunctie met een handmatig gelabelde testset.

Metrics:
- Hit@k: zit er minstens één relevant spel in de top k?
- MRR: hoe hoog staat het eerste relevante spel gemiddeld?

Gebruik vanuit de rootmap:
    python -m src.rag.evaluate
"""

import json

from .config import EVAL_PATH, USE_HYBRID
from .search import search_games


def evaluate(k: int = 3) -> dict[str, float]:
    with open(EVAL_PATH, encoding="utf-8") as f:
        eval_set = json.load(f)

    modus = "hybride (semantisch + BM25)" if USE_HYBRID else "alleen semantisch"
    print(f"Modus: {modus}\n")
    print(f"{'Query':<50} {'Hit':>4} {'RR':>6}")
    print("-" * 62)

    hits = 0
    reciprocal_ranks = 0.0

    for item in eval_set:
        relevant = set(item["relevant_urls"])
        results = search_games(item["query"], k=k, **item.get("filters", {}))
        urls = [r["source_url"] for r in results]

        ranks = [i for i, url in enumerate(urls) if url in relevant]
        hit = bool(ranks)
        rr = 1 / (ranks[0] + 1) if hit else 0.0

        hits += hit
        reciprocal_ranks += rr

        print(f"{item['query'][:50]:<50} {'ja' if hit else 'nee':>4} {rr:>6.2f}")

    n = len(eval_set)
    scores = {f"Hit@{k}": hits / n, "MRR": reciprocal_ranks / n}
    print("-" * 62)
    for name, value in scores.items():
        print(f"{name}: {value:.3f}")
    return scores


if __name__ == "__main__":
    evaluate(k=3)