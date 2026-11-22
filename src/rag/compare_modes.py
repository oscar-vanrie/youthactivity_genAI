"""
Vergelijkt semantisch en hybride zoeken per query.

Per query zie je: de rang van het relevante spel in beide modi, de overlap
van de top 10, en of de top 3 veranderd is. Zo zie je of BM25 echt iets doet.

Gebruik vanuit de rootmap:
    python -m src.rag.compare_modes
"""

import json

from .config import EVAL_PATH
from .search import _keyword_ranking, _rrf, _semantic_ranking


def rank_of(urls: list[str], relevant: set[str]) -> int | None:
    for i, url in enumerate(urls, start=1):
        if url in relevant:
            return i
    return None


def main() -> None:
    with open(EVAL_PATH, encoding="utf-8") as f:
        eval_set = json.load(f)

    print(f"{'Query':<45} {'sem':>4} {'hyb':>4} {'overlap10':>9} {'top3 gelijk':>11}")
    print("-" * 80)

    changed_top3 = 0
    for item in eval_set:
        query = item["query"]
        relevant = set(item["relevant_urls"])

        semantic = _semantic_ranking(query)
        keyword = _keyword_ranking(query)
        hybrid = [url for url, _ in _rrf([semantic, keyword])]

        overlap = len(set(semantic[:10]) & set(hybrid[:10]))
        same_top3 = semantic[:3] == hybrid[:3]
        changed_top3 += not same_top3

        sem_rank = rank_of(semantic, relevant)
        hyb_rank = rank_of(hybrid, relevant)
        print(
            f"{query[:45]:<45} {str(sem_rank):>4} {str(hyb_rank):>4} "
            f"{overlap:>9} {'ja' if same_top3 else 'nee':>11}"
        )

    print("-" * 80)
    print(f"Queries waarin de top 3 verandert door BM25: {changed_top3} van {len(eval_set)}")


if __name__ == "__main__":
    main()