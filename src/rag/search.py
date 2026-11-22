"""
Zoekfunctie voor het Multi-Agent systeem.

Standaard hybride: semantisch (ChromaDB) + keyword (BM25), samengevoegd met
Reciprocal Rank Fusion (RRF). Met USE_HYBRID = False alleen semantisch.
"""

import re
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from .config import CANDIDATE_POOL, UNKNOWN_AGE, USE_HYBRID
from .vector_store import get_vector_store, load_chunks

RRF_K = 60  # standaardwaarde uit de RRF-literatuur


def tokenize(text: str) -> list[str]:
    """Kleine tokenizer: kleine letters, alleen woorden. Geen stopwoorden."""
    return re.findall(r"\w+", text.lower())


@lru_cache(maxsize=1)
def _documents() -> list[Document]:
    """Alle chunks uit data/processed/chunks.json, één keer per proces ingelezen."""
    return load_chunks()


@lru_cache(maxsize=1)
def _bm25_index() -> BM25Okapi:
    """BM25-index op dezelfde chunks als de ChromaDB-database."""
    return BM25Okapi([tokenize(doc.page_content) for doc in _documents()])


@lru_cache(maxsize=1)
def _by_url() -> dict[str, Document]:
    return {doc.metadata["source_url"]: doc for doc in _documents()}


def _matches(
    meta: dict[str, Any],
    min_age: int | None,
    max_age: int | None,
    terrein: str | None,
    strict_age: bool = True,
) -> bool:
    """
    Leeftijdsfilter. Met strict_age=True moet het spel volledig binnen de
    gevraagde leeftijd vallen (niet alleen overlappen).
    """
    if min_age is not None or max_age is not None:
        game_min = meta.get("min_age", UNKNOWN_AGE)
        game_max = meta.get("max_age", UNKNOWN_AGE)
        if game_min < 0:
            return False  # leeftijd onbekend: niet tonen
        if strict_age:
            if min_age is not None and game_min < min_age:
                return False
            if max_age is not None and game_max > max_age:
                return False
        else:
            if min_age is not None and game_max < min_age:
                return False
            if max_age is not None and game_min > max_age:
                return False

    if terrein and terrein.lower() not in meta.get("terrein", "").lower():
        return False

    return True


def _semantic_ranking(query: str) -> list[str]:
    """source_urls gesorteerd op semantische gelijkenis (hoogste eerst)."""
    store = get_vector_store()
    hits = store.similarity_search_with_score(query, k=CANDIDATE_POOL)
    return [doc.metadata["source_url"] for doc, _ in hits]


def _keyword_ranking(query: str) -> list[str]:
    """source_urls gesorteerd op BM25-score (hoogste eerst)."""
    docs = _documents()
    scores = _bm25_index().get_scores(tokenize(query))
    order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    return [docs[i].metadata["source_url"] for i in order[:CANDIDATE_POOL]]


def _rrf(rankings: list[list[str]]) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score = som van 1 / (RRF_K + rang) over alle rangschikkingen."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, url in enumerate(ranking, start=1):
            scores[url] = scores.get(url, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def search_games(
    query: str,
    k: int = 3,
    min_age: int | None = None,
    max_age: int | None = None,
    terrein: str | None = None,
    strict_age: bool = True,
) -> list[dict[str, Any]]:
    """
    Zoekt de k meest relevante spellen, optioneel gefilterd.

    Returns: lijst van dicts met title, source_url, content, metadata, score.
    Een hogere score betekent een betere match.
    """
    if USE_HYBRID:
        ranking = _rrf([_semantic_ranking(query), _keyword_ranking(query)])
    else:
        ranking = [(url, 0.0) for url in _semantic_ranking(query)]

    by_url = _by_url()
    results: list[dict[str, Any]] = []
    for url, score in ranking:
        doc = by_url.get(url)
        if doc is None or not _matches(doc.metadata, min_age, max_age, terrein, strict_age=strict_age):
            continue
        results.append(
            {
                "title": doc.metadata.get("title", ""),
                "source_url": url,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score,
            }
        )
        if len(results) == k:
            break
    return results