"""Retrieval metrics with binary relevance. All are per-question; average them for macro scores.

For question q with gold set G and ranked list L:

    Recall@k   = |G ∩ top_k(L)| / |G|
    AllGold@k  = 1 if G ⊆ top_k(L) else 0
    RR         = 1 / rank of the first gold passage (0 if none in L)
    nDCG@k     = DCG@k / IDCG@k,   DCG@k = sum_{i<=k} rel_i / log2(i + 1)
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence


def recall_at_k(ranked: Sequence[str], gold: Collection[str], k: int) -> float:
    if not gold:
        raise ValueError("gold set is empty")
    return len(set(ranked[:k]) & set(gold)) / len(set(gold))


def all_gold_at_k(ranked: Sequence[str], gold: Collection[str], k: int) -> float:
    if not gold:
        raise ValueError("gold set is empty")
    return float(set(gold) <= set(ranked[:k]))


def reciprocal_rank(ranked: Sequence[str], gold: Collection[str]) -> float:
    g = set(gold)
    for i, pid in enumerate(ranked, start=1):
        if pid in g:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gold: Collection[str], k: int) -> float:
    g = set(gold)
    if not g:
        raise ValueError("gold set is empty")
    dcg = sum(1.0 / math.log2(i + 1) for i, pid in enumerate(ranked[:k], start=1) if pid in g)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(g)) + 1))
    return dcg / idcg


def evaluate_ranking(ranked: Sequence[str], gold: Collection[str], ks: Sequence[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(ranked, gold, k)
        out[f"allgold@{k}"] = all_gold_at_k(ranked, gold, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranked, gold, k)
    out["rr"] = reciprocal_rank(ranked, gold)
    return out
