"""Reciprocal Rank Fusion (Cormack, Clarke & Büttcher, SIGIR 2009).

    RRF(d) = sum_r  w_r / (k + rank_r(d))

``rank_r(d)`` is 1-based; a document absent from list r contributes nothing.
``k = 60`` is the paper's setting, used here as a starting point, not an optimum.
"""

from __future__ import annotations

from collections.abc import Sequence

from hopgraph.types import RankedList


def rrf(
    lists: Sequence[RankedList],
    k: int = 60,
    weights: Sequence[float] | None = None,
    depth: int | None = None,
) -> RankedList:
    """Fuse ranked lists. ``depth`` truncates each input list before fusion."""
    if weights is None:
        weights = [1.0] * len(lists)
    if len(weights) != len(lists):
        raise ValueError(f"{len(weights)} weights for {len(lists)} lists")
    if k < 0:
        raise ValueError("k must be non-negative")
    scores: dict[str, float] = {}
    for ranked, w in zip(lists, weights, strict=True):
        if w == 0:
            continue
        for rank, pid in enumerate(ranked.ids(depth), start=1):
            scores[pid] = scores.get(pid, 0.0) + w / (k + rank)
    return RankedList.from_scores(scores.items())


def fill_from(primary: RankedList, fallback: RankedList, n: int) -> RankedList:
    """Append fallback items not already in ``primary`` until ``n`` items exist.

    Appended items keep their fallback order; their scores are placed strictly
    below the primary list's minimum so the combined list remains ordered.
    """
    if len(primary) >= n:
        return primary.top(n)
    seen = set(primary.ids())
    items = list(primary.items)
    floor = min(primary.scores(), default=0.0)
    step = 1e-9
    for pid, _ in fallback:
        if len(items) >= n:
            break
        if pid not in seen:
            floor -= step
            items.append((pid, floor))
            seen.add(pid)
    return RankedList(items)
