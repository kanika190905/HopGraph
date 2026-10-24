"""Core data types shared by every stage of the pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Passage:
    """One searchable unit of text. ``id`` is the canonical, deduplicated ID."""

    id: str
    title: str
    text: str
    source: str = ""

    @property
    def content(self) -> str:
        """Text that BM25, the encoder, the reranker and spaCy all see."""
        return f"{self.title}\n{self.text}" if self.title else self.text


@dataclass(frozen=True)
class Question:
    """A benchmark question with canonical gold passage IDs."""

    id: str
    text: str
    gold_ids: tuple[str, ...]
    hops: int
    split: str
    answer: str = ""
    meta: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)


@dataclass
class RankedList:
    """Ordered ``(passage_id, score)`` pairs, best first. IDs are unique."""

    items: list[tuple[str, float]] = field(default_factory=list)

    @classmethod
    def from_scores(cls, pairs: Iterable[tuple[str, float]], k: int | None = None) -> RankedList:
        """Sort by score descending, breaking ties by passage ID so output is deterministic."""
        ordered = sorted(pairs, key=lambda p: (-p[1], p[0]))
        return cls(ordered[:k] if k is not None else ordered)

    @classmethod
    def from_array(cls, scores: Any, ids: list[str], k: int) -> RankedList:
        """Top-k strictly positive entries of a dense score array, ties broken by ID."""
        import numpy as np

        pos = np.flatnonzero(scores > 0)
        if pos.size > k:
            # Keep everything tied with the k-th score so the ID tie-break is exact.
            kth = np.partition(scores[pos], -k)[-k]
            pos = pos[scores[pos] >= kth]
        return cls.from_scores(((ids[i], float(scores[i])) for i in pos), k)

    def ids(self, k: int | None = None) -> list[str]:
        return [pid for pid, _ in (self.items[:k] if k is not None else self.items)]

    def scores(self, k: int | None = None) -> list[float]:
        return [s for _, s in (self.items[:k] if k is not None else self.items)]

    def top(self, k: int) -> RankedList:
        return RankedList(self.items[:k])

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[tuple[str, float]]:
        return iter(self.items)
