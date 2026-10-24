"""Shared fixtures and lightweight stubs so CI never downloads models."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
TINY = FIXTURES / "tiny_corpus" / "musique_tiny.jsonl"


class HashEncoder:
    """Deterministic bag-of-words hashing encoder (unit vectors), a stand-in for BGE."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"\w+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 8) % 2 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v + 1.0 / np.sqrt(self.dim)

    def encode_passages(self, texts):
        return np.stack([self._vec(t) for t in texts]).astype(np.float32)

    def encode_queries(self, texts):
        return self.encode_passages(texts)


@dataclass
class _Span:
    text: str
    label_: str


class _Doc:
    def __init__(self, text: str):
        self.text = text
        # "Entities": maximal runs of capitalized words (a crude NER stand-in).
        self.ents = [_Span(m.group(0), "ORG")
                     for m in re.finditer(r"(?:[A-Z][\w'-]*)(?:\s+[A-Z][\w'-]*)*", text)
                     if m.group(0) not in {"What", "Who", "Where", "Which", "The"}]

    @property
    def noun_chunks(self):
        return [_Span(w, "") for w in re.findall(r"\b[a-z]{4,}\b", self.text)]


class StubNLP:
    def __call__(self, text: str) -> _Doc:
        return _Doc(text)

    def pipe(self, texts, batch_size: int = 128, n_process: int = 1):
        for t in texts:
            yield _Doc(t)


class StubReranker:
    """Scores by query-token overlap; same interface as CrossEncoderReranker.rerank."""

    def rerank(self, query, candidates, passages, depth=None):
        from hopgraph.types import RankedList

        depth = depth or 20
        q = set(re.findall(r"\w+", query.lower()))
        head = candidates.ids(depth)
        scored = [(pid, float(len(q & set(re.findall(r"\w+", passages[pid].content.lower())))))
                  for pid in head]
        ranked = RankedList.from_scores(scored)
        tail = [(pid, -1.0 - i) for i, pid in enumerate(candidates.ids()[depth:])]
        return RankedList(ranked.items + tail)


@pytest.fixture
def tiny_path() -> Path:
    return TINY


@pytest.fixture
def hash_encoder() -> HashEncoder:
    return HashEncoder()


@pytest.fixture
def stub_nlp() -> StubNLP:
    return StubNLP()
