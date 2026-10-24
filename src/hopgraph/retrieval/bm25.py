"""Baseline A: BM25 lexical retrieval with ``bm25s``.

BM25 score of document d for query q, exactly as bm25s computes ``method="lucene"``
(verified in ``bm25s.scoring``; Kamphuis et al., ECIR 2020):

    score(q, d) = sum_{t in q} IDF(t) * tf(t,d) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
    IDF(t)      = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

The classic Robertson numerator factor (k1 + 1) is omitted; it scales every score
equally and so does not change rankings. ``k1`` controls term-frequency saturation;
``b`` controls document-length normalization.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import bm25s

from hopgraph.config import BM25Config
from hopgraph.types import Passage, RankedList

log = logging.getLogger(__name__)


class BM25Retriever:
    def __init__(self, cfg: BM25Config):
        self.cfg = cfg
        self._stemmer = None
        if cfg.stemmer:
            import Stemmer  # PyStemmer

            self._stemmer = Stemmer.Stemmer(cfg.stemmer)
        self.model: bm25s.BM25 | None = None
        self.ids: list[str] = []

    def _tokenize(self, texts: Sequence[str], return_ids: bool):
        return bm25s.tokenize(
            list(texts), lower=True, stopwords=self.cfg.stopwords, stemmer=self._stemmer,
            return_ids=return_ids, show_progress=False,
        )

    def index(self, passages: Sequence[Passage]) -> None:
        self.ids = [p.id for p in passages]
        tokens = self._tokenize([p.content for p in passages], return_ids=True)
        self.model = bm25s.BM25(k1=self.cfg.k1, b=self.cfg.b, method=self.cfg.method)
        self.model.index(tokens, show_progress=False)
        log.info("BM25 indexed %d passages (vocab %d)", len(self.ids), len(self.model.vocab_dict))

    def search(self, query: str, k: int | None = None) -> RankedList:
        assert self.model is not None, "index() or load() first"
        k = k or self.cfg.top_k
        toks = self._tokenize([query], return_ids=False)[0]
        toks = [t for t in toks if t in self.model.vocab_dict]
        if not toks:
            return RankedList([])
        return RankedList.from_array(self.model.get_scores(toks), self.ids, k)

    def save(self, out_dir: str | Path) -> None:
        assert self.model is not None
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save(str(out / "bm25s"))
        (out / "passage_ids.json").write_text(json.dumps(self.ids), encoding="utf-8")

    @classmethod
    def load(cls, cfg: BM25Config, in_dir: str | Path) -> BM25Retriever:
        r = cls(cfg)
        d = Path(in_dir)
        r.model = bm25s.BM25.load(str(d / "bm25s"))
        r.ids = json.loads((d / "passage_ids.json").read_text(encoding="utf-8"))
        if (r.model.k1, r.model.b) != (cfg.k1, cfg.b):
            raise ValueError(
                f"saved BM25 index has k1={r.model.k1}, b={r.model.b} but config asks for "
                f"k1={cfg.k1}, b={cfg.b}; rebuild the index"
            )
        return r
