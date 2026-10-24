"""Baseline D: cross-encoder reranking of the top candidates."""

from __future__ import annotations

from collections.abc import Mapping

from hopgraph.config import RerankConfig
from hopgraph.types import Passage, RankedList


class CrossEncoderReranker:
    def __init__(self, cfg: RerankConfig, device: str = "cpu"):
        from sentence_transformers import CrossEncoder

        self.cfg = cfg
        self.model = CrossEncoder(
            cfg.model, revision=cfg.revision, device=device, max_length=cfg.max_length
        )

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        out = self.model.predict(
            [(query, t) for t in texts], batch_size=self.cfg.batch_size, show_progress_bar=False
        )
        return [float(s) for s in out]

    def rerank(
        self, query: str, candidates: RankedList, passages: Mapping[str, Passage],
        depth: int | None = None,
    ) -> RankedList:
        """Rescore the top ``depth`` candidates; the tail keeps its original order below them."""
        depth = depth or self.cfg.depth
        head = candidates.ids(depth)
        scores = self.score(query, [passages[pid].content for pid in head])
        reranked = RankedList.from_scores(zip(head, scores, strict=True))
        floor = min(scores, default=0.0)
        tail = [(pid, floor - 1.0 - i * 1e-6) for i, pid in enumerate(candidates.ids()[depth:])]
        return RankedList(reranked.items + tail)
