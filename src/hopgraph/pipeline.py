"""Runs every retrieval system for one query, sharing stages and timing each one."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from hopgraph.config import Config
from hopgraph.evaluation.timing import StageTimer
from hopgraph.retrieval.bm25 import BM25Retriever
from hopgraph.retrieval.dense import DenseRetriever
from hopgraph.retrieval.fusion import rrf
from hopgraph.types import Passage, RankedList

# Stages each system needs, in execution order. A system's latency is the sum of its stages,
# because later stages consume earlier stages' output (graph retrieval starts from hybrid).
FIRST_STAGE = ["bm25", "dense_encode", "dense_search", "fusion"]
SYSTEM_STAGES: dict[str, list[str]] = {
    "bm25": ["bm25"],
    "dense": ["dense_encode", "dense_search"],
    "hybrid": FIRST_STAGE,
    "hybrid_ce": FIRST_STAGE + ["rerank"],
    "graph": FIRST_STAGE + ["link", "seed", "ppr", "graph_fusion"],
    "graph_only": FIRST_STAGE + ["link", "seed", "ppr", "graph_fusion"],
    "graph_ce": FIRST_STAGE + ["link", "seed", "ppr", "graph_fusion", "rerank_graph"],
}
GRAPH_STAGES = ["link", "seed", "ppr", "graph_fusion"]


def system_latency_ms(stage_ms: Mapping[str, float], system: str) -> float:
    return sum(stage_ms.get(s, 0.0) for s in SYSTEM_STAGES[system])


@dataclass
class QueryOutput:
    rankings: dict[str, RankedList]
    stage_ms: dict[str, float]
    info: dict = field(default_factory=dict)


class Pipeline:
    def __init__(
        self, cfg: Config, passages: Mapping[str, Passage], bm25: BM25Retriever,
        dense: DenseRetriever, reranker=None, graph=None,
    ):
        self.cfg = cfg
        self.passages = passages
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker
        self.graph = graph

    def run(self, query: str, systems: Iterable[str]) -> QueryOutput:
        systems = list(systems)
        cfg = self.cfg
        t = StageTimer()
        out: dict[str, RankedList] = {}
        info: dict = {}

        with t("bm25"):
            b = self.bm25.search(query, cfg.bm25.top_k)
        encoder = self.dense.encoder
        assert encoder is not None, "dense retriever needs a query encoder"
        with t("dense_encode"):
            qv = encoder.encode_queries([query])
        with t("dense_search"):
            d = self.dense.search_vectors(qv, cfg.dense.top_k)[0]
        with t("fusion"):
            h = rrf([b, d], k=cfg.fusion.k, weights=cfg.fusion.weights, depth=cfg.fusion.depth)
            h = h.top(cfg.fusion.depth)
        out["bm25"], out["dense"], out["hybrid"] = b, d, h

        if "hybrid_ce" in systems:
            with t("rerank"):
                out["hybrid_ce"] = self.reranker.rerank(query, h, self.passages, cfg.rerank.depth)

        if {"graph", "graph_only", "graph_ce"} & set(systems):
            g = self.graph.retrieve(query, h, t)
            out["graph"], out["graph_only"] = g.fused, g.graph_only
            info.update(g.info)
            if "graph_ce" in systems:
                with t("rerank_graph"):
                    out["graph_ce"] = self.reranker.rerank(
                        query, g.fused, self.passages, cfg.rerank.depth
                    )

        return QueryOutput({s: out[s] for s in systems}, t.ms(), info)
