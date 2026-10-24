"""Graph-enhanced retrieval: link -> seed -> PPR -> passage ranking -> RRF with hybrid."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hopgraph.config import GraphFusionConfig, PPRConfig, SeedingConfig
from hopgraph.evaluation.timing import StageTimer
from hopgraph.graph.build import EntityGraph
from hopgraph.graph.linking import LinkResult, QueryLinker
from hopgraph.graph.ppr import ppr
from hopgraph.retrieval.fusion import fill_from, rrf
from hopgraph.types import RankedList

log = logging.getLogger(__name__)


@dataclass
class GraphOutput:
    graph_only: RankedList  # PPR passage ranking, filled from hybrid
    fused: RankedList  # RRF(hybrid, PPR), filled from hybrid
    info: dict = field(default_factory=dict)


class GraphRetriever:
    def __init__(
        self, graph: EntityGraph, linker: QueryLinker, seeding: SeedingConfig, ppr_cfg: PPRConfig,
        fusion: GraphFusionConfig,
    ):
        self.g = graph
        self.linker = linker
        self.seeding = seeding
        self.ppr_cfg = ppr_cfg
        self.fusion = fusion
        self._pid_to_node = {pid: i for i, pid in enumerate(graph.passage_ids)}
        self._spec = graph.specificity
        _ = graph.transition  # build once, outside query timing

    def seed_vector(self, link: LinkResult, hybrid: RankedList) -> tuple[np.ndarray, str]:
        P, n = self.g.n_passages, self.g.n_nodes
        ent = np.zeros(n)
        for le in link.entities:
            w = le.sim * (self._spec[le.entity] if self.seeding.use_specificity else 1.0)
            ent[P + le.entity] += w
        pas = np.zeros(n)
        use_passages = self.seeding.always_passage_seed or ent.sum() == 0
        if use_passages and self.seeding.passage_seed_m > 0:
            for pid, score in hybrid.items[: self.seeding.passage_seed_m]:
                node = self._pid_to_node.get(pid)
                if node is not None:
                    pas[node] += max(score, 0.0)
        has_e, has_p = ent.sum() > 0, pas.sum() > 0
        if has_e and has_p:
            beta = self.seeding.passage_seed_beta
            return (1 - beta) * ent / ent.sum() + beta * pas / pas.sum(), "entities+passages"
        if has_e:
            return ent / ent.sum(), "entities"
        if has_p:
            return pas / pas.sum(), "passages"
        return ent, "empty"

    def passage_scores(self, p: np.ndarray) -> np.ndarray:
        P = self.g.n_passages
        if self.fusion.passage_score == "direct":
            return p[:P]
        # HippoRAG-1 style: sum of entity scores weighted by mention matrix.
        return np.asarray(self.g.PE @ p[P:]).ravel()

    def retrieve(
        self, query: str, hybrid: RankedList, timer: StageTimer, link: LinkResult | None = None,
    ) -> GraphOutput:
        """``link`` may be precomputed (tuning grids reuse one linking pass per query)."""
        depth = self.fusion.depth
        with timer("link"):
            if link is None:
                link = self.linker.link(query)
        with timer("seed"):
            s, seed_kind = self.seed_vector(link, hybrid)
        info: dict[str, Any] = {
            "link_path": link.path,
            "linked": [[self.g.entity_keys[e.entity], e.method, round(e.sim, 4)] for e in link.entities],
            "seed_kind": seed_kind,
        }
        if seed_kind == "empty":
            info.update(ppr_iter=0, ppr_converged=None)
            fallback = hybrid.top(depth)
            return GraphOutput(fallback, fallback, info)
        with timer("ppr"):
            res = ppr(self.g.transition, s, self.ppr_cfg.alpha, self.ppr_cfg.tol, self.ppr_cfg.max_iter)
        info.update(ppr_iter=res.n_iter, ppr_converged=res.converged)
        if not res.converged:
            log.debug("PPR did not converge for %r (delta %.2e)", query, res.l1_delta)
        with timer("graph_fusion"):
            graph_rank = RankedList.from_array(
                self.passage_scores(res.scores), self.g.passage_ids, depth
            )
            info["graph_candidates"] = len(graph_rank)
            graph_only = fill_from(graph_rank, hybrid, depth)
            fused = rrf([hybrid, graph_rank], k=self.fusion.k, weights=self.fusion.weights,
                        depth=depth)
            fused = fill_from(fused.top(depth), hybrid, depth)
        return GraphOutput(graph_only, fused, info)
