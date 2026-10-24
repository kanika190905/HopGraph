"""Tune baseline and graph hyper-parameters on a TRAIN-derived set, with equal care for both.

    python scripts/tune.py --config configs/musique_tune500.yaml

Protocol (fixed before any tuning result was seen):
  1. BM25 (k1, b) grid, objective = hybrid R@5 with default RRF.
  2. RRF (k, dense weight) grid with the chosen BM25, objective = hybrid R@5.
  3. Cross-encoder depth in {20, 50, 100}: choose the *smallest* depth whose hybrid_ce R@5
     is within 0.5 points of the best (deeper reranking costs ~linearly more latency).
  4. Graph grid (alpha, passage-seed beta, passage-seed m, graph RRF weight) on top of the
     tuned first stage, objective = graph_ce R@5 at the chosen CE depth (graph R@5 also logged).
Ties keep the default. CE scores are cached per (question, passage), so every depth and
every graph configuration is scored with the same model outputs.

Writes results/tuning_<name>/tuning.json and configs/frozen/<name>_frozen.yaml (overrides only).
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hopgraph.artifacts import dense_dir, entity_emb_path, graph_dir  # noqa: E402
from hopgraph.config import load_config  # noqa: E402
from hopgraph.evaluation.memory import RSSMonitor  # noqa: E402
from hopgraph.evaluation.metrics import recall_at_k  # noqa: E402
from hopgraph.evaluation.timing import StageTimer, pin_threads  # noqa: E402
from hopgraph.ingestion.corpus import corpus_hash, load_corpus  # noqa: E402
from hopgraph.retrieval.fusion import rrf  # noqa: E402
from hopgraph.runinfo import git_commit, setup_logging  # noqa: E402
from hopgraph.types import RankedList  # noqa: E402

log = logging.getLogger("tune")

K1 = [0.9, 1.2, 1.5, 2.0]
B = [0.4, 0.6, 0.75, 0.9]
RRF_K = [10, 30, 60, 100]
RRF_WD = [0.5, 1.0, 1.5, 2.0]  # dense weight; BM25 weight fixed at 1
CE_DEPTHS = [20, 50, 100]
ALPHA = [0.5, 0.85]
BETA = [0.25, 0.5, 0.75]
SEED_M = [3, 5, 10]
GRAPH_W = [0.5, 1.0, 2.0]  # graph-list weight in RRF(hybrid, graph)
DEPTH_TOLERANCE = 0.005


def mean_r5(lists: list[RankedList], questions) -> float:
    return float(np.mean([recall_at_k(r.ids(), q.gold_ids, 5) for r, q in zip(lists, questions, strict=True)]))


class CECache:
    def __init__(self, reranker, passages):
        self.r = reranker
        self.p = passages
        self.scores: list[dict[str, float]] = []
        self.pairs = 0
        self.seconds = 0.0

    def rerank(self, qi: int, query: str, cands: RankedList, depth: int) -> RankedList:
        while len(self.scores) <= qi:
            self.scores.append({})
        cache = self.scores[qi]
        head = cands.ids(depth)
        missing = [pid for pid in head if pid not in cache]
        if missing:
            t0 = time.perf_counter()
            for pid, s in zip(missing, self.r.score(query, [self.p[pid].content for pid in missing]),
                              strict=True):
                cache[pid] = s
            self.seconds += time.perf_counter() - t0
            self.pairs += len(missing)
        ranked = RankedList.from_scores((pid, cache[pid]) for pid in head)
        floor = min(ranked.scores(), default=0.0)
        tail = [(pid, floor - 1.0 - i * 1e-6) for i, pid in enumerate(cands.ids()[depth:])]
        return RankedList(ranked.items + tail)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if cfg.data.split != "train":
        sys.exit("tuning must run on a train-derived set (data.split == 'train')")
    setup_logging(f"tune:{cfg.name}")
    pin_threads(cfg.eval.threads)
    out_dir = Path(cfg.paths.results_dir) / f"tuning_{cfg.name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    import spacy

    from hopgraph.config import BM25Config
    from hopgraph.graph.build import EntityGraph
    from hopgraph.graph.linking import QueryLinker
    from hopgraph.graph.retriever import GraphRetriever
    from hopgraph.retrieval.bm25 import BM25Retriever
    from hopgraph.retrieval.dense import DenseRetriever, STEncoder
    from hopgraph.retrieval.rerank import CrossEncoderReranker

    passages, questions = load_corpus(cfg.paths.processed_dir)
    by_id = {p.id: p for p in passages}
    report: dict = {"config": args.config, "n_questions": len(questions), "n_passages": len(passages),
                    "data_hash": corpus_hash(cfg.paths.processed_dir), "git": git_commit(),
                    "protocol": __doc__}
    t_start = time.perf_counter()

    with RSSMonitor(limit_gb=cfg.eval.rss_limit_gb) as mon:
        enc = STEncoder(cfg.dense)
        dense = DenseRetriever.load(dense_dir(cfg), enc, 100)
        qv = enc.encode_queries([q.text for q in questions])
        dense_lists = dense.search_vectors(qv, 100)
        report["dense_r5"] = mean_r5(dense_lists, questions)

        # ---- 1. BM25 grid
        bm25_grid, bm25_lists_by = [], {}
        for k1, b in itertools.product(K1, B):
            bc = BM25Config(**{**cfg.bm25.__dict__, "k1": k1, "b": b})
            bm = BM25Retriever(bc)
            bm.index(passages)
            bl = [bm.search(q.text, 100) for q in questions]
            hyb = [rrf([x, d], k=60, depth=100).top(100) for x, d in zip(bl, dense_lists, strict=True)]
            bm25_grid.append({"k1": k1, "b": b, "bm25_r5": mean_r5(bl, questions),
                              "hybrid_r5": mean_r5(hyb, questions)})
            bm25_lists_by[(k1, b)] = bl
        default_bm = (cfg.bm25.k1, cfg.bm25.b)
        best = max(bm25_grid, key=lambda r: (r["hybrid_r5"], (r["k1"], r["b"]) == default_bm))
        k1, b = best["k1"], best["b"]
        bl = bm25_lists_by[(k1, b)]
        report["bm25_grid"] = bm25_grid
        report["bm25_choice"] = {"k1": k1, "b": b}
        log.info("BM25 choice k1=%s b=%s hybrid R@5=%.4f", k1, b, best["hybrid_r5"])

        # ---- 2. RRF grid
        rrf_grid, hyb_by = [], {}
        for k, wd in itertools.product(RRF_K, RRF_WD):
            hyb = [rrf([x, d], k=k, weights=[1.0, wd], depth=100).top(100)
                   for x, d in zip(bl, dense_lists, strict=True)]
            rrf_grid.append({"k": k, "w_dense": wd, "hybrid_r5": mean_r5(hyb, questions)})
            hyb_by[(k, wd)] = hyb
        best = max(rrf_grid, key=lambda r: (r["hybrid_r5"], (r["k"], r["w_dense"]) == (60, 1.0)))
        rrf_k, wd = best["k"], best["w_dense"]
        hybrid = hyb_by[(rrf_k, wd)]
        report["rrf_grid"] = rrf_grid
        report["rrf_choice"] = {"k": rrf_k, "w_dense": wd}
        log.info("RRF choice k=%s w_dense=%s hybrid R@5=%.4f", rrf_k, wd, best["hybrid_r5"])

        # ---- 3. cross-encoder depth
        ce = CECache(CrossEncoderReranker(cfg.rerank), by_id)
        depth_rows = []
        for depth in CE_DEPTHS:
            lists = [ce.rerank(i, q.text, hybrid[i], depth) for i, q in enumerate(questions)]
            depth_rows.append({"depth": depth, "hybrid_ce_r5": mean_r5(lists, questions)})
            log.info("CE depth %d: hybrid_ce R@5=%.4f (%d pairs so far, %.0fs)", depth,
                     depth_rows[-1]["hybrid_ce_r5"], ce.pairs, ce.seconds)
        best_r5 = max(r["hybrid_ce_r5"] for r in depth_rows)
        ce_depth = min(r["depth"] for r in depth_rows if r["hybrid_ce_r5"] >= best_r5 - DEPTH_TOLERANCE)
        report["ce_depth_grid"] = depth_rows
        report["ce_depth_choice"] = ce_depth
        report["ce_ms_per_pair"] = 1000 * ce.seconds / max(ce.pairs, 1)

        # ---- 4. graph grid on the tuned first stage
        g = EntityGraph.load(graph_dir(cfg))
        emb = np.load(entity_emb_path(cfg), mmap_mode="r") if cfg.linking.embed_tau else None
        nlp = spacy.load(cfg.graph.spacy_model, exclude=["lemmatizer"])
        linker = QueryLinker(g, cfg.linking, nlp=nlp, encoder=enc, entity_emb=emb,
                             labels=frozenset(cfg.graph.labels))
        links = [linker.link(q.text) for q in questions]
        graph_rows = []
        for alpha, beta, m, wg in itertools.product(ALPHA, BETA, SEED_M, GRAPH_W):
            c = copy.deepcopy(cfg)
            c.ppr.alpha, c.seeding.passage_seed_beta, c.seeding.passage_seed_m = alpha, beta, m
            c.graph_fusion.weights = [1.0, wg]
            gr = GraphRetriever(g, linker, c.seeding, c.ppr, c.graph_fusion)
            fused = [gr.retrieve(q.text, hybrid[i], StageTimer(), link=links[i]).fused
                     for i, q in enumerate(questions)]
            gce = [ce.rerank(i, q.text, fused[i], ce_depth) for i, q in enumerate(questions)]
            graph_rows.append({"alpha": alpha, "beta": beta, "m": m, "graph_weight": wg,
                               "graph_r5": mean_r5(fused, questions),
                               "graph_ce_r5": mean_r5(gce, questions)})
            log.info("graph %s", graph_rows[-1])
        default_g = (cfg.ppr.alpha, cfg.seeding.passage_seed_beta, cfg.seeding.passage_seed_m, 1.0)
        bestg = max(graph_rows, key=lambda r: (r["graph_ce_r5"],
                                               (r["alpha"], r["beta"], r["m"], r["graph_weight"]) == default_g))
        report["graph_grid"] = graph_rows
        report["graph_choice"] = bestg
        report["hybrid_ce_r5_at_choice"] = next(r["hybrid_ce_r5"] for r in depth_rows if r["depth"] == ce_depth)

    report["peak_rss_mb"] = round(mon.peak_mb, 1)
    report["seconds"] = round(time.perf_counter() - t_start, 1)
    report["ce_pairs_scored"] = ce.pairs
    (out_dir / "tuning.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    frozen = {
        "bm25": {"k1": k1, "b": b},
        "fusion": {"k": rrf_k, "weights": [1.0, wd]},
        "rerank": {"depth": ce_depth},
        "ppr": {"alpha": bestg["alpha"]},
        "seeding": {"passage_seed_beta": bestg["beta"], "passage_seed_m": bestg["m"]},
        "graph_fusion": {"weights": [1.0, bestg["graph_weight"]]},
    }
    fdir = Path("configs/frozen")
    fdir.mkdir(parents=True, exist_ok=True)
    (fdir / f"{cfg.name}_frozen.yaml").write_text(
        f"# Selected on {cfg.name} (train) by scripts/tune.py; see {out_dir / 'tuning.json'}\n"
        + yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8")
    log.info("frozen overrides: %s", frozen)


if __name__ == "__main__":
    main()
