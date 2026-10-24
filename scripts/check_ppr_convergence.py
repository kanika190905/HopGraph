"""Numerical check (uses no gold labels): how much does the PPR iteration cap change rankings?

For every question, the same seed vector is propagated with the configured cap and to full
convergence. The script reports how far the scores are from the fixed point and how
similar the top-k passage rankings are.

    python scripts/check_ppr_convergence.py --config configs/musique_200.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hopgraph.artifacts import bm25_dir, dense_dir, entity_emb_path, graph_dir  # noqa: E402
from hopgraph.config import load_config  # noqa: E402
from hopgraph.evaluation.timing import pin_threads  # noqa: E402
from hopgraph.graph.ppr import ppr  # noqa: E402
from hopgraph.ingestion.corpus import load_corpus  # noqa: E402
from hopgraph.retrieval.fusion import rrf  # noqa: E402
from hopgraph.runinfo import setup_logging  # noqa: E402
from hopgraph.types import RankedList  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ref-iter", type=int, default=2000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.name)
    pin_threads(cfg.eval.threads)

    import spacy

    from hopgraph.graph.build import EntityGraph
    from hopgraph.graph.linking import QueryLinker
    from hopgraph.graph.retriever import GraphRetriever
    from hopgraph.retrieval.bm25 import BM25Retriever
    from hopgraph.retrieval.dense import DenseRetriever, STEncoder

    passages, questions = load_corpus(cfg.paths.processed_dir)
    enc = STEncoder(cfg.dense)
    bm25 = BM25Retriever.load(cfg.bm25, bm25_dir(cfg))
    dense = DenseRetriever.load(dense_dir(cfg), enc, cfg.dense.top_k)
    g = EntityGraph.load(graph_dir(cfg))
    emb = np.load(entity_emb_path(cfg), mmap_mode="r") if cfg.linking.embed_tau else None
    nlp = spacy.load(cfg.graph.spacy_model, exclude=["lemmatizer"])
    linker = QueryLinker(g, cfg.linking, nlp=nlp, encoder=enc, entity_emb=emb,
                         labels=frozenset(cfg.graph.labels))
    gr = GraphRetriever(g, linker, cfg.seeding, cfg.ppr, cfg.graph_fusion)
    T = g.transition
    k = args.k

    rows = []
    for q in questions:
        h = rrf([bm25.search(q.text, cfg.bm25.top_k), dense.search(q.text, cfg.dense.top_k)],
                k=cfg.fusion.k, weights=cfg.fusion.weights, depth=cfg.fusion.depth)
        h = h.top(cfg.fusion.depth)
        s, kind = gr.seed_vector(linker.link(q.text), h)
        if kind == "empty":
            continue
        capped = ppr(T, s, cfg.ppr.alpha, cfg.ppr.tol, cfg.ppr.max_iter)
        ref = ppr(T, s, cfg.ppr.alpha, 1e-13, args.ref_iter)
        a = RankedList.from_array(gr.passage_scores(capped.scores), g.passage_ids, k).ids()
        b = RankedList.from_array(gr.passage_scores(ref.scores), g.passage_ids, k).ids()
        fa = rrf([h, RankedList.from_array(gr.passage_scores(capped.scores), g.passage_ids, 100)],
                 k=cfg.graph_fusion.k, weights=cfg.graph_fusion.weights).ids(k)
        fb = rrf([h, RankedList.from_array(gr.passage_scores(ref.scores), g.passage_ids, 100)],
                 k=cfg.graph_fusion.k, weights=cfg.graph_fusion.weights).ids(k)
        rows.append({
            "l1_to_fixed_point": float(np.abs(capped.scores - ref.scores).sum()),
            "last_step_delta": capped.l1_delta,
            "ref_converged": ref.converged, "ref_iter": ref.n_iter,
            "topk_overlap": len(set(a) & set(b)) / k,
            "topk_identical_order": a == b,
            "fused_topk_overlap": len(set(fa) & set(fb)) / k,
            "fused_topk_identical_order": fa == fb,
        })

    summary = {
        "alpha": cfg.ppr.alpha, "cap": cfg.ppr.max_iter, "tol": cfg.ppr.tol, "k": k,
        "n": len(rows),
        "mean_l1_to_fixed_point": float(np.mean([r["l1_to_fixed_point"] for r in rows])),
        "max_l1_to_fixed_point": float(np.max([r["l1_to_fixed_point"] for r in rows])),
        "mean_last_step_delta": float(np.mean([r["last_step_delta"] for r in rows])),
        "ref_all_converged": all(r["ref_converged"] for r in rows),
        "ref_mean_iter": float(np.mean([r["ref_iter"] for r in rows])),
        "graph_topk_mean_overlap": float(np.mean([r["topk_overlap"] for r in rows])),
        "graph_topk_identical_order_frac": float(np.mean([r["topk_identical_order"] for r in rows])),
        "fused_topk_mean_overlap": float(np.mean([r["fused_topk_overlap"] for r in rows])),
        "fused_topk_identical_order_frac": float(
            np.mean([r["fused_topk_identical_order"] for r in rows])),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
