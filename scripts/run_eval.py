"""Evaluate every configured system: quality pass, latency pass, then report tables.

    python scripts/run_eval.py --config configs/musique_200.yaml [--tag name] [--set key=value ...]

Outputs go to results/<name>[_<tag>]_<run_id>/:
config.yaml, meta.json, env.json, per_query.jsonl, latency.jsonl, metrics.csv,
latency.csv, bootstrap.csv, summary.json, run.log
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hopgraph.artifacts import bm25_dir, dense_dir, entity_emb_path, graph_dir  # noqa: E402
from hopgraph.config import load_config, parse_overrides, save_config  # noqa: E402
from hopgraph.evaluation.memory import RSSMonitor  # noqa: E402
from hopgraph.evaluation.metrics import evaluate_ranking  # noqa: E402
from hopgraph.evaluation.report import analyze  # noqa: E402
from hopgraph.evaluation.timing import pin_threads  # noqa: E402
from hopgraph.ingestion.corpus import corpus_hash, load_corpus  # noqa: E402
from hopgraph.pipeline import Pipeline, system_latency_ms  # noqa: E402
from hopgraph.runinfo import environment, git_commit, run_id, setup_logging  # noqa: E402

log = logging.getLogger("run_eval")
SAVE_TOP = 20  # ranked IDs stored per system per query (gold ranks are stored to depth 100)


def build_pipeline(cfg, passages, systems):
    from hopgraph.retrieval.bm25 import BM25Retriever
    from hopgraph.retrieval.dense import DenseRetriever, STEncoder

    t = {}
    t0 = time.perf_counter()
    enc = STEncoder(cfg.dense)
    t["encoder_load_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    bm25 = BM25Retriever.load(cfg.bm25, bm25_dir(cfg))
    dense = DenseRetriever.load(dense_dir(cfg), enc, cfg.dense.top_k)
    t["index_load_s"] = time.perf_counter() - t0
    by_id = {p.id: p for p in passages}
    if bm25.ids != [p.id for p in passages] or dense.ids != bm25.ids:
        raise RuntimeError("index passage order does not match the corpus; rebuild indexes")

    reranker = None
    if {"hybrid_ce", "graph_ce"} & set(systems):
        from hopgraph.retrieval.rerank import CrossEncoderReranker

        t0 = time.perf_counter()
        reranker = CrossEncoderReranker(cfg.rerank)
        t["reranker_load_s"] = time.perf_counter() - t0

    graph = None
    if {"graph", "graph_only", "graph_ce"} & set(systems):
        import numpy as np
        import spacy

        from hopgraph.graph.build import EntityGraph
        from hopgraph.graph.linking import QueryLinker
        from hopgraph.graph.retriever import GraphRetriever

        t0 = time.perf_counter()
        g = EntityGraph.load(graph_dir(cfg))
        if g.passage_ids != bm25.ids:
            raise RuntimeError("graph passage order does not match the corpus; rebuild graph")
        emb = None
        if cfg.linking.embed_tau is not None:
            emb = np.load(entity_emb_path(cfg), mmap_mode="r")
        nlp = spacy.load(cfg.graph.spacy_model, exclude=["lemmatizer"])
        linker = QueryLinker(g, cfg.linking, nlp=nlp, encoder=enc, entity_emb=emb,
                             labels=frozenset(cfg.graph.labels))
        graph = GraphRetriever(g, linker, cfg.seeding, cfg.ppr, cfg.graph_fusion)
        t["graph_load_s"] = time.perf_counter() - t0
    return Pipeline(cfg, by_id, bm25, dense, reranker, graph), t


def gold_ranks(ranked_ids: list[str], gold: tuple[str, ...]) -> list[int | None]:
    pos = {pid: i + 1 for i, pid in enumerate(ranked_ids)}
    return [pos.get(g) for g in gold]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. ppr.alpha=0.5")
    ap.add_argument("--no-latency", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    systems = cfg.eval.systems
    threads = pin_threads(cfg.eval.threads)
    passages, questions = load_corpus(cfg.paths.processed_dir)
    data_hash = corpus_hash(cfg.paths.processed_dir)
    git = git_commit()
    rid = run_id(cfg.hash(), git.get("commit"), data_hash)
    name = cfg.name + (f"_{args.tag}" if args.tag else "")
    run_dir = Path(cfg.paths.results_dir) / f"{name}_{rid}"
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(rid, log_file=run_dir / "run.log")
    save_config(cfg, run_dir / "config.yaml")
    (run_dir / "env.json").write_text(json.dumps(environment(), indent=2), encoding="utf-8")
    log.info("run %s -> %s; systems %s; %d questions, %d passages", rid, run_dir, systems,
             len(questions), len(passages))

    meta = {
        "run_id": rid, "name": name, "config_hash": cfg.hash(), "data_hash": data_hash, "git": git,
        "systems": systems, "ks": cfg.eval.ks, "n_questions": len(questions),
        "n_passages": len(passages), "threads": threads, "overrides": args.set,
        "bootstrap_n": cfg.eval.bootstrap_n, "bootstrap_seed": cfg.eval.bootstrap_seed,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with RSSMonitor(limit_gb=cfg.eval.rss_limit_gb) as mon:
        pipe, load_times = build_pipeline(cfg, passages, systems)
        meta["load_seconds"] = {k: round(v, 3) for k, v in load_times.items()}
        meta["rss_after_load_mb"] = round(mon.peak_mb, 1)

        # ---- quality pass (single run per query)
        t0 = time.perf_counter()
        with (run_dir / "per_query.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for i, q in enumerate(questions):
                out = pipe.run(q.text, systems)
                rec = {"qid": q.id, "hops": q.hops, "hop_type": q.meta.get("hop_type"),
                       "question": q.text, "gold": list(q.gold_ids), "systems": {},
                       "stage_ms": {k: round(v, 3) for k, v in out.stage_ms.items()}}
                for s in systems:
                    ids = out.rankings[s].ids()
                    rec["systems"][s] = {
                        "ranked": ids[:SAVE_TOP],
                        "gold_ranks": gold_ranks(ids, q.gold_ids),
                        "metrics": evaluate_ranking(ids, q.gold_ids, cfg.eval.ks),
                    }
                if out.info:
                    rec["graph_info"] = out.info
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if (i + 1) % 50 == 0:
                    log.info("quality pass %d/%d", i + 1, len(questions))
        meta["quality_pass_seconds"] = round(time.perf_counter() - t0, 2)

        # ---- latency pass: warm-up, then median of repeats per query
        if not args.no_latency:
            lat_q = questions[: cfg.eval.latency_n]
            for q in lat_q[: cfg.eval.warmup]:
                pipe.run(q.text, systems)
            t0 = time.perf_counter()
            with (run_dir / "latency.jsonl").open("w", encoding="utf-8", newline="\n") as f:
                for q in lat_q:
                    reps = [pipe.run(q.text, systems).stage_ms for _ in range(cfg.eval.latency_repeats)]
                    stages = sorted({s for r in reps for s in r})
                    rec = {
                        "qid": q.id,
                        "stage_ms": {s: statistics.median(r.get(s, 0.0) for r in reps) for s in stages},
                        "system_ms": {sys_: statistics.median(system_latency_ms(r, sys_) for r in reps)
                                      for sys_ in systems},
                    }
                    f.write(json.dumps(rec) + "\n")
            meta["latency_pass_seconds"] = round(time.perf_counter() - t0, 2)
            meta["latency_n"] = len(lat_q)
            meta["latency_repeats"] = cfg.eval.latency_repeats
            meta["warmup"] = cfg.eval.warmup

    meta["peak_rss_mb"] = round(mon.peak_mb, 1)
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    summary = analyze(run_dir)
    log.info("summary: %s", json.dumps(summary, indent=2))
    print(run_dir)


if __name__ == "__main__":
    main()
