"""Aggregate saved per-query results into tables: metrics, latency, bootstrap, oracle.

Everything here reads only the files written by ``scripts/run_eval.py`` so reports
can be regenerated without re-running retrieval.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hopgraph.evaluation.bootstrap import holm, paired_bootstrap
from hopgraph.evaluation.timing import latency_summary
from hopgraph.pipeline import GRAPH_STAGES, SYSTEM_STAGES

BASELINES = ("bm25", "dense", "hybrid", "hybrid_ce")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def metric_names(ks: Sequence[int]) -> list[str]:
    names = []
    for k in ks:
        names += [f"recall@{k}", f"allgold@{k}", f"ndcg@{k}"]
    return names + ["rr"]


def metrics_table(records: list[dict], systems: Sequence[str], ks: Sequence[int]) -> pd.DataFrame:
    rows = []
    groups = [("all", records)] + [
        (f"{h}hop", [r for r in records if r["hops"] == h])
        for h in sorted({r["hops"] for r in records})
    ]
    for subset, recs in groups:
        for s in systems:
            row = {"system": s, "subset": subset, "n": len(recs)}
            for m in metric_names(ks):
                row[m] = float(np.mean([r["systems"][s]["metrics"][m] for r in recs]))
            row["mrr"] = row.pop("rr")
            rows.append(row)
    return pd.DataFrame(rows)


def latency_table(lat: list[dict], systems: Sequence[str]) -> pd.DataFrame:
    rows = []
    for s in systems:
        row = {"system": s, "n": len(lat)}
        row.update(latency_summary([r["system_ms"][s] for r in lat]))
        rows.append(row)
    stages = sorted({st for r in lat for st in r["stage_ms"]})
    for st in stages:
        row = {"system": f"stage:{st}", "n": len(lat)}
        row.update(latency_summary([r["stage_ms"].get(st, 0.0) for r in lat]))
        rows.append(row)
    return pd.DataFrame(rows)


def per_query(records: list[dict], system: str, metric: str) -> np.ndarray:
    return np.array([r["systems"][system]["metrics"][metric] for r in records])


def strongest_baseline(records: list[dict], systems: Sequence[str], metric: str = "recall@5") -> str:
    cands = [s for s in BASELINES if s in systems]
    return max(cands, key=lambda s: per_query(records, s, metric).mean())


def primary_comparisons(records: list[dict], systems: Sequence[str]) -> list[tuple[str, str]]:
    """Pre-registered Gate-1 family (Holm-adjusted), all on Recall@5.

    The headline test pairs the graph system with the *same* reranking as the
    strongest baseline: graph_ce vs hybrid_ce if hybrid_ce is strongest, else graph vs it.
    """
    best = strongest_baseline(records, systems)
    head = ("graph_ce", best) if best == "hybrid_ce" and "graph_ce" in systems else ("graph", best)
    pairs = [head, ("graph", "hybrid"), ("graph_ce", "hybrid_ce"), ("graph_only", "hybrid")]
    out: list[tuple[str, str]] = []
    for a, b in pairs:
        if a in systems and b in systems and (a, b) not in out:
            out.append((a, b))
    return out


SECONDARY = [("hybrid", "bm25"), ("hybrid", "dense"), ("hybrid_ce", "hybrid"), ("dense", "bm25")]


def bootstrap_table(
    records: list[dict], systems: Sequence[str], ks: Sequence[int], n: int, seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    primary = primary_comparisons(records, systems)
    for a, b in primary:
        r = paired_bootstrap(per_query(records, a, "recall@5"), per_query(records, b, "recall@5"),
                             n=n, seed=seed)
        rows.append({"family": "primary", "a": a, "b": b, "metric": "recall@5", **r.as_dict()})
    adj = holm([r["p_value"] for r in rows])
    for row, p in zip(rows, adj, strict=True):
        row["p_holm"] = p
    extra_metrics = [f"recall@{k}" for k in ks if k != 5] + ["allgold@5", "rr"]
    for a, b in primary:
        for m in extra_metrics:
            res = paired_bootstrap(per_query(records, a, m), per_query(records, b, m), n=n, seed=seed)
            rows.append({"family": "secondary_unadjusted", "a": a, "b": b, "metric": m,
                         **res.as_dict(), "p_holm": None})
    for a, b in SECONDARY:
        if a in systems and b in systems:
            res = paired_bootstrap(per_query(records, a, "recall@5"),
                                   per_query(records, b, "recall@5"), n=n, seed=seed)
            rows.append({"family": "secondary_unadjusted", "a": a, "b": b, "metric": "recall@5",
                         **res.as_dict(), "p_holm": None})
    return pd.DataFrame(rows)


def oracle_router(
    records: list[dict], lat: list[dict], base: str, graph: str, metric: str = "recall@5",
) -> dict:
    """Perfect per-query routing: run the graph stage only where it raises ``metric``.

    Ties go to ``base`` (graph did not help). Quality numbers use every question; latency
    numbers use the per-query medians of the latency pass, so they cover only the timed
    subset (``n_latency``). The routing decision for each timed question is the same one
    used in the quality numbers.
    """
    mb_all = per_query(records, base, metric)
    mg_all = per_query(records, graph, metric)
    use_all = mg_all > mb_all
    out: dict = {
        "base": base, "graph": graph, "metric": metric, "n_quality": len(records),
        "graph_helps_frac": float(use_all.mean()),
        "graph_hurts_frac": float((mg_all < mb_all).mean()),
        "tie_frac": float((mg_all == mb_all).mean()),
        f"{metric}_base": float(mb_all.mean()),
        f"{metric}_always_graph": float(mg_all.mean()),
        f"{metric}_oracle": float(np.where(use_all, mg_all, mb_all).mean()),
    }
    lat_by_q = {r["qid"]: r for r in lat}
    timed = [i for i, r in enumerate(records) if r["qid"] in lat_by_q]
    if not timed:
        return out
    use = use_all[timed]
    lb = np.array([lat_by_q[records[i]["qid"]]["system_ms"][base] for i in timed])
    lg = np.array([lat_by_q[records[i]["qid"]]["system_ms"][graph] for i in timed])
    graph_stage = np.array([
        sum(lat_by_q[records[i]["qid"]]["stage_ms"].get(s, 0.0) for s in GRAPH_STAGES)
        for i in timed
    ])
    oracle_lat = np.where(use, lg, lb)
    out.update({
        "n_latency": len(timed),
        "graph_helps_frac_latency_subset": float(use.mean()),
        "mean_ms_base": float(lb.mean()),
        "mean_ms_always_graph": float(lg.mean()),
        "mean_ms_oracle": float(oracle_lat.mean()),
        "graph_stage_mean_ms": float(graph_stage.mean()),
        "graph_stage_share_of_always_graph": float(graph_stage.mean() / lg.mean()),
        "oracle_latency_saving_vs_always_graph": float(1 - oracle_lat.mean() / lg.mean()),
    })
    return out


def random_router(
    records: list[dict], base: str, graph: str, rate: float, n: int = 1000, seed: int = 0,
    metric: str = "recall@5",
) -> dict:
    """Expected metric when the graph is used on a random ``rate`` fraction of queries."""
    mb = per_query(records, base, metric)
    mg = per_query(records, graph, metric)
    rng = np.random.default_rng(seed)
    vals = [np.where(rng.random(mb.size) < rate, mg, mb).mean() for _ in range(n)]
    return {"rate": rate, f"{metric}_mean": float(np.mean(vals)),
            f"{metric}_p2.5": float(np.quantile(vals, 0.025)),
            f"{metric}_p97.5": float(np.quantile(vals, 0.975))}


def link_stats(records: list[dict]) -> dict:
    paths = [r.get("graph_info", {}).get("link_path", "n/a") for r in records]
    seeds = [r.get("graph_info", {}).get("seed_kind", "n/a") for r in records]
    n_linked = [len(r.get("graph_info", {}).get("linked", [])) for r in records]
    conv = [r.get("graph_info", {}).get("ppr_converged") for r in records]
    iters = [r.get("graph_info", {}).get("ppr_iter", 0) for r in records]
    return {
        "link_paths": pd.Series(paths).value_counts().to_dict(),
        "seed_kinds": pd.Series(seeds).value_counts().to_dict(),
        "mean_linked_entities": float(np.mean(n_linked)),
        "frac_no_entity": float(np.mean([x == 0 for x in n_linked])),
        "ppr_converged_frac": float(np.mean([c is True for c in conv])),
        "ppr_mean_iter": float(np.mean(iters)),
    }


def analyze(run_dir: str | Path) -> dict:
    d = Path(run_dir)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    systems, ks = meta["systems"], meta["ks"]
    records = read_jsonl(d / "per_query.jsonl")
    lat = read_jsonl(d / "latency.jsonl") if (d / "latency.jsonl").exists() else []

    mt = metrics_table(records, systems, ks)
    mt.to_csv(d / "metrics.csv", index=False)
    bt = bootstrap_table(records, systems, ks, meta["bootstrap_n"], meta["bootstrap_seed"])
    bt.to_csv(d / "bootstrap.csv", index=False)
    summary: dict = {"strongest_baseline": strongest_baseline(records, systems),
                     "primary_comparisons": primary_comparisons(records, systems)}
    if lat:
        lt = latency_table(lat, systems)
        lt.to_csv(d / "latency.csv", index=False)
        oracles = []
        for base, graph in (("hybrid", "graph"), ("hybrid_ce", "graph_ce")):
            if base in systems and graph in systems:
                o = oracle_router(records, lat, base, graph)
                o["random_at_oracle_rate"] = random_router(records, base, graph,
                                                           o["graph_helps_frac"])
                oracles.append(o)
        summary["oracle"] = oracles
    if any("graph_info" in r for r in records):
        summary["linking"] = link_stats(records)
    (d / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def stages_for(system: str) -> list[str]:
    return SYSTEM_STAGES[system]
