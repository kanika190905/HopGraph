"""Regenerate figures from a saved run directory (no retrieval is re-run).

    python scripts/make_plots.py results/published/musique_1000_gate1_369e1e2ead --per-query results/musique_1000_gate1_369e1e2ead

Figures: recall_at_k.png (needs per_query.jsonl for gold ranks), recall_vs_latency.png,
recall_by_hops.png. Palette: validated categorical slots 1-4 (scripts/validate_palette.js,
light mode: all checks pass; aqua/yellow are below 3:1 contrast, so every line and point is
also labelled directly).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
CURVES = ["hybrid", "hybrid_ce", "graph", "graph_ce"]  # fixed order -> fixed colors
LABEL = {"bm25": "BM25", "dense": "Dense", "hybrid": "Hybrid (RRF)", "hybrid_ce": "Hybrid + CE",
         "graph": "Graph-fused", "graph_only": "Graph only", "graph_ce": "Graph-fused + CE"}


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=10)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)


def recall_at_k_curve(per_query: Path, out: Path, n_label: str) -> None:
    recs = [json.loads(line) for line in per_query.open(encoding="utf-8")]
    ks = list(range(1, 21))
    fig, ax = plt.subplots(figsize=(6.4, 4.0), facecolor=SURFACE)
    for color, s in zip(SLOTS, CURVES, strict=True):
        ys = []
        for k in ks:
            vals = [sum(1 for r in rec["systems"][s]["gold_ranks"] if r is not None and r <= k)
                    / len(rec["gold"]) for rec in recs]
            ys.append(100 * sum(vals) / len(vals))
        ax.plot(ks, ys, color=color, linewidth=2, label=LABEL[s])
    style(ax, f"Recall@k on MuSiQue dev ({n_label})", "k (passages retrieved)", "Recall@k (%)")
    ax.text(20, ax.get_ylim()[0] + 1, "CE reranks the top 20, so each CE curve\nmeets its "
            "first-stage curve at k = 20", ha="right", va="bottom", fontsize=7, color=MUTED)
    ax.set_xlim(1, 20.5)
    ax.set_xticks([1, 2, 5, 10, 15, 20])
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def recall_vs_latency(run: Path, out: Path, n_label: str) -> None:
    m = pd.read_csv(run / "metrics.csv")
    m = m[m.subset == "all"].set_index("system")
    lat = pd.read_csv(run / "latency.csv").set_index("system")
    fig, ax = plt.subplots(figsize=(6.4, 4.0), facecolor=SURFACE)
    for s in m.index:
        if s == "graph_only":
            continue  # identical latency to "graph"; shown in the tables
        fam = 1 if s.startswith("graph") else 0
        x, y = lat.loc[s, "mean_ms"], 100 * m.loc[s, "recall@5"]
        ax.scatter([x], [y], s=64, color=SLOTS[fam], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(LABEL[s], (x, y), xytext=(7, -3), textcoords="offset points", fontsize=8, color=INK2)
    ax.scatter([], [], s=64, color=SLOTS[0], label="Baselines")
    ax.scatter([], [], s=64, color=SLOTS[1], label="Graph-enhanced")
    ax.set_xscale("log")
    style(ax, f"Recall@5 vs mean query latency ({n_label}, CPU, 8 threads)",
          "mean latency per query (ms, log scale)", "Recall@5 (%)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def recall_by_hops(run: Path, out: Path, n_label: str) -> None:
    m = pd.read_csv(run / "metrics.csv")
    m = m[m.subset != "all"]
    hops = sorted(m.subset.unique())
    pair = ["hybrid_ce", "graph_ce"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), facecolor=SURFACE)
    width = 0.36
    for j, s in enumerate(pair):
        ys = [100 * m[(m.subset == h) & (m.system == s)]["recall@5"].item() for h in hops]
        xs = [i + (j - 0.5) * (width + 0.02) for i in range(len(hops))]
        ax.bar(xs, ys, width=width, color=SLOTS[j], label=LABEL[s], edgecolor=SURFACE, linewidth=2)
        for x, y in zip(xs, ys, strict=True):
            ax.annotate(f"{y:.1f}", (x, y), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8, color=INK2)
    ns = [int(m[(m.subset == h)]["n"].iloc[0]) for h in hops]
    ax.set_xticks(range(len(hops)), [f"{h} (n={n})" for h, n in zip(hops, ns, strict=True)])
    style(ax, f"Recall@5 by hop count ({n_label})", "", "Recall@5 (%)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run directory with metrics.csv / latency.csv")
    ap.add_argument("--per-query", default=None, help="directory holding per_query.jsonl")
    ap.add_argument("--label", default=None)
    a = ap.parse_args()
    run = Path(a.run)
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    label = a.label or f"{meta['n_questions']:,} questions"
    recall_vs_latency(run, run / "recall_vs_latency.png", label)
    recall_by_hops(run, run / "recall_by_hops.png", label)
    pq = Path(a.per_query or run) / "per_query.jsonl"
    if pq.exists():
        recall_at_k_curve(pq, run / "recall_at_k.png", label)
    print("figures written to", run)


if __name__ == "__main__":
    main()
