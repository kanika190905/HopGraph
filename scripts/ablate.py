"""Scripted graph ablations: one change at a time against the tuned reference.

    python scripts/ablate.py --config configs/musique_tune500_tuned.yaml

Each variant is a list of dotted overrides. Variants that change graph construction rebuild
the graph first (artefact directories are hashed by setting, so nothing stale is reused).
Systems are hybrid / graph / graph_only, without the cross-encoder, so each variant takes about
a minute. Output: results/ablations_<name>/ablations.csv (+ .json) with Δ vs the reference and
a paired-bootstrap CI on graph Recall@5.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hopgraph.config import load_config  # noqa: E402
from hopgraph.evaluation.bootstrap import paired_bootstrap  # noqa: E402
from hopgraph.evaluation.report import per_query, read_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NO_DATE = "graph.labels=[PERSON,NORP,FAC,ORG,GPE,LOC,PRODUCT,EVENT,WORK_OF_ART,LAW,LANGUAGE]"
VARIANTS: dict[str, list[str]] = {
    "reference": [],
    "no_specificity": ["seeding.use_specificity=false"],
    "no_passage_seeds": ["seeding.passage_seed_m=0"],
    "no_entity_seeds": ["linking.use_ner=false", "linking.use_ngram=false",
                        "linking.embed_tau=null", "linking.use_noun_chunks=false"],
    "no_ner_linking": ["linking.use_ner=false"],
    "no_ngram_linking": ["linking.use_ngram=false"],
    "no_embedding_linking": ["linking.embed_tau=null"],
    "entity_sum_scoring": ["graph_fusion.passage_score=entity_sum"],
    "equal_rrf_weights": ["graph_fusion.weights=[1.0,1.0]"],
    "no_title_entities": ["graph.include_titles=false"],
    "no_date_entities": [NO_DATE],
    "synonym_edges_0.85": ["graph.synonym_tau=0.85"],
    "synonym_edges_0.90": ["graph.synonym_tau=0.9"],
    "name_label_keys": ["graph.entity_key=name_label"],
    "count_mention_weights": ["graph.mention_weight=count"],
}
REBUILD_PREFIXES = ("graph.",)
SYSTEMS = "eval.systems=[hybrid,graph,graph_only]"


def run(args: list[str]) -> str:
    print("+", " ".join(args), flush=True)
    out = subprocess.run([sys.executable, *args], cwd=ROOT, check=True, capture_output=True, text=True)
    return out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    if cfg.data.split != "train":
        sys.exit("ablations that inform design decisions must run on train-derived data")
    out_dir = ROOT / cfg.paths.results_dir / f"ablations_{cfg.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    names = a.only or list(VARIANTS)
    run_dirs: dict[str, str] = {}
    for name in names:
        sets = VARIANTS[name]
        if any(s.startswith(REBUILD_PREFIXES) for s in sets):
            run(["scripts/build_graph.py", "--config", a.config, "--set", *sets])
        run_dirs[name] = run(["scripts/run_eval.py", "--config", a.config, "--tag", f"abl_{name}",
                              "--no-latency", "--set", SYSTEMS, *sets])

    ref = read_jsonl(ROOT / run_dirs["reference"] / "per_query.jsonl") if "reference" in run_dirs else None
    rows = []
    for name in names:
        recs = read_jsonl(ROOT / run_dirs[name] / "per_query.jsonl")
        summ = json.loads((ROOT / run_dirs[name] / "summary.json").read_text(encoding="utf-8"))
        row = {"variant": name, "overrides": "; ".join(VARIANTS[name]), "run": run_dirs[name]}
        for s in ("hybrid", "graph", "graph_only"):
            for m in ("recall@5", "recall@10", "allgold@5"):
                row[f"{s}_{m}"] = round(100 * float(per_query(recs, s, m).mean()), 2)
        row["frac_no_entity"] = summ.get("linking", {}).get("frac_no_entity")
        row["mean_linked_entities"] = summ.get("linking", {}).get("mean_linked_entities")
        if ref is not None and name != "reference":
            assert [r["qid"] for r in recs] == [r["qid"] for r in ref]
            b = paired_bootstrap(per_query(recs, "graph", "recall@5"), per_query(ref, "graph", "recall@5"),
                                 n=10_000, seed=0)
            row.update({"d_graph_r5": round(100 * b.diff, 2), "ci_low": round(100 * b.ci_low, 2),
                        "ci_high": round(100 * b.ci_high, 2), "p": round(b.p_value, 4)})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ablations.csv", index=False)
    (out_dir / "ablations.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(df[["variant", "graph_recall@5", "d_graph_r5", "ci_low", "ci_high", "graph_recall@10",
              "graph_allgold@5"]].to_string(index=False))


if __name__ == "__main__":
    main()
