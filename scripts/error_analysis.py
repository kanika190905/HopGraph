"""Why does the graph miss gold passages? Graph-distance analysis of misses (no re-retrieval).

For every (question, gold passage) that a system misses in its top-k, measure how far the
gold passage is from the query's linked entities in the entity–passage graph:

    hop 1  gold passage mentions a linked entity directly
    hop 3  gold passage shares an entity with a passage that mentions a linked entity
    >3     not reachable within 3 graph steps (or no entity linked)

It also counts gold passages whose entities are all singletons (no possible bridge).

    python scripts/error_analysis.py --config configs/musique_1000.yaml \
        --run results/musique_1000_gate1_369e1e2ead --system graph_ce --k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hopgraph.artifacts import graph_dir  # noqa: E402
from hopgraph.config import load_config  # noqa: E402
from hopgraph.graph.build import EntityGraph  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--system", default="graph_ce")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    g = EntityGraph.load(graph_dir(cfg))
    pidx = {pid: i for i, pid in enumerate(g.passage_ids)}
    kidx = {k: i for i, k in enumerate(g.entity_keys)}
    PE = (g.PE > 0).astype(np.float64).tocsr()
    EP = PE.T.tocsr()
    counts = g.entity_passage_counts

    recs = [json.loads(line) for line in (Path(a.run) / "per_query.jsonl").open(encoding="utf-8")]
    dist = Counter()
    by_hops = {}
    singleton_only = 0
    no_link = 0
    n_miss = 0
    reach1: list[float] = []
    reach3: list[float] = []
    for r in recs:
        ranks = r["systems"][a.system]["gold_ranks"]
        linked = [kidx[e[0]] for e in r.get("graph_info", {}).get("linked", []) if e[0] in kidx]
        seeds = np.zeros(g.n_entities)
        seeds[linked] = 1.0
        hop1 = PE @ seeds > 0  # passages mentioning a linked entity
        hop3 = PE @ (EP @ hop1.astype(np.float64)) > 0  # passages sharing an entity with those
        reach1.append(hop1.mean())
        reach3.append(hop3.mean())
        for gid, rank in zip(r["gold"], ranks, strict=True):
            if rank is not None and rank <= a.k:
                continue
            n_miss += 1
            pi = pidx[gid]
            ents = PE[pi].indices
            if ents.size and (counts[ents] == 1).all():
                singleton_only += 1
            if not linked:
                no_link += 1
                d = ">3"
            elif hop1[pi]:
                d = "1"
            elif hop3[pi]:
                d = "3"
            else:
                d = ">3"
            dist[d] += 1
            by_hops.setdefault(r["hops"], Counter())[d] += 1

    out = {
        "system": a.system, "k": a.k, "questions": len(recs), "missed_gold": n_miss,
        "graph_distance_from_linked_entities": {k: dist[k] for k in ("1", "3", ">3")},
        # Context: a "3" is only informative if few passages are 3 steps away.
        "mean_corpus_frac_within_1": float(np.mean(reach1)),
        "mean_corpus_frac_within_3": float(np.mean(reach3)),
        "missed_gold_only_singleton_entities": singleton_only,
        "missed_gold_no_entity_linked": no_link,
        "by_question_hops": {str(h): dict(c) for h, c in sorted(by_hops.items())},
    }
    print(json.dumps(out, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
