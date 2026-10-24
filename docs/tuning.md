# Week-2 tuning record (train-derived set)

*All numbers come from `results/published/tuning_musique_tune500/tuning.json` (commit `c2f37d7`
code; run 2026-10-24 23:33 → 2026-10-25 00:22, 49 min, peak RSS 1.5 GB).*

## Tuning set

`configs/musique_tune500.yaml`: 500 questions sampled (seed 0) from **MuSiQue-Ans train**
(19,938 records, 0 skipped). Their merged candidates form a separate corpus: 10,000
paragraphs → 7,998 passages. No dev question or dev-only passage is involved, and
`scripts/tune.py` refuses to run unless `data.split == "train"`.

## Protocol (written into `scripts/tune.py` before it ran)

1. BM25 (k1, b) ∈ {0.9, 1.2, 1.5, 2.0} × {0.4, 0.6, 0.75, 0.9}. Objective: hybrid R@5 with default RRF.
2. RRF k ∈ {10, 30, 60, 100} × dense weight ∈ {0.5, 1, 1.5, 2} (BM25 weight 1). Objective: hybrid R@5.
3. Cross-encoder depth ∈ {20, 50, 100}. Choose the **smallest** depth within 0.5 points of the best hybrid_ce R@5.
4. Graph: α ∈ {0.5, 0.85} × passage-seed share β ∈ {0.25, 0.5, 0.75} × passage seeds m ∈ {3, 5, 10} × graph RRF weight ∈ {0.5, 1, 2}. Objective: graph_ce R@5 at the chosen CE depth, on top of the tuned first stage.

Ties keep the default. CE scores were cached per (question, passage), so every depth and
every graph configuration was scored with identical model outputs (83,866 pairs, 30.3 ms per pair).

## Results (R@5, %, on the 500 train questions)

| Stage | Default | Selected | Δ |
|---|---|---|---|
| BM25 | k1 1.5, b 0.75 → hybrid 59.2 | **k1 0.9, b 0.75** → hybrid 59.2 | +0.0 (k1 1.5 scored 59.17; 0.9 scored 59.18) |
| RRF | k 60, w_dense 1 → hybrid 59.2 | **k 10, w_dense 1.5** → hybrid **61.1** | +1.9 |
| CE depth | 20 → hybrid_ce 63.9 | **50** → hybrid_ce **64.6** (100 → 64.4) | +0.7 |
| Graph | α .85, β .5, m 5, w 1 → graph_ce 64.9 | **α 0.85, β 0.25, m 3, w 2.0** → graph_ce **69.8** | +4.9 |

On the tuning set the tuned graph_ce beats the tuned hybrid_ce by 5.2 points.
**This is not a result.** It is the selection objective, evaluated on the same questions used to
select it, so it is optimistically biased. The graph had the larger grid (54 configurations
against the baseline's 35), so its winner's curse is larger too.

## Things to note

- **The BM25 change is noise-level.** k1 0.9 beat 1.5 by 0.02 points. The rule takes the maximum, but no real improvement was measured.
- **Three graph settings sit on a grid edge** (β = 0.25, m = 3, graph weight = 2.0 are the extremes tried). The true optimum may lie beyond them. I did not extend the grid after seeing this, to avoid an open-ended search. If extended later, it must be on train data and reported as a second tuning round.
- The selected settings move the graph toward a *more entity-driven* walk: less seed mass on hybrid passages, fewer passage seeds, and more weight on the graph ranking in fusion.
- Deeper reranking gives diminishing returns: depth 100 scored *below* depth 50. So the baseline is not under-reranked at 50.
- Tuning-set absolute scores (hybrid 61.1) are higher than dev scores (hybrid 49.6 on dev-1,000). The train sample's corpus has fewer near-duplicate distractors (20% duplicates vs 42% on dev-1,000), so the two sets are not interchangeable in difficulty.

## Frozen for evaluation

`configs/frozen/musique_tune500_frozen.yaml` → merged into `configs/musique_1000_tuned.yaml`,
which is evaluated **once** on the same 1,000 dev questions as the Gate-1 retest.
