# Error analysis

*Source: the tuned dev-1,000 run (`results/published/musique_1000_tuned_final_45b38b0ff5`) and the
train-set ablations (`results/published/ablations_musique_tune500_tuned`). Regenerate with
`scripts/error_analysis.py` and `scripts/ablate.py`. Some counts below come from ad-hoc
analysis of the saved `per_query.jsonl` and are labelled as such.*

## 1. Where the graph wins and loses (graph_ce vs hybrid_ce, R@5)

| | Questions | Mean linked entities |
|---|---|---|
| Graph helps | 142 (14.2%) | 3.51 |
| Graph hurts | 52 (5.2%) | **4.25** |
| Tie | 806 | — |

By MuSiQue composition type (R@5, hybrid_ce → graph_ce; ad-hoc):

| Type | n | hybrid_ce | graph_ce | Δ | graph (no CE) |
|---|---|---|---|---|---|
| 2hop | 518 | 64.0 | 68.6 | +4.6 | 66.6 |
| 3hop1 (chain) | 230 | 53.0 | 56.2 | +3.2 | 54.6 |
| 3hop2 | 74 | 62.2 | 62.2 | +0.0 | 59.5 |
| 4hop1 (chain) | 106 | 33.0 | 36.1 | +3.1 | 34.4 |
| 4hop2 | 36 | 29.2 | 38.2 | +9.0 | 41.7 |
| 4hop3 | 36 | 39.6 | 43.1 | +3.5 | 50.7 |

The 36-question subsets are far too small for these per-type differences to be reliable.

## 2. The dominant failure: noisy query links

In queries where the graph hurts, the linked set usually contains **generic entities** next to
the real one, and those entities pull in hub passages:

| Question (abridged) | Linked | Graph-promoted distractors |
|---|---|---|
| 2017 population of the country where *Bedari* was made | `2017`/DATE, `bedari` | "World population", "Togo", "Houston" |
| Where do greyhound buses depart … *Sleep Is the Enemy* | `sleep is the enemy`, `greyhound`/n-gram, `buses`/n-gram | "Out Where the Buses Don't Run", "Philadelphia" |
| When does real time with the author of *The New New Rules* start again in 2018 | `2018`/DATE, `time`/n-gram, `start`/n-gram, `new new rules` | "Daylight saving time", "2018 Major League Baseball season" |

In these cases the gold bridge passage drops out of the top 5, for example from rank 2 to rank 71.

**The ablations confirm it (train tuning set, graph R@5).** Removing the dictionary n-gram linker
*raises* R@5 by **+1.5 [+0.2, +2.9]**, although it is the most prolific linker: it produced 2,094
links on dev against 1,360 from NER. Dropping DATE entities is neutral (+0.4 [−0.3, +1.1]).
The n-gram linker was added on Day 2 (ADR 0001 §4) to rescue lower-case mentions. On balance
it adds more noise than recall.

## 3. Graph distance of missed gold passages

For each gold passage missed at k = 5 (`scripts/error_analysis.py`), the table gives its graph
distance from the query's linked entities:

| System | Missed gold | Mentions a linked entity (1 step) | Shares an entity with such a passage (3 steps) | Farther / unreachable |
|---|---|---|---|---|
| hybrid_ce | 1,254 | 269 | 841 | 144 |
| graph | 1,182 | 222 | 808 | 152 |
| graph_ce | 1,153 | 225 | 783 | 145 |

Context matters. On average, only **0.63%** of the corpus (about 73 passages) is one step from
the linked entities, but **42%** is within three steps, because hub entities connect almost
everything. So:

- About 225 missed gold passages sit in the small one-step neighbourhood and still rank outside the top 5. That is a **ranking/scoring failure**, not a reachability one.
- "Three steps away" covers almost half the corpus, so it is not evidence that the graph "had" the answer. Restricting walks to low-degree entities, or degree-normalizing scores, is the obvious next experiment.
- Only about 145 misses (12%) are unreachable, which is the true ceiling for this graph.

## 4. Degree bias

The graph's top-20 passages average **103.6 words**, against 94.3 for hybrid's (corpus mean
82.3). Longer passages mention more entities, gain more edges, and collect more PPR mass. The
same effect makes graph_ce's rerank about 20% slower at equal depth. Switching mention
weights from binary to counts (which weights a passage's edges by how often it mentions each
entity) *raised* graph R@5 by **+1.6 [+0.5, +2.7]** on the tuning set. So the degree effect is
not simply harmful, and its net effect needs a direct degree-normalization ablation.

## 5. What each component contributes (train ablations, graph R@5, reference 67.3)

| Removed / changed | Δ R@5 | 95% CI | Reading |
|---|---|---|---|
| entity seeds (passage-only PPR) | −4.9 | [−6.7, −3.2] | the entity graph, not mere passage diffusion, carries the effect |
| passage seeds | −2.6 | [−4.8, −0.4] | hybrid passages anchor the walk |
| entity-sum scoring instead of passage nodes | −1.9 | [−3.5, −0.3] | HippoRAG-2-style passage nodes beat HippoRAG-1 aggregation here |
| title entities | −1.3 | [−2.4, −0.1] | titles compensate for spaCy-sm misses |
| NER linking | −1.0 | [−1.6, −0.5] | |
| specificity seed weights | −0.7 | [−1.5, −0.0] | small |
| embedding-NN linking | −0.3 | [−0.6, 0.0] | rarely fires (92 links on dev) |
| name+label entity keys | −0.5 | [−1.5, +0.5] | n.s.; supports name-only keys (ADR 0001 §5) |
| synonym edges τ = 0.85 / 0.90 | 0.0 / +0.6 | [−1.1, +1.0] / [−0.2, +1.5] | n.s. |
| DATE entities removed | +0.4 | [−0.3, +1.1] | n.s. |
| equal RRF weights (1:1 instead of 1:2) | +0.4 | [−0.6, +1.3] | n.s. |
| **n-gram linker removed** | **+1.5** | **[+0.2, +2.9]** | helps to remove |
| **count mention weights** | **+1.6** | **[+0.5, +2.7]** | helps |

These are single-change ablations with unadjusted CIs, run on 500 train questions without the
cross-encoder. Interactions were not tested, for example whether the two improvements add up.

## 6. After the v2 change (held-out run)

The locked v2 configuration (no n-gram linker, count weights) links far fewer entities: 1.83 per
query on the held-out set, against 3.55 for v1 on dev-1,000. Its seed neighbourhoods are
correspondingly narrower: within three graph steps lies 31% of the corpus, against 42% for v1.
Of the 1,621 gold passages graph_ce misses at k = 5 on held-out, 293 directly mention a linked
entity, 1,010 are three steps away, and 318 are farther or unreachable
(`results/published/musique_heldout1417_heldout_e21b488fc6/error_distance_graph_ce.json`).
The held-out set and corpus differ from dev-1,000, so these are descriptive numbers, not a
controlled comparison.

## 7. Implications

1. Done: v2 (no n-gram linker + count weights) was confirmed on train and then on the 1,417 held-out questions (`docs/heldout_protocol.md`).
2. Degree normalization and low-degree-only expansion target the ranking failures in §3. Any such change now needs new evaluation data, because all MuSiQue dev questions have been seen.
