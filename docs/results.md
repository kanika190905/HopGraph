# Results

*Every number comes from a saved run in `results/published/`. R@k, AllGold and MRR are
percentages. Δ is a paired difference in absolute points with a 95% paired-bootstrap CI
(10,000 resamples, seed 0). Hardware: AMD Ryzen 7 8840U, 8 threads pinned, 15.3 GB RAM,
CPU only, Windows 11.*

## Definitive result: held-out confirmation (MuSiQue-Ans dev, 1,417 never-evaluated questions)

Run `results/published/musique_heldout1417_heldout_e21b488fc6` (commit `4af7cc4`, clean tree).
These are positions 1,000–2,416 of the seed-0 permutation, with **zero question overlap** with
dev-1,000. Their 28,326 candidates merge into 14,965 passages; 5,391 of those passages also
appear in the dev-1,000 corpus because both slices draw on the same Wikipedia paragraphs, but
none of these questions' labels was ever used. The configuration was locked before the run:
train-tuned v1 plus the v2 changes (no n-gram linker, count mention weights), adopted on train
under a pre-registered rule ([`heldout_protocol.md`](heldout_protocol.md)). It was evaluated **once**.

| System | R@2 | R@5 | R@10 | AllGold@5 | MRR | mean ms | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| bm25 | 32.7 | 42.5 | 49.3 | 9.4 | 69.1 | 0.6 | 0.6 | 0.8 |
| dense | 36.7 | 48.1 | 55.7 | 15.2 | 74.2 | 19.0 | 18.9 | 21.7 |
| hybrid | 39.9 | 52.5 | 60.6 | 17.4 | 77.1 | 19.8 | 19.7 | 22.7 |
| hybrid_ce (top 50) | 43.3 | 56.7 | 64.1 | 22.1 | 82.7 | 1,391 | 1,390 | 1,787 |
| graph_only | 40.1 | 57.5 | 66.4 | 25.1 | 76.6 | 74.7 | 70.3 | 97.4 |
| **graph** | 41.5 | **59.9** | **71.1** | 26.4 | 80.7 | **74.7** | 70.3 | 97.4 |
| graph_ce (top 50) | **44.5** | 59.8 | 70.1 | **27.6** | **83.2** | 1,683 | 1,664 | 2,165 |

| Comparison (R@5) | Δ | 95% CI | p |
|---|---|---|---|
| **graph_ce − hybrid_ce** (pre-registered headline, Holm) | **+3.1** | **[+2.2, +4.1]** | 0.0006 |
| graph − hybrid (Holm) | +7.3 | [+6.2, +8.5] | 0.0006 |
| graph_only − hybrid (Holm) | +5.0 | [+3.7, +6.3] | 0.0006 |
| **graph − hybrid_ce** (pre-registered secondary, unadjusted) | **+3.2** | **[+1.8, +4.5]** | 0.0002 |
| *exploratory:* graph_ce − graph | −0.0 | [−1.2, +1.2] | 0.95 |
| *exploratory:* hybrid_ce − hybrid | +4.2 | [+3.1, +5.3] | 0.0002 |

Secondary metrics, graph_ce − hybrid_ce: R@2 +1.2 [+0.6, +1.9], R@10 +6.0 [+4.8, +7.3],
AllGold@5 +5.5 [+4.0, +7.1], MRR +0.6 [−0.2, +1.2]. graph − hybrid_ce: AllGold@5 +4.3 [+2.2, +6.4].

By hop count, R@5 for hybrid_ce → graph_ce → graph: 2-hop (n = 734) 64.9 → 68.9 → 68.1;
3-hop (n = 456) 55.3 → 56.6 → 56.2; 4-hop (n = 227) 33.4 → 37.2 → **40.5**.

**What the held-out run establishes:**

1. **The headline replicates.** The graph-enhanced reranked pipeline beats the tuned
   reranked baseline by +3.1 [+2.2, +4.1], above the pre-registered +3 bar, on questions that
   played no part in any decision. The estimate sits between the two dev-1,000 results (+2.4
   untuned, +3.9 tuned), which is the shrinkage you would expect once selection effects are removed.
2. **The cheap pipeline wins.** Graph-fused retrieval *without* any reranker beats tuned
   hybrid + cross-encoder by +3.2 [+1.8, +4.5] R@5 at **75 ms vs 1,391 ms** mean latency (about
   19× faster). This was a stated hypothesis before the run, not a post-hoc finding.
3. With v2, the cross-encoder adds nothing on top of the graph (−0.0 [−1.2, +1.2]) while
   costing about 1.6 s per query. Reranking still helps the plain hybrid list (+4.2).

Index build at 14,965 passages: BGE embedding 449 s (33.3 passages/s, 1.3 GB peak RSS),
NER 89 s, entity-name embedding (82,087 names) 129 s, CSR graph (97,052 nodes, 319,872
non-zeros) 0.5 s. PPR costs 40.5 ms per query. Evaluation peak RSS was 1.51 GB.

Oracle router (held-out): with the cross-encoder, the graph stage is 3.3% of latency and the
oracle saves 14%, so routing is not worth building there. Without it, the graph stage is 73%,
the graph helps 23.9% / hurts 5.6% of queries, and the oracle saves 53% (74.7 → 34.9 ms) while
raising R@5 from 59.9 to 62.1.

---

*The sections below are the **development** history on dev-1,000, which was evaluated twice.
They are kept for transparency; the held-out section above is the result to cite.*

## Development result (MuSiQue-Ans dev, 1,000 questions, tuned on train)

Run `results/published/musique_1000_tuned_final_45b38b0ff5` (commit `47d9ac5`, clean tree).
Hyper-parameters for **every** system were selected on 500 MuSiQue **train** questions
(`docs/tuning.md`) and frozen before this single evaluation.

| System | R@2 | R@5 | R@10 | AllGold@5 | MRR | mean ms | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| bm25 (k1 0.9, b 0.75) | 32.5 | 42.7 | 49.1 | 10.0 | 69.2 | 0.6 | 0.6 | 0.7 |
| dense (BGE-small) | 36.4 | 48.5 | 56.3 | 14.7 | 75.2 | 19.3 | 18.9 | 22.8 |
| hybrid (RRF k 10, w_dense 1.5) | 39.6 | 51.9 | 59.8 | 17.0 | 79.1 | 20.0 | 19.7 | 23.6 |
| hybrid_ce (rerank top 50) | 43.0 | 55.9 | 63.0 | 20.9 | 84.0 | 1,416 | 1,416 | 1,864 |
| graph_only | 36.5 | 53.7 | 63.1 | 18.5 | 74.4 | 65.3 | 61.4 | 87.0 |
| **graph** (RRF hybrid + PPR, graph weight 2) | 41.5 | 58.4 | 68.8 | 24.0 | 82.7 | **65.3** | 61.4 | 87.0 |
| **graph_ce** (rerank top 50) | **44.7** | **59.8** | **69.4** | **27.0** | **84.6** | 1,739 | 1,701 | 2,207 |

### Pre-registered comparisons (Recall@5, Holm-adjusted)

| Comparison | Δ R@5 | 95% CI | p (Holm) |
|---|---|---|---|
| **graph_ce − hybrid_ce** (headline) | **+3.9** | **[+2.8, +5.0]** | 0.0006 |
| graph − hybrid | +6.6 | [+5.3, +7.9] | 0.0006 |
| graph_only − hybrid | +1.8 | [+0.3, +3.3] | 0.019 |

Secondary, unadjusted (graph_ce − hybrid_ce): R@2 +1.7 [+1.0, +2.4], R@10 +6.4 [+5.0, +7.9],
**AllGold@5 +6.1 [+4.3, +7.9]**, MRR +0.6 [−0.2, +1.4]. On the 800 questions that were not
in the Day-3 spike: headline +4.0 [+2.8, +5.3].

### Exploratory, not pre-registered (unadjusted)

| Comparison | Δ R@5 | Δ AllGold@5 | Mean latency |
|---|---|---|---|
| **graph − hybrid_ce** | **+2.5 [+1.1, +3.9]** | +3.1 [+0.7, +5.4] | **65 ms vs 1,416 ms** |
| graph_ce − graph | +1.4 [+0.1, +2.7] | +3.0 [+0.9, +5.1] | 1,739 ms vs 65 ms |

I noticed the first comparison *after* seeing the results, so it is a hypothesis to confirm
on held-out data, not a finding. If it holds, the practical conclusion is that the LLM-free
graph without any reranker beat the tuned cross-encoder pipeline at about 1/22 of the latency
on this CPU. Adding the cross-encoder on top of the graph then buys only 1.4 more points for
27× the latency.

### By hop count (R@5)

| Hops (n) | hybrid | hybrid_ce | graph | graph_ce |
|---|---|---|---|---|
| 2 (518) | 59.7 | 64.0 | 66.6 | 68.6 |
| 3 (304) | 51.8 | 55.3 | 55.8 | 57.7 |
| 4 (178) | 29.4 | 33.6 | **39.2** | 37.9 |

## Generalization: 2WikiMultiHopQA (dev, 1,000 questions, **no 2Wiki tuning**)

Run `results/published/twowiki_1000_frozen_4f4c683702`. It uses exactly the MuSiQue-train-selected
settings (`configs/twowiki_1000.yaml`): 10,000 candidates → 6,222 passages, all 2,480 gold IDs
mapped, a graph of 47,487 nodes. This was a quality-only run (`--no-latency`), because latency
was measured on MuSiQue. It recorded commit `9348fe3` with a dirty tree: the uncommitted change
moved `--set` parsing into `config.py` and does not change evaluation behaviour.

| System | R@2 | R@5 | R@10 | AllGold@5 | MRR |
|---|---|---|---|---|---|
| bm25 | 55.8 | 65.7 | 70.9 | 32.6 | 93.9 |
| dense | 63.0 | 69.6 | 72.2 | 38.7 | 98.6 |
| hybrid | 62.9 | 70.0 | 73.3 | 38.7 | 98.5 |
| hybrid_ce | 62.5 | 70.3 | 74.4 | 39.2 | 98.4 |
| graph_only | 61.9 | 76.1 | **92.2** | 47.6 | 92.6 |
| **graph** | **64.9** | **77.9** | 88.2 | **51.1** | 97.5 |
| graph_ce | 63.7 | 75.4 | 84.6 | 47.6 | 98.5 |

| Comparison (R@5) | Δ | 95% CI | p (Holm) |
|---|---|---|---|
| **graph_ce − hybrid_ce** (pre-registered headline) | **+5.2** | [+4.3, +6.1] | 0.0006 |
| graph − hybrid | +7.8 | [+6.8, +8.9] | 0.0006 |
| graph_only − hybrid | +6.1 | [+4.9, +7.3] | 0.0006 |
| *exploratory:* graph − hybrid_ce | +7.6 | [+6.5, +8.8] | unadjusted |
| *exploratory:* graph_ce − graph | **−2.4** | [−3.7, −1.2] | unadjusted |
| *exploratory:* hybrid_ce − hybrid | +0.2 | [−0.7, +1.1] | unadjusted |

By hop count, R@5 for hybrid_ce → graph_ce → graph: 2-hop (n = 760) 75.7 → 80.9 → 83.2;
4-hop bridge-comparison (n = 240) 53.1 → 58.0 → 60.8.

What this shows:

- **The graph effect transfers** without any 2Wiki tuning, and it is larger than on MuSiQue. 2Wiki questions are built from Wikidata relations between named entities, which suits an NER-based graph.
- **The MS MARCO cross-encoder does not transfer.** It adds nothing to hybrid (+0.2, n.s.) and *lowers* the graph ranking by 2.4 points. On 2Wiki the best system is the graph *without* a reranker. This is an observation about this particular reranker, not about reranking in general.
- Recall@10 for graph_only is 92.2 against 73.3 for hybrid: the graph's main contribution is candidate recall.

## Effect of tuning (same 1,000 dev questions)

| | Untuned defaults (Gate-1 retest) | Tuned on train |
|---|---|---|
| hybrid_ce R@5 | 55.0 (rerank 20) | 55.9 (rerank 50) |
| graph_ce R@5 | 57.4 | 59.8 |
| **Δ graph_ce − hybrid_ce** | +2.4 [+1.6, +3.3] | **+3.9 [+2.8, +5.0]** |
| graph R@5 | 50.8 | 58.4 |

Tuning helped the baseline a little (+0.9) and the graph more (+2.4 reranked, +7.6 without
the reranker). The main gain came from settings that weight the graph ranking more heavily
(graph RRF weight 2, fewer passage seeds). Deeper reranking of the hybrid list (50 instead
of 20) did *not* close the gap, which was the main fairness worry after Gate 1.

## Gate 1: final verdict — **Go**

Pre-registered bar: Δ R@5 ≥ +3 against the strongest baseline, with the CI excluding 0.
The tuned headline is **+3.9 [+2.8, +5.0]**, with the baseline and the graph tuned under the
same protocol on train data. Scope of the claim:

- **It generalizes.** On 2WikiMultiHopQA, with the same frozen settings and no 2Wiki tuning, the headline is +5.2 [+4.3, +6.1] (section above).
- **Retrieval only.** No answer-generation quality was measured.
- **Tuning effort was not exactly equal.** The graph grid had 54 configurations, the baseline 35, and three graph settings landed on a grid edge (`docs/tuning.md`).

## Gate 2 (early read on the tuned run): is routing worth it?

Oracle router: run the graph stage only on queries where it raises R@5 (ties → skip).
Quality numbers use all 1,000 questions; latency uses the 200 timed questions.

| Pipeline | Graph helps / hurts | Graph-stage share of latency | Mean ms: always-graph → oracle | Oracle saving | R@5: base / always / oracle |
|---|---|---|---|---|---|
| hybrid → graph | 23.0% / 6.1% | **69%** | 65.3 → 32.5 | **50%** | 51.9 / 58.4 / **60.9** |
| hybrid_ce → graph_ce | 14.2% / 5.2% | 2.6% | 1,739 → 1,454 | 16% | 55.9 / 59.8 / 61.8 |

- **With the cross-encoder, routing fails the gate.** The graph stage costs 2.6% of latency, below the 20% bar. Most of the oracle's 16% saving comes from skipping the slower rerank of the graph list, not from skipping PPR.
- **Without the cross-encoder, the oracle passes both relative thresholds**, and because the graph hurts 6.1% of queries, a perfect router would also *raise* R@5 by 2.5. But the absolute saving is only about 33 ms per query, and prior work warns that learned routers recover much less than the oracle (Per-Query Gating reports AUC 0.16–0.70).
- Recommendation: a learned router is **permitted, but low priority**. If built, it should target the no-reranker pipeline, and it must recover at least 50% of the 33 ms oracle saving without losing more than 1 R@5 point against always-graph.

## Index build and memory (MuSiQue dev-1,000: 11,524 passages)

| Stage | Time | Peak RSS | Disk |
|---|---|---|---|
| Corpus (19,989 candidates → 11,524 passages) | < 1 s | — | — |
| BM25 index | 0.7 s | 107 MB | 5.3 MB |
| BGE-small passage embedding | 375.8 s (30.7 passages/s) | 2.8 GB | 17.7 MB `.npy` + 17.7 MB FAISS |
| spaCy NER (NER component only) | 71.8 s | — | — |
| Entity-name embedding (65,476 names) | 109.9 s | — | 100.6 MB |
| CSR graph (77,000 nodes, 246,736 non-zeros) + transition | 0.2 s | 988 MB (script) | 6.2 MB |
| Evaluation, all models loaded (tuned run) | — | 1.46 GB | — |

PPR costs 31.4 ms per query at 246,736 non-zeros (9.0 ms at 71,904 non-zeros on the 200-question corpus).
The cross-encoder costs about 30 ms per (query, passage) pair, which makes reranking 50
candidates (1.4 s) the dominant cost of every reranked pipeline.

## Why the reranked graph list is slower to rerank

At the same depth, reranking the graph-fused list took 1,673 ms against 1,396 ms for the
hybrid list. The graph's top-20 passages average 103.6 words against 94.3 for hybrid (corpus
mean 82.3). Longer passages mention more entities, so they have more edges, and
PageRank-style walks favour high-degree nodes. This is a measured correlation, not a
verified cause. A degree-normalized passage score is a pending ablation.

## Ablations (train tuning set, graph without the cross-encoder)

Single-change ablations against the tuned reference (graph R@5 67.3 on 500 train questions).
The full table and interpretation are in [`error_analysis.md`](error_analysis.md) §5.
Removing entity seeds costs −4.9 [−6.7, −3.2], so the entity graph carries the effect.
Passage seeds (−2.6), passage-node scoring (−1.9 vs entity-sum), title entities (−1.3) and NER
linking (−1.0) all help. Synonym edges, DATE entities and label-aware keys show no reliable
effect. Two changes **improve** on the reference: removing the n-gram linker (+1.5 [+0.2, +2.9])
and count mention weights (+1.6 [+0.5, +2.7]). They are candidates for a v2 configuration,
which must be confirmed on train and then evaluated once on held-out dev questions.

## Limitations

- MuSiQue dev-1,000 was evaluated twice (untuned, tuned). The held-out 1,417 questions have now been used once, for the locked v2 configuration, so all 2,417 MuSiQue dev questions have been seen. Any future configuration needs fresh data, for example another train-derived split, as MuSiQue's test labels are not public.
- The 2Wiki results use v1 (with the n-gram linker and binary weights); v2 has not been run on 2Wiki.
- 2Wiki latency was not measured. Its corpus (6.2k passages) is smaller than MuSiQue's (11.5k), so PPR should be cheaper, but that has not been measured.
- The headline compares *retrieval*. Whether better passages give better answers was not measured.
- The 200-question spike overestimated the no-reranker gain (+3.1 at 200, +1.1 at 1,000 untuned). Small-sample estimates should not be trusted.
- Ablations (synonym edges, specificity, DATE, title entities, n-gram linker, label-aware keys, degree normalization) are not yet run, so it is not yet known which components carry the effect.
