# Results

*Every number comes from a saved run in `results/published/`. R@k, AllGold and MRR are
percentages. Δ is a paired difference in absolute points with a 95% paired-bootstrap CI
(10,000 resamples, seed 0). Hardware: AMD Ryzen 7 8840U, 8 threads pinned, 15.3 GB RAM,
CPU only, Windows 11.*

## Headline (MuSiQue-Ans dev, 1,000 questions, tuned on train)

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

- **One dataset so far.** The 2WikiMultiHopQA generalization run uses the same frozen settings (no 2Wiki tuning) and is in progress.
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

- Results so far cover one dataset (MuSiQue) and 1,000 dev questions. The dev set is also where HippoRAG-style papers report results, and MuSiQue's test labels are not public.
- The headline compares *retrieval*. Whether better passages give better answers was not measured.
- The 200-question spike overestimated the no-reranker gain (+3.1 at 200, +1.1 at 1,000 untuned). Small-sample estimates should not be trusted.
- Ablations (synonym edges, specificity, DATE, title entities, n-gram linker, label-aware keys, degree normalization) are not yet run, so it is not yet known which components carry the effect.
