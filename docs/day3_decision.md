# Day-3 decision: is there a graph effect?

*2026-10-24. Every number below comes from a saved run under `results/published/`. Values
are percentages unless marked otherwise. "Δ" is a paired difference in absolute percentage points.*

## 1. What was run

| | |
|---|---|
| Data | MuSiQue-Ans **dev**, 200 questions (seeded permutation, seed 0), candidates merged into one shared corpus: 3,999 paragraphs → **3,235** passages after deduplication; all 535 gold IDs mapped; 0 records skipped |
| Hops | 104 × 2-hop, 57 × 3-hop, 39 × 4-hop |
| Run | `results/published/musique_200_day3_04db6f6105` (config hash `daf08b286474`, corpus hash `cab7f1d683b2`). The run started before the first Git commit, so its metadata records `commit: null`; the code is identical to commit `a27e215`. |
| Hardware | AMD Ryzen 7 8840U (8C/16T), 15.3 GB RAM, Windows 11, CPU only, 8 threads pinned (torch, faiss, OMP, MKL) |
| Tuning | **None.** Every setting is the pre-registered default in `configs/musique_200.yaml`. The baselines are therefore *not yet tuned* (see §8). |

Systems are defined in the README. The graph used passage–entity mention edges only: no
synonym edges, binary weights, spaCy `en_core_web_sm` NER plus passage titles, entities
keyed by normalized name, PPR α = 0.85 with a 50-iteration cap, and seeds of linked
entities × specificity mixed 50/50 with the top-5 hybrid passages.

## 2. Retrieval quality (200 questions)

| System | R@2 | R@5 | R@10 | AllGold@5 | MRR |
|---|---|---|---|---|---|
| bm25 | 32.9 | 42.9 | 50.5 | 10.5 | 69.6 |
| dense | 39.8 | 50.2 | 59.4 | 15.0 | 80.0 |
| hybrid | 37.3 | 48.7 | 61.0 | 11.5 | 79.0 |
| hybrid_ce | 41.2 | 54.7 | 63.0 | 18.0 | 83.5 |
| graph_only | 37.7 | 51.4 | **71.5** | 14.5 | 79.5 |
| graph | 39.4 | 51.8 | 69.5 | 14.5 | 80.4 |
| graph_ce | **42.2** | **57.1** | 68.1 | **21.0** | **84.4** |

Strongest baseline (highest R@5): **hybrid_ce**.

By hop count (R@5): the graph's gain over hybrid is similar at every depth (2-hop 54.8 → 58.2,
3-hop 49.7 → 52.6, 4-hop 30.8 → 33.3), so it is not concentrated in 3- and 4-hop questions.
The subsets are small (39–104 questions), so these per-hop gaps carry wide uncertainty.

## 3. Paired bootstrap (10,000 resamples, seed 0)

Primary family, all R@5, Holm-adjusted. The headline test pairs the graph with the *same*
reranking as the strongest baseline. This rule is coded in
`evaluation/report.py::primary_comparisons` and was fixed before the results were inspected.

| Comparison | Δ R@5 | 95% CI | p (Holm) |
|---|---|---|---|
| **graph_ce − hybrid_ce** (headline) | **+2.5** | [+0.7, +4.3] | 0.010 |
| graph − hybrid | +3.1 | [+1.7, +4.7] | 0.0006 |
| graph_only − hybrid | +2.8 | [−0.3, +5.8] | 0.073 |

Secondary comparisons (unadjusted): graph_ce − hybrid_ce R@10 **+5.1** [+2.8, +7.5],
AllGold@5 +3.0 [+0.5, +6.0], MRR +0.9 [−0.3, +2.4]; graph − hybrid R@10 **+8.6** [+5.8, +11.5].
hybrid − dense R@5 −1.5 [−4.6, +1.6]: untuned RRF does **not** beat dense alone at k = 5.

Per-query counts: graph vs hybrid helped 16, hurt 0, tied 184. graph_ce vs hybrid_ce helped 17, hurt 6, tied 177.

## 4. Latency (median of 5 repeats per query after 10 warm-up queries; 200 queries)

| System / stage | mean ms | p50 ms | p95 ms |
|---|---|---|---|
| bm25 | 0.6 | 0.6 | 0.7 |
| dense (encode 20.7 + FAISS 0.4) | 21.1 | 20.8 | 26.1 |
| hybrid | 21.9 | 21.5 | 26.9 |
| graph (= hybrid + link 14.3 + seed 0.2 + PPR 9.0 + fusion 0.4) | 45.7 | 42.8 | 63.8 |
| hybrid_ce (rerank 20: 648 ms) | 670 | 661 | 1,029 |
| graph_ce (rerank 20 of graph list: 742 ms) | 788 | 776 | 1,119 |

- Cross-encoder reranking is **93–97% of end-to-end latency**. The entire graph stage costs about 24 ms.
- graph_ce's rerank is about 94 ms slower than hybrid_ce's, although both rerank 20 passages. The likely reason is that the graph promotes longer passages (more tokens). This has **not been verified** yet.

## 5. Resources

| Stage | Time | Peak RSS | Disk |
|---|---|---|---|
| Corpus build (200 q) | 0.26 s | — | — |
| BM25 index | 0.24 s | 73 MB | 1.7 MB |
| BGE-small embedding of 3,235 passages | 108.6 s (29.8 passages/s) + 38 s model load | 2.82 GB | 5.0 MB `.npy` + 5.0 MB FAISS |
| spaCy NER (3,235 passages, NER only, `n_process=1`) | 21.3 s | — | — |
| Entity-name embedding (21,268 names) | 34.5 s | — | 32.7 MB |
| CSR graph build + transition matrix | 0.14 s | 757 MB (whole script) | 1.9 MB |
| Evaluation (all models loaded) | — | **1.13 GB** | — |

Graph: 24,503 nodes (3,235 passages + 21,268 entities), 35,952 mention edges (71,904 CSR
non-zeros), 11.1 entities per passage, no isolated passages. **17,939 of 21,268 entities
(84%) occur in only one passage**, so they cannot act as bridges. The largest entity
touches 369 passages. Only 3.1% of NER names receive more than one coarse label, which
supports keying entities by name alone.

## 6. Oracle router (perfect per-query choice, ties → no graph)

| Pipeline | Graph helps | Graph stage share | Mean ms: always-graph → oracle | Oracle saving | R@5 base / always / oracle |
|---|---|---|---|---|---|
| hybrid vs graph | 8.0% | 52% | 45.7 → 23.8 | **48%** | 48.7 / 51.8 / 51.8 |
| hybrid_ce vs graph_ce | 8.5% | 3% | 788 → 680 | 14% | 54.7 / 57.1 / 58.0 |

- Without the cross-encoder, the oracle saves 48% of latency, but that is **only about 22 ms per query**. The graph never hurt a query in this configuration, so always running it costs 24 ms and loses nothing.
- With the cross-encoder, the graph stage is 3% of latency. Most of the oracle's 14% saving comes from skipping the slower rerank of the graph list, not from skipping PPR.
- Random routing at the oracle's 8% rate reaches R@5 48.9, essentially the hybrid score.

Against Gate 2's thresholds, the no-reranker pipeline numerically passes (share ≥ 20%,
saving ≥ 20%), but the absolute savings are tiny. The reranker pipeline fails (share < 20%).
This is an early read on 200 questions; Gate 2 is formally evaluated in Week 3.

## 7. PPR convergence

With α = 0.85, the error shrinks by a factor of 0.85 per iteration, so reaching L1 < 10⁻⁶
needs about 85 iterations. All 200 queries therefore stopped at the 50-iteration cap
(`ppr_converged = false`). A gold-free check (`scripts/check_ppr_convergence.py`,
`results/published/ppr_convergence_musique_200.json`) compared the capped result with the fully
converged one (tolerance 10⁻¹³, mean 150 iterations). The mean L1 distance to the fixed
point was 1.8 × 10⁻⁵ (max 1.1 × 10⁻⁴). **The top-10 graph passage set was identical for
all 200 queries** (order identical for 99.5%), and the fused top-10 was identical in order
for 100%. The cap is immaterial to rankings and stays at 50.

## 8. Caveats

1. **Baselines are untuned.** BM25 k1/b, RRF k and weights, and the cross-encoder depth (20) are defaults. The spec requires a *properly tuned* baseline, tuned on train-derived development data (Week 2). For example, a larger rerank depth could close part of the gap, because graph_ce's gain is partly a recall@10 effect.
2. **200 questions is small.** The CIs are wide (±1.8 points at R@5).
3. **Deviations from the spec**, kept because each makes the LLM-free graph workable:
   - *Passage titles* are added as entity mentions. spaCy missed "Bong Joon-ho" as PERSON in a smoke test, and titles are part of the passage text that every system sees.
   - A *dictionary n-gram linker* matches query n-grams against entity names. It runs alongside NER exact match (not mentioned in the spec), because MuSiQue questions often contain lower-case or unlabelled entity mentions.
   - Entities are keyed by *name*, not name + label (3.1% label conflict, see §5). The `name_label` mode exists as an ablation.
4. The n-gram linker also links generic words that match a title ("independence", "state", "producer"). Specificity weighting limits their pull, but they are noise. This is a candidate ablation on tuning data, not a change to make against these 200 questions.
5. Retrieval gains are **not** answer-quality gains; no generation was run.

## 9. 1,000-question retest

Run `results/published/musique_1000_gate1_369e1e2ead` (commit `d8998fd`). The tree was
marked dirty only because `ingestion/musique.py` had been edited, and the evaluation does
not use that file. The configuration is identical to the spike apart from `n_questions`.
1,000 dev questions, 19,989 candidates → **11,524** passages (HippoRAG reports 11,656 for
its own 1,000-question MuSiQue sample). The first 200 questions are the spike set, and the
other 800 are fresh.

| System | R@2 | R@5 | R@10 | AllGold@5 | MRR |
|---|---|---|---|---|---|
| bm25 | 32.2 | 42.5 | 50.0 | 10.0 | 69.6 |
| dense | 36.4 | 48.5 | 56.3 | 14.7 | 75.2 |
| hybrid | 37.9 | 49.6 | 57.4 | 14.3 | 78.0 |
| hybrid_ce | 42.5 | 55.0 | 61.0 | 20.5 | 83.4 |
| graph_only | 35.4 | 50.0 | 63.3 | 13.4 | 75.2 |
| graph | 39.4 | 50.8 | **66.1** | 15.2 | 79.6 |
| graph_ce | **43.9** | **57.4** | 65.5 | **23.0** | **84.6** |

| Comparison (R@5) | all 1,000 | fresh 800 only |
|---|---|---|
| **graph_ce − hybrid_ce** (headline, Holm p = 0.0006) | **+2.4 [+1.6, +3.3]** | +2.6 [+1.7, +3.6] |
| graph − hybrid (Holm p = 0.0006) | +1.1 [+0.8, +1.6] | +0.8 [+0.5, +1.2] |
| graph_only − hybrid (Holm p = 0.46) | +0.4 [−0.7, +1.5] | −0.0 [−1.2, +1.2] |

Secondary (unadjusted): graph_ce − hybrid_ce R@10 **+4.5** [+3.4, +5.5], AllGold@5 +2.5
[+1.4, +3.6], MRR +1.2 [+0.5, +2.0]; graph − hybrid R@10 **+8.7** [+7.4, +10.0].
By hop count (R@5, hybrid_ce → graph_ce): 2-hop 63.1 → 65.6 (n = 518), 3-hop 54.6 → 56.1
(n = 304), 4-hop 32.0 → 35.7 (n = 178).

**What changed from 200 to 1,000 questions.** The no-reranker gain shrank from +3.1 to +1.1,
so the 200-question estimate was optimistic. The reranked gain held steady at +2.4. The graph
mainly improves the **candidate pool** (R@10 +8.7): passages the hybrid ranks 6–20 move into
the top 10. The cross-encoder then turns that larger pool into R@5. So whether the gain
survives a *deeper* rerank of the hybrid list (depth 50/100, Week-2 tuning) is the key open
question, because a deeper rerank also enlarges the hybrid pool.

**Latency at 11.5k passages** (200 timed queries, 5 repeats): hybrid 20.8 ms, graph 66.5 ms
(PPR 31.9 ms on a graph with 77,000 nodes and 246,736 non-zeros, linking 12.8 ms),
hybrid_ce 656 ms, graph_ce 750 ms (p95 968 / 1,036 ms). The evaluation process peaked at
**3.9 GB RSS**, up from 1.1 GB at 200 questions; the cause is not yet investigated.

**Graph at 11.5k passages:** 65,476 entities, 123,368 mention edges, 83% singleton entities,
largest entity degree 1,420, 4.0% of NER names with conflicting labels. Build: NER 71.8 s,
entity-name embedding 109.9 s, CSR build 0.2 s, 6.2 MB on disk. BGE passage embedding took
375.8 s (30.7 passages/s).

**Oracle router at 1,000 questions** (quality on all 1,000; latency on the 200 timed):

| Pipeline | Graph helps / hurts | Graph stage share | Oracle saving vs always-graph |
|---|---|---|---|
| hybrid → graph | 3.1% / 0.0% | 69% | 65% (66.5 → 23.4 ms) |
| hybrid_ce → graph_ce | 9.3% / 2.9% | 6% | 12% (750 → 662 ms) |

## 10. Gate 1 decision

Pre-registered rule: **Go** if Δ R@5 ≥ +3 with a CI excluding 0; **Inconclusive** if +1.5 to +3
or the CI touches 0 (then scale to 1,000 and retest); **Pivot** if < +1.5 on both datasets
after one improvement attempt.

| | Headline Δ R@5 | Band |
|---|---|---|
| 200 questions | +2.5 [+0.7, +4.3] | Inconclusive → retest |
| 1,000 questions | +2.4 [+1.6, +3.3] | **Inconclusive** (CI excludes 0, but the point estimate is below +3) |

**Decision: conditional continue. This is not a "Go".** The retest settled the question of
*whether* there is an effect: a small, statistically reliable gain over the strongest
untuned baseline that holds on the 800 fresh questions. It did not reach the pre-registered
+3 bar, and I will not report it as having done so. The rules have no explicit outcome for
"still inconclusive after the retest", so I chose the cheapest next step that can decide it:

1. **Week-2 fair tuning on a train-derived set** (`configs/musique_tune500.yaml`, 500 MuSiQue
   *train* questions, protocol fixed in `scripts/tune.py`). The baseline gets the same
   tuning effort as the graph, including cross-encoder depth 50/100.
2. Re-evaluate the frozen configurations once on dev-1,000.
3. **Pivot trigger:** if the tuned headline Δ falls below +1.5 R@5, or its CI includes 0,
   apply the pivot rule: try one synonym-edge or seeding fix on tuning data, add 2Wiki, then
   pivot if it still fails.

**Gate 2 early read.** The graph stage costs about 46 ms per query, which is 6% of the
reranked pipeline's latency. Selective routing can therefore save at most about 12% (88 ms) in
the configuration that actually benefits from the graph. That is below the 20% bar, so a
learned router looks unlikely to be worth building. The formal check is in Week 3.
