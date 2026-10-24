# Interview preparation

Every number below comes from `docs/results.md` (MuSiQue dev-1,000, tuned on train). If a
number changes after the 2Wiki run or the ablations, update it here. Never quote a number
that is not in a saved run.

## One-line pitch

A CPU-only, LLM-free retrieval system that tests whether an entity graph with Personalized
PageRank improves multi-hop passage retrieval over a tuned hybrid baseline, and measures
whether selective graph execution is worth its cost.

## 60-second version

Multi-hop questions need passages that never mention the question's words. The bridge entity
connects them, but neither keyword search nor embeddings follow that link. I built an entity
graph with spaCy, no LLM, linking passages that share entities, and ran Personalized PageRank
from the query's entities to pull in passages one or two hops away. I compared it against a
properly tuned BM25 + dense + cross-encoder baseline on MuSiQue, tuning everything on train
data and evaluating once on 1,000 dev questions. The graph raised Recall@5 by 3.9 points over
the tuned reranked baseline, with a 95% bootstrap interval of +2.8 to +5.0. The graph without
any reranker matched or beat the cross-encoder pipeline at about 65 ms instead of 1.4 s on my
laptop CPU, though I flag that comparison as exploratory. The idea is not new: it follows
HippoRAG. My contribution is the LLM-free, CPU-only engineering and a careful, uncertainty-aware evaluation.

## Three-minute technical version

1. **Data.** MuSiQue-Ans gives each question 20 candidate paragraphs. I merge candidates from 1,000 questions into one corpus, deduplicate by content hash (19,989 → 11,524 passages), and map gold IDs, failing loudly on any miss.
2. **Baselines.** bm25s, BGE-small with a FAISS flat inner-product index (normalized vectors, so the inner product is the cosine), RRF fusion, and a MiniLM cross-encoder over the top 50.
3. **Graph.** Passage and entity nodes, mention edges, stored as a 77,000-node SciPy CSR matrix with 246,736 non-zeros. Entities come from spaCy NER plus passage titles.
4. **Query time.** Link query entities (NER exact match, a dictionary n-gram match, embedding fallback). Seed PPR with the entities weighted by specificity (1/document frequency) plus the top-3 hybrid passages. Run sparse power iteration (about 31 ms). Rank passage nodes and fuse with hybrid via RRF.
5. **Evaluation.** Recall@2/5/10, AllGold@5, MRR; paired bootstrap with Holm correction; per-stage latency as the median of 5 repeats with pinned threads; RSS sampled every 100 ms.
6. **Decisions.** A pre-registered gate said "inconclusive" at +2.4 before tuning, so I tuned both systems fairly on train data, which gave +3.9. An oracle router showed that routing can save at most 2.6% of latency when the cross-encoder is used, so I did not build a learned router for that pipeline.

## Likely questions, with answers tied to the code

**BM25 vs TF-IDF?** BM25 saturates term frequency (k1) and normalizes for length (b). TF-IDF
grows linearly with repetition. bm25s's Lucene variant drops the (k1+1) numerator because it
rescales every score equally. See `retrieval/bm25.py`. Tuning picked k1 = 0.9, but that changed
the tuning-set score by only 0.02 points.

**Why normalize embeddings? Why IndexFlatIP?** For unit vectors the inner product equals the
cosine. Flat means exact search with no approximation error, and at 11.5k × 384 float32 the
vectors take 17.7 MB, so approximate indexes would buy nothing. `DenseRetriever.build` refuses
unnormalized vectors.

**RRF and k?** score = Σ w/(k + rank). It is rank-based, so BM25 and cosine scales never need
calibrating. Cormack et al. fixed k = 60 in a pilot. Tuning on train chose k = 10 with dense
weight 1.5, which lifted hybrid R@5 by 1.9 on the tuning set.

**PPR math, and what is a dangling node?** p ← α(Mp + dangling_mass·s) + (1−α)s, where M is
column-stochastic. A dangling node has no out-edges (for example, a passage with no entities),
so its column is zero and its probability mass would leak away. I return that mass to the seeds,
which matches NetworkX; 24 randomized tests check this. Convergence needs about
ln(1e-6)/ln(0.85) ≈ 85 iterations, while I cap at 50. I checked that the cap leaves the top-10
unchanged for 200/200 queries.

**Why CSR instead of NetworkX?** PPR is one sparse mat-vec per iteration. CSR does that in
about 0.6 ms at 246k non-zeros (31 ms for 50 iterations), whereas a Python-level graph
traversal would be orders of magnitude slower. NetworkX is used as the test oracle.

**Entity-linking errors and hubs?** spaCy-sm misses many entities (it missed "Bong Joon-ho" in
a smoke test), so I added titles and dictionary n-gram matching. Hubs such as countries spread
probability everywhere, which is why seeds are weighted by specificity. Error analysis: on
average only 0.6% of the corpus is one hop from the linked entities, but 42% is within three
steps. Hubs make "reachable" almost meaningless, which motivates degree-normalization ablations.

**Why is the cross-encoder slow?** It runs one transformer pass per (query, passage) pair,
about 30 ms per pair here, so 50 candidates take about 1.4 s. That is about 98% of the
hybrid + cross-encoder pipeline's latency (1,396 of 1,416 ms).

**Router labels and leakage?** A label is 1 if graph R@5 > hybrid R@5 (ties are 0), generated
only on train. The tuning script refuses non-train data. The oracle router computes the upper
bound first: with the cross-encoder, the graph stage is 2.6% of latency, so routing cannot pay off.

**Paired bootstrap?** Resample questions with replacement and recompute mean(A) − mean(B) on
the same resampled questions, 10,000 times. Pairing cancels question difficulty. The
percentile interval is the 95% CI, and Holm adjusts the family of primary tests.

**p95 and memory?** Each query is repeated 5 times after a warm-up and the median is taken,
then p50/p95 are computed across queries. A daemon thread samples RSS every 100 ms and aborts
above 12 GB. Peak RSS in the tuned evaluation was 1.46 GB.

**Scaling to millions of passages?** At 1M passages, the dense vectors would take 1.5 GB, so
move to IVF/HNSW. The graph stays sparse (about 11 entities per passage), but full-graph PPR
grows with the number of non-zeros, so switch to local-push PPR (Andersen–Chung–Lang) or a
seed-centred subgraph. NER indexing at about 160 passages/s (11,524 passages in 72 s) would
take roughly 1.7 h per million passages with one process. I have not measured any of this; it
is a plan.

## Resume bullets (filled only with measured numbers)

- Built an LLM-free multi-hop retrieval system (spaCy entity graph, SciPy sparse Personalized PageRank, RRF fusion) that improved MuSiQue Recall@5 by **+3.9 points (95% CI [+2.8, +5.0])** over a tuned BM25 + dense + cross-encoder baseline, running on a CPU laptop at **65 ms** mean latency without a reranker.
- Developed a reproducible evaluation harness with paired-bootstrap intervals and Holm correction, per-stage p50/p95 latency and peak-RSS tracking. Oracle analysis showed that selective graph routing could save at most **16%** of latency in the reranked pipeline, which led me to deprioritize a learned router.
