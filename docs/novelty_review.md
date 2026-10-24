# HopGraph — Novelty and Feasibility Review

*Written 2026-10-24, before implementation started. I verified each source in this
session by opening its arXiv abstract page, GitHub repository, or Hugging Face API
record, unless it is marked otherwise. I read abstracts and selected sections, not
every paper in full. Any claim below that rests on a search-result summary rather
than the primary page is labelled **(unverified detail)**.*

## 1. Bottom line

| Question | Answer |
|---|---|
| Is graph retrieval with Personalized PageRank (PPR) new? | **No.** HippoRAG (2024) and HippoRAG 2 (2025) do this. |
| Is passage-node + entity-node graph with passage seeding new? | **No.** HippoRAG 2 does exactly this (with an LLM for triples/filtering). |
| Is an LLM-free graph for multi-hop retrieval new? | **No.** TIGRAG (2026) builds an LLM-free token co-occurrence graph; LiteRAG (2026) removes LLM control at retrieval time. |
| Is routing between vanilla and graph RAG new? | **No.** EA-GraphRAG, R²Adapter, GraphRAG-Router, RAGRouter-Bench, Adaptive-RAG all study it. |
| Is a router trained on *retrieval gain* rather than answer correctness new? | **No.** "Per-Query Gating of LLM Rerankers for Multi-Hop Retrieval" (2026) trains a per-query gate on retrieval-gain labels with pre-call score features, Platt calibration and an oracle analysis on MuSiQue/2Wiki/HotpotQA. |
| Is a score-skewness feature new? | **No.** SkewRoute (EMNLP Findings 2025) routes on retrieval-score skewness. We must cite it if we use the feature. |
| Is a three-way allocator (nothing / graph / rerank) new? | **Not established.** Adaptive Re-Ranking (2026) routes among rerankers; Per-Query Gating gates a reranker atop a graph pipeline. I did not find a paper that jointly chooses graph expansion vs. cross-encoder reranking under one budget, but my search was limited; treat as an untested hypothesis only. |
| What *is* defensible? | An **engineering study**: a spaCy-NER (no LLM), CPU-only, HippoRAG-2-style entity–passage graph evaluated **at retrieval level** against a **tuned BM25+dense+RRF+cross-encoder baseline**, with paired bootstrap CIs, per-stage latency, peak RAM and an oracle routing bound. I found no published number for this exact configuration (absence in a limited search is not proof it doesn't exist). |

**Recommendation: GO** for the Day 1–3 spike, framed as a reproducible engineering
study. **Not** a novelty claim. The router remains an evidence-gated experiment
(Gate 2), and its methodology closely follows existing work, which we must cite.

## 2. Prior work, verified

### Graph retrieval with PPR

**HippoRAG** — Gutiérrez et al., *HippoRAG: Neurobiologically Inspired Long-Term
Memory for Large Language Models*, NeurIPS 2024. [arXiv:2405.14831](https://arxiv.org/abs/2405.14831).
Code: [OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) (MIT).
- Graph: noun-phrase nodes from **LLM OpenIE** (GPT-3.5-turbo); query NER also by LLM.
- **Node specificity** \(s_i = |P_i|^{-1}\), where \(P_i\) is the set of passages node \(i\) was extracted from. It is **multiplied into each query node's probability** before PPR (i.e., it weights *seeds*, not edges).
- PPR damping **0.5**; synonym-edge threshold **τ = 0.8** (both tuned on 100 MuSiQue train examples).
- Passage score = PPR node distribution × the |N|×|P| phrase–passage count matrix.
- Evaluation: 1,000 dev questions each for MuSiQue (corpus **11,656** passages), 2Wiki (6,119), HotpotQA (9,221); metrics R@2, R@5, AR@2, AR@5, EM/F1.

**HippoRAG 2** — Gutiérrez et al., *From RAG to Memory: Non-Parametric Continual
Learning for Large Language Models*, ICML 2025. [arXiv:2502.14802](https://arxiv.org/abs/2502.14802).
- Adds **passage nodes** linked to phrases by "contains" context edges; synonym edges (τ = 0.8); relation edges from triples.
- PPR damping **0.5**. Seeds = top phrase nodes after **LLM recognition-memory filtering** + **all passage nodes weighted by dense similarity × 0.05**.
- If no triples survive filtering, falls back to dense retrieval (no graph search).
- Reported MuSiQue Recall@5: NV-Embed-v2 **69.7**, HippoRAG 2 **74.7** (Table 3). NV-Embed-v2 is a 7B embedding model — not CPU-friendly, so our numbers will not be comparable.
- The paper does not mention node specificity (it's not used, per my read of the fetched page).

**Implication for HopGraph:** our planned graph (entity + passage nodes, passage seeding, synonym edges, specificity seeds) is essentially *HippoRAG 2 minus the LLM*, with spaCy NER instead of OpenIE. That is the honest framing. Our default PPR damping in the spec (α = 0.85) differs from HippoRAG's 0.5; we will tune it on dev data and report both.

**Microsoft GraphRAG** — Edge et al., *From Local to Global: A Graph RAG Approach to
Query-Focused Summarization*, [arXiv:2404.16130](https://arxiv.org/abs/2404.16130) (v2 Feb 2025).
Code: [microsoft/graphrag](https://github.com/microsoft/graphrag) (MIT). LLM-built entity graph +
LLM community summaries, targeting *global sensemaking* questions; evaluated on answer comprehensiveness/diversity. Not a passage-retrieval system, so not directly comparable; relevant only as context.

**TIGRAG** — Bonifazi et al., *Efficient Retrieval-Augmented Generation via Token
Co-occurrence Graphs*, [arXiv:2606.30093](https://arxiv.org/abs/2606.30093) (June 2026).
LLM-free graph from sliding-window token co-occurrence; graph expansion + neural reranker;
iterative bridging-entity query expansion; reports gains over dense and graph RAG on three
multi-hop benchmarks and lower indexing time/latency. The abstract does not mention PPR
(a search-result summary of the full text says it does — **unverified detail**). No code link on the abstract page.

**LiteRAG** — Coll Tejeda et al., [arXiv:2609.10239](https://arxiv.org/abs/2609.10239) (Sept 2026).
Replaces LLM control at retrieval time with query-conditioned algorithmic exploration,
hub penalization and adaptive thresholding. Whether indexing is LLM-free is not stated in the abstract.

### Routing between conventional and graph retrieval

**Adaptive-RAG** — Jeong et al., NAACL 2024, [arXiv:2403.14403](https://arxiv.org/abs/2403.14403).
Code: [starsuzi/Adaptive-RAG](https://github.com/starsuzi/Adaptive-RAG) (Apache-2.0).
A small LM classifier predicts query complexity (no retrieval / single-step / multi-step);
labels come from model **answer outcomes** plus dataset bias.

**EA-GraphRAG** — Dong et al., *Use Graph When It Needs*, [arXiv:2602.03578](https://arxiv.org/abs/2602.03578) (Feb 2026).
Syntactic features → lightweight complexity score → dense RAG for low scores, graph
retrieval for high scores, **complexity-aware RRF for borderline** queries. This is close to our
rule-based router + RRF fusion idea. Evaluation is answer accuracy.

**R²Adapter** — Guo et al., *A Routing and Rewriting Adapter for Efficient Hybrid RAG*,
[arXiv:2609.02894](https://arxiv.org/abs/2609.02894). The abstract page says "submitted 1 Jul 2026",
which does not match a 2609 (Sept) identifier — check before citing a date.
Lightweight router between vanilla and graph RAG; rewrites uncertain graph-routed queries;
up to **59% fewer graph-RAG calls at comparable answer accuracy** on three multi-hop QA benchmarks;
works with HippoRAG 2 and GraphRAG. Router features and label definition not stated in the abstract.
No code link on the abstract page.

**SkewRoute** — Wang et al., Findings of EMNLP 2025, [arXiv:2505.23841](https://arxiv.org/abs/2505.23841).
Training-free: skewness of the sorted retrieval-score distribution predicts query difficulty;
routes between small and large LLMs in KG-RAG; reports up to 50% fewer large-LLM calls.
*Our "skewness of top-20 scores" router feature comes from this idea.*

**GraphRAG-Router** — Fan et al., [arXiv:2604.16401](https://arxiv.org/abs/2604.16401) (Mar 2026).
Hierarchical router over heterogeneous GraphRAG frameworks *and* generator LLMs, SFT warm-up
then two-stage RL with a curriculum cost-aware reward; ~30% less large-LLM use on six QA benchmarks.

**RAGRouter-Bench** — Wang et al., [arXiv:2602.00296](https://arxiv.org/abs/2602.00296) (Jan 2026).
7,727 queries, 21,460 documents, five RAG paradigms; finds no paradigm wins everywhere.
Follow-up: *Lightweight Query Routing for Adaptive RAG: A Baseline Study on RAGRouter-Bench*,
[arXiv:2604.03455](https://arxiv.org/abs/2604.03455) — five classical classifiers × three feature
regimes (TF-IDF, MiniLM embeddings, hand-crafted structural features). Closest to our
"logistic regression on cheap features" plan. (Do not confuse with RAGRouter, arXiv:2505.23052, which routes among retrieval-augmented LLMs.)

**Han et al.** — *RAG vs. GraphRAG: A Systematic Evaluation and Key Insights*,
[arXiv:2502.11371](https://arxiv.org/abs/2502.11371) (v1 Feb 2025, v3 Mar 2026). RAG better on
single-hop/detail queries, GraphRAG better on multi-hop reasoning; complementary; explores
selection and integration strategies. Exact selection-strategy details **not verified** (I did not read the full PDF).

### Closest overlap with the router methodology

**Per-Query Gating of LLM Rerankers for Multi-Hop Retrieval** — Bacellar,
[arXiv:2609.22880](https://arxiv.org/abs/2609.22880) (Sept 2026). Gate decides whether to run an
LLM reranker on top of HippoRAG 2. Features: 27 score/lexical statistics of two retrieval
lists + PCA of a small query embedding. **Label = retrieval gain** (LastHop@K). Platt
calibration, budget-based thresholds, random-gate baseline, and a documented **correction of an
over-optimistic oracle**. Results: skips 51% of reranker calls at −1.2 pp average; AUC only 0.16–0.70.
Code and per-query results in the arXiv source archive.

**Adaptive Re-Ranking** — Genc, Korukluoglu, Allan, [arXiv:2606.25249](https://arxiv.org/abs/2606.25249) (June 2026).
Routes each query among BM25 / MiniLM-L6 / BGE-v2-m3 reranking with utility-based labels;
oracle gains are large but learning the router from limited supervision is "non-trivial".

**Implication:** a retrieval-gain-labelled, cheap-feature, calibrated router with an oracle
bound and a random-routing baseline is **existing methodology**. If HopGraph builds one, it is a
replication in a new setting (LLM-free graph, CPU), and must cite these papers. The lesson from
both papers — *oracles look great, learned routers recover much less* — is exactly why Gate 2
computes the oracle first.

### Retrieval building blocks

| Component | Source | Verified fact |
|---|---|---|
| RRF | Cormack, Clarke, Büttcher, SIGIR 2009, pp. 758–759, [DOI 10.1145/1571941.1572114](https://doi.org/10.1145/1571941.1572114), [PDF](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) | Score = Σ 1/(k + r); k = 60 was fixed in a pilot and not tuned further. |
| BM25S | Lù, [arXiv:2407.03618](https://arxiv.org/abs/2407.03618); [xhluca/bm25s](https://github.com/xhluca/bm25s) (MIT) | Eager sparse scoring: precomputes BM25 scores into sparse matrices at index time. |
| FAISS | [facebookresearch/faiss](https://github.com/facebookresearch/faiss) (MIT) | `IndexFlatIP` = exact inner-product search. |
| spaCy | [explosion/spaCy](https://github.com/explosion/spaCy) (MIT); `en_core_web_sm` model card (MIT) | |
| Sentence Transformers | [UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers) (Apache-2.0) | |
| BGE-small | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) (MIT, HF API) | 384-dim. Retrieval queries use an instruction prefix (to verify in model card during Day 1). |
| Cross-encoder | [cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) (Apache-2.0, HF API) | |
| Local-push PPR | Andersen, Chung, Lang, FOCS 2006 | **Not verified this session** — only needed if profiling shows PPR is a bottleneck. |

## 3. Datasets, verified

| Dataset | Official source | License (verified) | Notes |
|---|---|---|---|
| MuSiQue | [StonyBrookNLP/musique](https://github.com/StonyBrookNLP/musique) | **CC BY 4.0** — README states it; the repo `LICENSE` file is the CC BY 4.0 text; GitHub API reports `CC-BY-4.0`. | This resolves the blueprint's license question. Official download is a Google Drive zip via `gdown`. |
| MuSiQue mirror used | HF [`dgslibisey/MuSiQue`](https://huggingface.co/datasets/dgslibisey/MuSiQue) @ `c8f4f8c9` | (inherits CC BY 4.0) | `musique_ans_v1.0_dev.jsonl` SHA-256 `15fa6379…abf3b`, 30,439,728 bytes — **identical** in independent mirror [`bdsaglam/musique`](https://huggingface.co/datasets/bdsaglam/musique). Train file 241,046,755 bytes, SHA-256 `83a75b1e…490a`, also identical across both mirrors. I have not hash-checked against the Google Drive original. |
| 2WikiMultiHopQA | [Alab-NII/2wikimultihop](https://github.com/Alab-NII/2wikimultihop) | Repo: Apache-2.0 (GitHub API) | Not downloaded yet. |
| HotpotQA | [hotpotqa/hotpot](https://github.com/hotpotqa/hotpot) | Repo: Apache-2.0 (GitHub API). The dataset itself is commonly cited as CC BY-SA 4.0 — **not verified this session**. | Optional. |
| HippoRAG released samples | HF [`osunlp/HippoRAG_v2`](https://huggingface.co/datasets/osunlp/HippoRAG_v2); `osunlp/HippoRAG_2` has identical files | **No license declared** on the HF dataset card (only `language: en`). Code repo is MIT. | `musique.json` 12.5 MB, `musique_corpus.json` 6.2 MB; `2wikimultihopqa.json` 6.5 MB + corpus 3.1 MB. Don't redistribute; use only as a cross-reference. |

**MuSiQue-Ans dev, measured from the file:** 2,417 questions; 2,401 have 20 candidate
paragraphs, 16 have 17–19; 2–4 supporting paragraphs (1,252 / 760 / 405); hop types
`2hop` 1,252, `3hop1` 568, `3hop2` 192, `4hop1` 246, `4hop2` 64, `4hop3` 95. Fields:
`id, paragraphs[idx,title,paragraph_text,is_supporting], question, question_decomposition[id,question,answer,paragraph_support_idx], answer, answer_aliases, answerable`.

MuSiQue's official test split has no public labels, so our "held-out test" must be carved
from **train** (dev is used for tuning *and* final reporting in HippoRAG's protocol). We will
document the split protocol explicitly: tune on a train-derived dev subset, report on official dev.

## 4. Feasibility on this machine

- AMD Ryzen 7 8840U (8 cores / 16 threads), 15.3 GB RAM, 351 GB free disk, Windows 11, no GPU.
- All models are small (BGE-small ~33M params, MiniLM-L6 cross-encoder ~22M params, spaCy sm ~12 MB).
- The HippoRAG-scale MuSiQue corpus (~11.6k passages for 1,000 questions) easily fits in RAM.
- Main risk is **quality**, not compute: spaCy NER misses many bridge entities that an LLM OpenIE would catch (works, songs, events often unlabeled), so the graph effect may be small. Gate 1 tests this directly.

## 5. Illustrative examples (fact-checked)

| Example | Chain | Status |
|---|---|---|
| Capital of the country where Parasite's director was born | Parasite → Bong Joon-ho → born Daegu, South Korea (14 Sep 1969) → Seoul | Verified via search (Wikipedia, TMDB). |
| Founded first: Sundar Pichai's undergrad institute or Stanford? | Pichai → B.Tech, IIT Kharagpur → est. 1950 (as Eastern Higher Technical Institute) / first session Aug 1951 | IIT KGP part verified (Wikipedia; sources disagree 1950 vs 1951). Stanford's 1885 founding is from general knowledge, **not re-verified** this session. Stanford is older under either IIT date. |
| Composer for the film where SRK coached a women's hockey team | Shah Rukh Khan → Chak De! India (2007) → Salim–Sulaiman | Verified via search (Wikipedia soundtrack page). |

These are documentation examples only; nothing in the pipeline encodes them.

## 6. Go / no-go

**GO** — with these commitments:
1. Frame HopGraph as an LLM-free, CPU-only *replication/engineering study* of HippoRAG-2-style graph retrieval against a strong tuned baseline.
2. Cite HippoRAG/HippoRAG 2 for the graph design and specificity weighting; SkewRoute for skewness features; Per-Query Gating and Adaptive Re-Ranking for retrieval-gain routing with oracle analysis; EA-GraphRAG and R²Adapter for vanilla-vs-graph routing.
3. Pass Gate 1 before investing in graph ablations; pass Gate 2 before writing any router code.
4. Never claim novelty for routing, PPR, or retrieval-gain labels.
