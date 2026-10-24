# Technical foundations

This page explains the ideas HopGraph is built on. Each equation is written to match
what the code actually computes, and each section points to the file that implements it.

## 1. Retrieval versus full RAG

*Retrieval-augmented generation* (RAG) has two halves:

1. **Retrieval** finds passages likely to contain the answer.
2. **Generation** asks a language model to write an answer from those passages.

HopGraph covers **only the first half**. Its output is a ranked list of passage IDs, and
it is evaluated by checking whether the *gold* supporting passages appear near the top.
Better retrieval does not automatically give better answers: a generator might already
cope with a missing passage, or it might still fail when every passage is present. So
HopGraph never claims answer-quality gains from retrieval metrics alone.

## 2. Multi-hop questions and bridge entities

A *multi-hop* question needs facts from several passages that are linked through an
intermediate (*bridge*) entity that the question never names.

| Question | Chain (each arrow is one passage) |
|---|---|
| What is the capital of the country where the director of *Parasite* was born? | *Parasite* → **Bong Joon-ho** → born in Daegu, **South Korea** → capital **Seoul** |
| Which was founded first: the institute where Sundar Pichai did his undergraduate degree, or Stanford University? | Sundar Pichai → **IIT Kharagpur** → first session 1951 (est. 1950 by some sources) — compare with Stanford, founded 1885 |
| Who composed the music for the film in which Shah Rukh Khan played a women's hockey coach? | Shah Rukh Khan → ***Chak De! India*** → **Salim–Sulaiman** |

(Facts were checked during the novelty review; see `docs/novelty_review.md` §5. These
examples are documentation only and appear nowhere in the retrieval code.)

**Why keyword and dense retrieval miss evidence.** The passage about Bong Joon-ho's
birthplace never mentions *Parasite*, "capital" or "country". BM25 scores it only on
overlapping words, and a dense encoder puts it near the question only if their meanings
happen to be close. The *bridge* (Bong Joon-ho) links the two passages, but neither
retriever follows links. A graph that connects passages through shared entities can.

## 3. BM25 (`src/hopgraph/retrieval/bm25.py`)

BM25 is a bag-of-words ranking function. bm25s's `lucene` variant, which HopGraph uses, computes

$$
\text{score}(q,d)=\sum_{t\in q}\text{IDF}(t)\cdot
\frac{tf(t,d)}{tf(t,d)+k_1\left(1-b+b\,\frac{|d|}{\text{avgdl}}\right)},
\qquad
\text{IDF}(t)=\ln\!\left(1+\frac{N-df(t)+0.5}{df(t)+0.5}\right)
$$

- $tf(t,d)$: how often term $t$ occurs in passage $d$; $df(t)$: how many passages contain $t$; $N$: corpus size.
- **$k_1$ (default 1.5)** controls *saturation*. The 10th occurrence of a word adds much less than the 1st. Small $k_1$ saturates quickly; large $k_1$ keeps rewarding repetition.
- **$b$ (default 0.75)** controls *length normalization*. $b=1$ fully penalizes long passages for containing more words; $b=0$ ignores length.
- **Versus TF-IDF.** TF-IDF multiplies raw term frequency by IDF, so repeating a word keeps raising the score linearly and long documents win. BM25 adds saturation ($k_1$) and length normalization ($b$).
- The classic Robertson formula has an extra $(k_1+1)$ factor in the numerator. It multiplies every score by the same constant, so it changes no ranking, and bm25s omits it.

Text is lower-cased, English stop words are removed, and tokens are stemmed with the
Snowball English stemmer (PyStemmer).

## 4. Embeddings and vector search (`src/hopgraph/retrieval/dense.py`)

A sentence encoder (BGE-small, 384 dimensions) maps text to a vector $e$. HopGraph
normalizes every vector to unit length ($\lVert e\rVert = 1$), so

$$
\text{sim}(q,d)=e_q^\top e_d=\lVert e_q\rVert\lVert e_d\rVert\cos\theta=\cos\theta .
$$

The inner product therefore *is* cosine similarity. FAISS `IndexFlatIP` computes the exact
inner product with every stored vector: there is no approximation, and results are
deterministic for fixed inputs. `DenseRetriever.build` refuses embeddings whose norms are
not 1, because otherwise inner product and cosine would silently differ.

Following the BGE v1.5 model card, queries (but not passages) are prefixed with
`Represent this sentence for searching relevant passages: `.

## 5. Reciprocal Rank Fusion (`src/hopgraph/retrieval/fusion.py`)

RRF merges ranked lists using only ranks, not scores:

$$
\text{RRF}(d)=\sum_{r}\frac{w_r}{k+\text{rank}_r(d)}
$$

Ranks start at 1, and a passage missing from list $r$ gets nothing from it. BM25 and
cosine scores are on incomparable scales, which is why rank fusion is convenient.
$k=60$ comes from Cormack et al. (SIGIR 2009), who fixed it in a pilot study. A larger $k$
flattens the difference between rank 1 and rank 10; a smaller $k$ trusts each list's top
ranks more. HopGraph treats $k$ and the weights $w_r$ as tunable on development data.

*Worked example.* List A = [x, y, z], list B = [y, w], k = 60:
y = 1/62 + 1/61 = 0.03252, x = 1/61 = 0.01639, w = 1/62 = 0.01613, z = 1/63 = 0.01587.
Fused order: y, x, w, z. (This exact case is a unit test in `tests/test_fusion_metrics.py`.)

## 6. Cross-encoder reranking (`src/hopgraph/retrieval/rerank.py`)

A *bi-encoder* (Section 4) embeds the query and the passage separately. A *cross-encoder*
reads them together as one input, `[CLS] query [SEP] passage`, so every query token can
attend to every passage token. It is far more accurate, but it needs one full transformer
pass **per candidate**, so it reranks only the top-$n$ hybrid candidates. On this laptop,
reranking 20 MuSiQue candidates with MiniLM-L6 takes about 0.6 s per query (measured; see
results). That makes it the most expensive stage by far.

## 7. The entity graph (`src/hopgraph/graph/`)

**Nodes.** There is one node per deduplicated passage and one per normalized entity name.
Entities come from spaCy `en_core_web_sm` NER (no LLM), plus each passage's Wikipedia-style
title with its parenthetical disambiguator removed ("Green (Steve Hillage album)" → "green").

**Normalization.** Names are NFKC-normalized and lower-cased; possessive 's, periods and
apostrophes are dropped; other punctuation becomes a space; a leading "the" is removed;
and whitespace is collapsed. So "The Beatles" → "beatles" and "Bong Joon-ho" → "bong joon ho".

**Edges.** Passage–entity *mention* edges are stored symmetrically, with binary or count
weights. Optional entity–entity *synonym* edges link names whose embedding cosine is at
least τ. The adjacency matrix $A$ is a SciPy CSR matrix with passages at indices $[0,P)$ and
entities at $[P, P+E)$.

**Hub control.** A name mentioned in many passages, such as "united states", spreads
probability everywhere. HippoRAG's *node specificity* down-weights it:

$$
\text{specificity}(e)=\frac{1}{\#\{\text{passages mentioning } e\}}
$$

As in HippoRAG, HopGraph uses specificity to scale **seed weights**, not edge weights.

## 8. Personalized PageRank (`src/hopgraph/graph/ppr.py`)

Picture a random walker that starts at the query's seed nodes. At each step it either
follows an edge (probability $\alpha$) or jumps back to a seed (probability $1-\alpha$).
Nodes it visits often are relevant to the query, *including nodes several hops away*.

Let $A_{uv}$ be the weight of edge $u\to v$ and $d_u=\sum_v A_{uv}$. The column-stochastic
transition matrix is $M=(D^{-1}A)^\top$, so $M_{vu}=A_{uv}/d_u$ is the probability of
stepping from $u$ to $v$. With a seed vector $s$ ($s\ge 0$, $\sum s = 1$):

$$
p_{t+1}=\alpha\left(Mp_t+\Big(\sum_{u:\,d_u=0}p_t[u]\Big)s\right)+(1-\alpha)s,\qquad p_0=s .
$$

- **Dangling nodes** ($d_u=0$, for example a passage with no extracted entity) have an all-zero column in $M$. Without the middle term their probability mass would leak away. HopGraph returns it to the seeds, so $\sum p_t = 1$ at every step. This is the same fixed point as `networkx.pagerank(G, alpha, personalization=s, dangling=s)`, which 24 randomized tests check, including disconnected and dangling-node graphs.
- **Convergence.** The error shrinks by a factor of $\alpha$ per step, so reaching an L1 change below $10^{-6}$ takes about $\ln(10^{-6})/\ln\alpha$ iterations: 85 for $\alpha=0.85$, but only 20 for HippoRAG's $\alpha=0.5$. Under the pre-registered cap of 50 iterations, $\alpha=0.85$ stops at an L1 change of roughly $10^{-4}$, and the result records `converged=False`. Whether that changes rankings is measured in `docs/day3_decision.md`.
- **Seeds.** Each linked query entity gets weight $\text{specificity}(e)\times\text{matchSimilarity}(e)$. The top-$m$ hybrid passages get weight proportional to their RRF score. Each part is normalized to sum to 1, then the two are mixed: $s=(1-\beta)s_{\text{ent}}+\beta s_{\text{pas}}$, with default $\beta=0.5$, $m=5$. If no entity links, $s=s_{\text{pas}}$.
- **From node scores to passages.** In the default `direct` mode, passages are graph nodes and are ranked by their own PPR score. The `entity_sum` ablation ranks passage $i$ by $\sum_e \text{PE}_{ie}\,p_e$, as HippoRAG 1 does.
- The final graph-enhanced ranking is RRF(hybrid, PPR ranking). If PPR scores fewer passages than the requested depth, the list is filled from the hybrid ranking.

## 9. Evaluation metrics (`src/hopgraph/evaluation/metrics.py`)

For gold set $G$ and ranked list $L$:

- $\text{Recall@}k = |G\cap\text{top}_k(L)|/|G|$: the fraction of supporting passages found.
- $\text{AllGold@}k = 1$ if every gold passage is in the top $k$. This is stricter, and it is what a reader needing every hop requires.
- RR $=1/\text{rank}$ of the first gold passage (0 if none is retrieved); MRR is its mean.
- Binary nDCG@k rewards gold passages that appear earlier.

All metrics are macro-averaged over questions.

**Paired bootstrap.** To compare systems A and B on the *same* questions, HopGraph resamples
the question set with replacement 10,000 times (fixed seed) and recomputes
$\text{mean}(A)-\text{mean}(B)$ each time. The 2.5th and 97.5th percentiles form a 95% interval.
Pairing matters: a question that is hard for both systems cancels out.
