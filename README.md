# HopGraph

**An LLM-free, CPU-first study of graph-enhanced multi-hop passage retrieval and adaptive retrieval costs.**

> On a CPU-only laptop, does an LLM-free entity graph with Personalized PageRank improve
> multi-hop passage retrieval over a properly tuned hybrid retrieval system with a
> cross-encoder reranker? And if so, can cheap features predict which queries benefit
> enough to justify the extra cost?

HopGraph is an **information-retrieval** project. It returns ranked evidence passages and
evaluates them against gold supporting passages. It does **not** generate answers.

**Status: Gate 1 retested; Week-2 tuning in progress (2026-10-24).** Baselines, the graph
retriever and the evaluation harness work and are tested. With **untuned** defaults on 1,000
MuSiQue dev questions, the graph + cross-encoder pipeline beats hybrid + cross-encoder by
**+2.4 Recall@5 (95% CI [+1.6, +3.3])**. That is reliable, but below the pre-registered +3
"Go" bar, so the next step is to check whether the gain survives fair baseline tuning. See
[`docs/day3_decision.md`](docs/day3_decision.md). Nothing here is a final result.

## What is (and is not) new

Graph retrieval with Personalized PageRank (HippoRAG, HippoRAG 2), LLM-free graph RAG
(TIGRAG), routing between vanilla and graph RAG (EA-GraphRAG, R²Adapter, SkewRoute), and
retrieval-gain-labelled routers (Per-Query Gating of LLM Rerankers) are all **prior work**;
see [`docs/novelty_review.md`](docs/novelty_review.md). HopGraph's contribution is an
**engineering study**: a HippoRAG-2-style entity–passage graph built with spaCy NER instead
of an LLM, compared at retrieval level against a strong BM25 + dense + RRF + cross-encoder
baseline, with paired bootstrap intervals, per-stage latency, peak RAM and an
oracle-routing upper bound, all on a 16 GB CPU laptop.

## Systems compared

| System | Pipeline |
|---|---|
| `bm25` | bm25s (k1 = 1.5, b = 0.75, Snowball stemming, English stop words) |
| `dense` | BGE-small-en-v1.5 + FAISS `IndexFlatIP` (exact cosine) |
| `hybrid` | RRF(BM25, dense), k = 60 |
| `hybrid_ce` | hybrid → MiniLM-L6 cross-encoder over the top 20 |
| `graph_only` | link query entities → seed PPR (entities × specificity + top-5 hybrid passages) → rank passage nodes |
| `graph` | RRF(hybrid, graph_only) |
| `graph_ce` | graph → cross-encoder over the top 20 |

## Setup (Windows PowerShell, CPU only)

Tested with Python 3.12.10 on Windows 11. (The spec asked for 3.11, which is not installed
on the development machine; every dependency has native 3.12 Windows wheels.)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
```

The lock file pins the CPU-only PyTorch build (`--extra-index-url https://download.pytorch.org/whl/cpu`)
and the spaCy `en_core_web_sm` 3.8.0 wheel. The models download on first use: BGE-small
~127 MB, MiniLM cross-encoder ~87 MB, spaCy ~13 MB.

## Reproduce the Day-3 spike

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py scripts\download_data.py musique_dev                        # 30 MB, SHA-256 verified
& $py scripts\build_corpus.py --config configs\musique_200.yaml   # passages/questions/manifest
& $py scripts\build_bm25.py   --config configs\musique_200.yaml
& $py scripts\embed.py        --config configs\musique_200.yaml   # ~2 min on 8 cores
& $py scripts\build_graph.py  --config configs\musique_200.yaml   # spaCy NER + CSR graph
& $py scripts\run_eval.py     --config configs\musique_200.yaml --tag day3
& $py -m pytest -q                                                # no model downloads needed
```

Each run writes `results/<name>_<run_id>/` containing the exact config, Git commit,
corpus hash, environment, per-query JSONL (ranked IDs, metrics, stage timings, linking
path), metric/latency/bootstrap CSVs and `summary.json` (strongest baseline, oracle router,
linking statistics). The run ID is a hash of config + commit + data, so changing any of
them produces a new directory.

## Repository layout

```
src/hopgraph/   ingestion/  retrieval/  graph/  evaluation/  pipeline.py  config.py
scripts/        download_data, build_corpus, build_bm25, embed, build_graph, run_eval
configs/        one YAML per experiment (unknown keys are errors)
tests/          73 tests, including PPR vs NetworkX and end-to-end on a hand-labelled corpus
docs/           novelty_review, foundations, day3_decision, CHECKLIST, THIRD_PARTY
```

## Data and licenses

MuSiQue (CC BY 4.0) is downloaded by script and **never committed**. Model and library
licenses are listed in [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md).
