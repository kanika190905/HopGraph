# HopGraph

**An LLM-free, CPU-first study of graph-enhanced multi-hop passage retrieval and adaptive retrieval costs.**

> On a CPU-only laptop, does an LLM-free entity graph with Personalized PageRank improve
> multi-hop passage retrieval over a properly tuned hybrid retrieval system with a
> cross-encoder reranker? And if so, can cheap features predict which queries benefit
> enough to justify the extra cost?

HopGraph is an **information-retrieval** project. It returns ranked evidence passages and
evaluates them against gold supporting passages. It does **not** generate answers.

**Status (2026-10-25): result confirmed on held-out data.** All hyper-parameters were chosen
on MuSiQue **train** data. The configuration was then locked and evaluated **once** on the
1,417 MuSiQue-Ans dev questions that no decision had touched:

| | Recall@5 | AllGold@5 | mean latency (CPU, 8 threads) |
|---|---|---|---|
| Hybrid (BM25 + dense, RRF) | 52.5 | 17.4 | 20 ms |
| Hybrid + cross-encoder (top 50) | 56.7 | 22.1 | 1,391 ms |
| **Graph-fused** (+ spaCy entity graph, PPR) | **59.9** | 26.4 | **75 ms** |
| Graph-fused + cross-encoder | 59.8 | **27.6** | 1,683 ms |

- **Pre-registered headline**, graph + CE vs hybrid + CE: **+3.1 Recall@5, 95% CI [+2.2, +4.1]**.
- **Pre-registered secondary**, graph-fused *without any reranker* vs tuned hybrid + CE:
  **+3.2 [+1.8, +4.5]**, at about **19× lower latency**.
- **Generalization:** on 2WikiMultiHopQA, with MuSiQue-tuned settings and no 2Wiki tuning,
  the headline is +5.2 [+4.3, +6.1]. The MS MARCO cross-encoder does not help there.

The full report is in [`docs/results.md`](docs/results.md), the protocol in
[`docs/heldout_protocol.md`](docs/heldout_protocol.md), and ablations and error analysis in
[`docs/error_analysis.md`](docs/error_analysis.md). [`docs/day3_decision.md`](docs/day3_decision.md)
shows how the decision evolved from the 200-question spike, where the estimate was inconclusive.
These are retrieval metrics; answer quality was not measured.

![Recall@5 vs latency](results/published/musique_heldout1417_heldout_e21b488fc6/recall_vs_latency.png)

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

Settings shown are the locked held-out configuration (`configs/musique_heldout1417.yaml`),
chosen on train data. The untuned defaults appear in `configs/musique_200.yaml`.

| System | Pipeline |
|---|---|
| `bm25` | bm25s (k1 = 0.9, b = 0.75, Snowball stemming, English stop words) |
| `dense` | BGE-small-en-v1.5 + FAISS `IndexFlatIP` (exact cosine) |
| `hybrid` | RRF(BM25, dense), k = 10, dense weight 1.5 |
| `hybrid_ce` | hybrid → MiniLM-L6 cross-encoder over the top 50 |
| `graph_only` | link query entities (spaCy NER, embedding fallback, noun chunks) → seed PPR (entities × specificity, plus 25% mass on the top-3 hybrid passages; α = 0.85) on an entity–passage graph with mention-count edge weights → rank passage nodes |
| `graph` | RRF(hybrid, graph_only) with weights 1 : 2 |
| `graph_ce` | graph → cross-encoder over the top 50 |

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

## Reproduce

Quick check: the Day-3 spike on 200 questions takes about 35 minutes.

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py scripts\download_data.py musique_dev                        # 30 MB, SHA-256 verified
& $py scripts\tasks.py reproduce --config configs\musique_200.yaml --tag day3
& $py -m pytest -q                                                # no model downloads needed
```

Headline result: tune on train, then evaluate once on dev-1,000. Allow about 5 hours on 8
cores, most of it cross-encoder scoring.

```powershell
& $py scripts\download_data.py musique_train                      # 241 MB, SHA-256 verified
foreach ($c in "configs\musique_tune500.yaml", "configs\musique_1000.yaml") {
  & $py scripts\build_corpus.py --config $c; & $py scripts\embed.py --config $c; & $py scripts\build_graph.py --config $c
}
& $py scripts\tune.py --config configs\musique_tune500.yaml       # writes configs\frozen\musique_tune500_frozen.yaml
& $py scripts\build_bm25.py --config configs\musique_1000_tuned.yaml
& $py scripts\run_eval.py   --config configs\musique_1000_tuned.yaml --tag final
& $py scripts\make_plots.py results\<run dir>
```

`configs/musique_1000_tuned.yaml` is `musique_1000.yaml` with the frozen overrides merged in.
2WikiMultiHopQA (`configs/twowiki_1000.yaml`) needs the official `data.zip` (246 MB, linked
from the Alab-NII/2wikimultihop README); extract `data/dev.json` to `data/raw/2wiki/dev.json`.

Each run writes `results/<name>_<run_id>/` containing the exact config, Git commit,
corpus hash, environment, per-query JSONL (ranked IDs, metrics, stage timings, linking
path), metric/latency/bootstrap CSVs and `summary.json` (strongest baseline, oracle router,
linking statistics). The run ID is a hash of config + commit + data, so changing any of
them produces a new directory.

## Repository layout

```
src/hopgraph/   ingestion/  retrieval/  graph/  evaluation/  pipeline.py  config.py
scripts/        download_data, build_corpus, build_bm25, embed, build_graph, run_eval, tune, make_plots, error_analysis, tasks
configs/        one YAML per experiment (unknown keys are errors)
tests/          78 tests (incl. PPR vs NetworkX, end-to-end on a hand-labelled corpus); CI runs them without torch
docs/           results, day3_decision, tuning, novelty_review, foundations, architecture, adr/, CHECKLIST, THIRD_PARTY
```

## Data and licenses

MuSiQue (CC BY 4.0) is downloaded by script and **never committed**. Model and library
licenses are listed in [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md).
