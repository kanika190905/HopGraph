# Implementation checklist

Status key: **done** = exists in the repo *and* has a passing test or a saved run ·
**partial** · **pending** · **blocked** (with reason).

Last updated: 2026-10-24, after the Gate-1 retest and during Week-2 tuning.

## Environment and repository

| Item | Status | Evidence / notes |
|---|---|---|
| Python env (3.12.10 venv; 3.11 not installed) | done | `requirements.lock.txt`; ADR 0001 §1 |
| CPU-only PyTorch (2.14.1+cpu) | done | `torch.cuda.is_available() == False` |
| Native Windows install of faiss-cpu / spaCy / bm25s | done | no WSL needed |
| Git repository | done | Git is **not on PATH**; GitHub Desktop's bundled git was used. Install Git for Windows, or set `GIT_EXECUTABLE` so run metadata records the commit. |
| `pyproject.toml`, ruff | done | `ruff check` clean |
| GitHub Actions CI (Ubuntu + Windows, no torch/models) | partial | workflow written; the identical dependency set passes all 75 tests locally in a torch-free venv. **Not yet run on GitHub** (no remote configured). |
| Makefile + `scripts/tasks.py` (Windows equivalent) | done | `tasks.py reproduce` ran the full 1,000-question pipeline |
| mypy on stabilized modules | pending | configured, not yet run |
| Fresh-clone reproduction | pending | |

## Core implementation

| Item | Status | Evidence |
|---|---|---|
| MuSiQue ingestion, dedupe, fail-loud gold mapping | done | `tests/test_ingestion.py`; 200 / 1,000 / 500-train manifests; 0 skipped records |
| `.txt`/`.md` user-document chunking | pending | |
| BM25 (bm25s) | done | tests + runs |
| Dense (BGE-small, FAISS IndexFlatIP, cached `.npy`) | done | save/load equivalence test |
| Hybrid RRF | done | hand-computed test |
| Cross-encoder rerank | done | depth 20 evaluated; depths 50/100 in tuning |
| spaCy entity extraction + title entities | done | tests (stub NER) + runs |
| CSR entity–passage graph, save/load | done | `tests/test_graph.py` |
| Synonym edges | done (code) | test only; ablation pending |
| Sparse PPR with dangling handling | done | 24 randomized NetworkX comparisons; convergence check |
| Query linking with logged fallbacks | done | test + link-path stats in every run |
| Graph/hybrid fusion with hybrid fill | done | end-to-end tests |
| Router | pending, evidence-gated | early Gate-2 read is negative (graph stage = 6% of reranked latency) |

## Evaluation

| Item | Status |
|---|---|
| Recall@2/5/10, AllGold@k, RR/MRR, nDCG@k | done |
| Per-stage + end-to-end p50/p95 latency (5 repeats, warm-up, pinned threads) | done |
| Index build time and peak RSS | done |
| Paired bootstrap (10k) + Holm | done |
| Oracle + random router bound | done (report fix 2026-10-24, tested) |
| Gate 1 (200 → 1,000 questions) | done: **inconclusive** (+2.4 [+1.6, +3.3]); conditional continue |
| Baseline + graph tuning on train-derived set | in progress (`scripts/tune.py`) |
| Ablations (synonyms, specificity, DATE, titles, n-gram linker, label keys) | pending |
| 2WikiMultiHopQA | pending (official zip is 246 MB on Dropbox) |
| Error analysis | pending (first notes in `docs/day3_decision.md` §8) |

## Known issues

- PPR hits the 50-iteration cap at α = 0.85. Immaterial to rankings (ADR 0001 §6).
- Evaluation peak RSS is 3.9 GB at 11.5k passages vs 1.1 GB at 3.2k; cause not yet profiled.
- graph_ce reranks are ~50–90 ms slower than hybrid_ce reranks at the same depth. The likely cause is longer passages, not yet verified.
