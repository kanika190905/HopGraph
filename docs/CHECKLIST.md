# Implementation checklist

Status key: **done** = exists in the repo *and* has a passing test or a saved run ·
**partial** · **pending** · **blocked** (with reason).

Last updated: 2026-10-24 (Day 1–3 spike).

## Environment and repository

| Item | Status | Evidence / notes |
|---|---|---|
| Python env (3.12.10 venv; 3.11 not installed) | done | `requirements.lock.txt`; deviation explained in README |
| CPU-only PyTorch (2.14.1+cpu) | done | `torch.cuda.is_available() == False` |
| Native Windows install of faiss-cpu / spaCy / bm25s | done | No WSL needed |
| Git repository | done | Git is **not on PATH**; GitHub Desktop's bundled git was used. Set `GIT_EXECUTABLE` for run metadata, or install Git for Windows. |
| `pyproject.toml`, ruff config | done | `ruff check` clean |
| GitHub Actions CI | pending | |
| Makefile / Windows task runner | pending | |
| mypy on stabilized modules | pending | configured in `pyproject.toml`, not yet run |

## Core implementation

| Item | Status | Evidence |
|---|---|---|
| MuSiQue ingestion, dedupe, gold mapping (fail loudly) | done | `tests/test_ingestion.py`; 200-q manifest: 3,999 candidates → 3,235 passages, 0 skipped |
| `.txt`/`.md` user-document chunking | pending | later milestone |
| BM25 (bm25s) | done | tests + run |
| Dense (BGE-small, FAISS IndexFlatIP, cached .npy) | done | save/load equivalence test |
| Hybrid RRF | done | hand-computed RRF test |
| Cross-encoder rerank (depth 20) | done | run; depths 50/100 pending |
| spaCy entity extraction + title entities | done | tests (stub NER) + run |
| CSR entity–passage graph, save/load | done | `tests/test_graph.py` |
| Synonym edges | done (code), not yet evaluated | test only; ablation pending |
| Sparse PPR with dangling handling | done | 24 randomized NetworkX comparisons |
| Query linking with logged fallbacks | done | `tests/test_graph.py::test_linking_paths` |
| Graph/hybrid fusion with hybrid fill | done | end-to-end tests |
| Router | pending (evidence-gated) | Gate 2 |

## Evaluation

| Item | Status |
|---|---|
| Recall@2/5/10, AllGold@k, RR/MRR, nDCG@k | done |
| Per-stage + end-to-end p50/p95 latency (5 repeats, warm-up, pinned threads) | done (Day-3 run) |
| Index build time and peak RSS | done (corpus, BM25, dense, graph) |
| Paired bootstrap (10k) + Holm | done |
| Oracle router bound | done (Day-3 run) |
| BM25 / RRF / CE-depth tuning on train-derived dev | pending |
| Ablations | pending |
| 2WikiMultiHopQA | pending (download not started) |
| Error analysis | pending |

## Known issues

- PPR hits the 50-iteration cap at α = 0.85 (analysis in `docs/day3_decision.md`).
