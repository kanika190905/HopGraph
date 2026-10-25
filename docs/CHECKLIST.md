# Implementation checklist

Status key: **done** = exists in the repo *and* has a passing test or a saved run ·
**partial** · **pending** · **blocked** (with reason).

Last updated: 2026-10-25, after the tuned dev-1,000 evaluation.

## Environment and repository

| Item | Status | Evidence / notes |
|---|---|---|
| Python env (3.12.10 venv; 3.11 not installed) | done | `requirements.lock.txt`; ADR 0001 §1 |
| CPU-only PyTorch (2.14.1+cpu) | done | `torch.cuda.is_available() == False` |
| Native Windows install (faiss-cpu, spaCy, bm25s) | done | no WSL needed |
| Git repository | done | Git is **not on PATH**; GitHub Desktop's bundled git was used. Install Git for Windows, or set `GIT_EXECUTABLE`. |
| ruff, mypy (whole package) | done | both clean |
| GitHub Actions CI (Ubuntu + Windows, no torch/models) | partial | the same dependency set passes ruff + mypy + 78 tests locally in a torch-free venv; **not yet run on GitHub** (no remote) |
| Makefile + `scripts/tasks.py` | done | `tasks.py reproduce` ran end to end |
| Fresh-clone reproduction | pending | |

## Core implementation

| Item | Status | Evidence |
|---|---|---|
| MuSiQue + 2Wiki ingestion, dedupe, fail-loud gold mapping | done | tests; manifests (0 skipped records) |
| `.txt`/`.md` chunking | done | `tests/test_documents.py` (not yet wired into a CLI) |
| BM25, dense (FAISS IndexFlatIP), RRF, cross-encoder | done | tests + runs |
| spaCy NER + title entities, CSR graph, save/load | done | tests + runs |
| Sparse PPR with dangling handling | done | 24 NetworkX comparisons; convergence check |
| Query linking with logged fallbacks | done | tests; link paths in every run |
| Graph/hybrid fusion with hybrid fill | done | end-to-end tests |
| Synonym edges | done (code) | not yet evaluated |
| Router | pending | Gate 2 early read: allowed only for the no-reranker pipeline (oracle saving 50% ≈ 33 ms), low priority |

## Evaluation

| Item | Status |
|---|---|
| Metrics, latency (p50/p95, per stage), index time, peak RSS, bootstrap + Holm, oracle + random router | done |
| Gate 1 | done: **Go** on MuSiQue after train-only tuning (+3.9 [+2.8, +5.0]) |
| Tuning on a train-derived set | done (`docs/tuning.md`) |
| 2WikiMultiHopQA (frozen config) | done (quality only): headline +5.2 [+4.3, +6.1] |
| Ablations | done: 14 single-change variants on the train tuning set (`docs/error_analysis.md` §5) |
| Error analysis | done (first pass): link noise, graph distance, degree bias (`docs/error_analysis.md`) |
| v2 (no n-gram linker + count weights) confirmed on train | done: graph +2.3 [+0.8, +3.8]; graph_ce +0.6 (rule met) |
| Held-out confirmation on the 1,417 unused MuSiQue dev questions | done, run once: headline +3.1 [+2.2, +4.1]; graph (no CE) − hybrid_ce +3.2 [+1.8, +4.5] |
| Router (Gate 2) | decision pending: allowed only for the no-reranker pipeline |
| Interview notes | done (`docs/interview_prep.md`), numbers from MuSiQue only |

## Known issues

- PPR hits the 50-iteration cap at α = 0.85. Immaterial to rankings (ADR 0001 §6).
- Peak RSS was 3.9 GB in the untuned dev-1,000 run but 1.46 GB in the tuned run on the same corpus. Not explained yet.
- Graph-fused top-20 passages are ~10% longer than hybrid's, which slows their rerank by about 20%.
