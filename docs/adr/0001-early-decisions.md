# ADR 0001 — Decisions made during the Day 1–3 spike

Status: accepted, 2026-10-24. Each decision records the evidence behind it.

## 1. Python 3.12 instead of 3.11

The development machine has Python 3.12.10 and 3.13.5, but not 3.11. torch 2.14.1+cpu,
faiss-cpu 1.15.1, spaCy 3.8.16 and bm25s 0.3.15 all installed natively from wheels on
Windows 3.12, and the full test suite passes. Installing 3.11 alongside would add setup
steps without a measured benefit. `pyproject.toml` declares `>=3.11`.

## 2. MuSiQue from a Hugging Face mirror, hash-pinned

The official release is a Google Drive zip fetched with `gdown`. The HF mirrors
`dgslibisey/MuSiQue` and `bdsaglam/musique` serve files with the official names whose
SHA-256 digests match each other. `scripts/download_data.py` pins the revision and verifies
the hash. Limitation: I have not compared the hash against the Google Drive original itself.

## 3. Passage titles as entity mentions

spaCy `en_core_web_sm` failed to tag "Bong Joon-ho" as PERSON in a smoke test, and it tagged
the album "Green" as PERSON. MuSiQue paragraphs carry Wikipedia titles that name the
paragraph's subject, which makes them the most reliable entity in each passage. The
parenthetical disambiguator is dropped ("Green (Steve Hillage album)" → "green"). Titles are
part of the text every system indexes, so no system gets extra information.
`graph.include_titles` toggles this for ablation.

## 4. Dictionary n-gram linking alongside NER exact match

MuSiQue questions often name entities in ways NER misses: lower-case words, album titles,
or "the Green performer". Matching query n-grams (longest first, non-overlapping,
stopword-only grams skipped) against the entity-name dictionary needs no model. It is a
standard gazetteer technique, not a contribution. `linking.use_ngram` toggles it.
Known side effect: generic title words ("state", "producer") also link. Specificity
weighting dampens them.

## 5. Entities keyed by normalized name (not name + label)

Measured on the 200-question corpus: only 3.1% of NER names carry more than one coarse label,
while label noise from `en_core_web_sm` is visible (an album tagged PERSON). Splitting nodes
by label would break the title ↔ NER-mention connection that bridging depends on.
`graph.entity_key: name_label` remains as an ablation.

## 6. PPR iteration cap kept at 50 despite non-convergence at α = 0.85

All queries hit the cap (L1 change ≈ 4 × 10⁻⁵ at stop), but the fully converged rankings have
identical top-10 sets for 200/200 queries (`results/published/ppr_convergence_musique_200.json`).
The cap does not affect results, so the pre-registered setting stands. HippoRAG uses α = 0.5,
which converges in about 20 iterations, and that value is in the tuning grid.

## 7. Dangling-node mass returns to the seed vector

The spec's update $p_{t+1}=\alpha Mp_t+(1-\alpha)s$ loses probability mass at dangling nodes
(passages with no entities). The implementation adds $\alpha(\sum_{\text{dangling}}p_t)s$, which
keeps $\sum p=1$ and matches `networkx.pagerank(..., dangling=s)` (24 randomized tests).
