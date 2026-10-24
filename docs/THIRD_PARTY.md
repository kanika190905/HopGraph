# Third-party licenses and attribution

I verified each license below on 2026-10-24 through the GitHub license API, the Hugging Face
model/dataset API, or the repository's own LICENSE file, as noted. HopGraph redistributes
**none** of the datasets or model weights. Scripts download them from the original sources.

## Datasets

| Dataset | Source | License | How verified | Notes |
|---|---|---|---|---|
| MuSiQue v1.0 | [StonyBrookNLP/musique](https://github.com/StonyBrookNLP/musique) | CC BY 4.0 | README statement + `LICENSE` file (CC BY 4.0 text) + GitHub API `CC-BY-4.0` | Downloaded from HF mirror `dgslibisey/MuSiQue@c8f4f8c9`; SHA-256 matches the independent mirror `bdsaglam/musique`. Cite Trivedi et al., TACL 2022. |
| 2WikiMultiHopQA | [Alab-NII/2wikimultihop](https://github.com/Alab-NII/2wikimultihop) | Apache-2.0 (repository) | GitHub API | The README names no separate data license. Not downloaded yet. |
| HotpotQA | [hotpotqa/hotpot](https://github.com/hotpotqa/hotpot) | Apache-2.0 (repository) | GitHub API | The dataset is widely cited as CC BY-SA 4.0, which I have **not verified**. Optional, not used yet. |
| HippoRAG released samples | [osunlp/HippoRAG_v2](https://huggingface.co/datasets/osunlp/HippoRAG_v2) | none declared | HF dataset card | Not used in the pipeline; would serve only as a cross-reference. |

## Models

| Model | License | How verified | Pinned revision |
|---|---|---|---|
| [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | MIT | HF API `cardData.license` | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` |
| [cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) | Apache-2.0 | HF API `cardData.license` | `233902d25c440f23af6f7d6e94d2946bac0bee0a` |
| [spaCy en_core_web_sm 3.8.0](https://github.com/explosion/spacy-models/releases/tag/en_core_web_sm-3.8.0) | MIT | HF API (`spacy/en_core_web_sm`) | wheel SHA-256 in `requirements.lock.txt` |

## Libraries (main dependencies)

| Library | License | How verified |
|---|---|---|
| bm25s | MIT | GitHub API (xhluca/bm25s) |
| faiss-cpu | MIT | GitHub API (facebookresearch/faiss) |
| spaCy | MIT | GitHub API (explosion/spaCy) |
| sentence-transformers | Apache-2.0 | GitHub API (UKPLab/sentence-transformers) |
| PyTorch, NumPy, SciPy, NetworkX, pandas, scikit-learn, matplotlib, PyYAML, psutil, PyStemmer | permissive (BSD/MIT-style) | from general knowledge; **not individually re-verified** this session |

## Prior work the design follows

HopGraph's graph design follows **HippoRAG** (Gutiérrez et al., NeurIPS 2024) and **HippoRAG 2**
(Gutiérrez et al., ICML 2025): PPR over a phrase/passage graph, node specificity, synonym
edges, and passage-node seeding. HopGraph replaces their LLM extraction with spaCy NER.
RRF comes from Cormack, Clarke and Büttcher (SIGIR 2009). See `docs/novelty_review.md`.
