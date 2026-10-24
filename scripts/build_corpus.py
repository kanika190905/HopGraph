"""Build passages.jsonl / questions.jsonl / manifest.json from a raw benchmark file.

    python scripts/build_corpus.py --config configs/musique_200.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hopgraph.config import load_config  # noqa: E402
from hopgraph.ingestion.corpus import corpus_hash, file_sha256, save_corpus  # noqa: E402
from hopgraph.ingestion.musique import load_musique  # noqa: E402
from hopgraph.ingestion.twowiki import load_twowiki  # noqa: E402

SOURCES = {
    "musique": "MuSiQue v1.0 (musique_ans), HF mirror dgslibisey/MuSiQue@c8f4f8c9",
    "twowiki": "2WikiMultiHopQA official data.zip (Dropbox link in Alab-NII/2wikimultihop README), "
               "zip sha256 e8e57c0aafc4a26d41131e320ebb5afb6f2aca86b8a6e6611b08f52033cb7d04",
}
from hopgraph.runinfo import setup_logging  # noqa: E402

log = logging.getLogger("build_corpus")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.name)

    raw = Path(cfg.data.raw_path)
    if not raw.exists():
        sys.exit(f"raw data not found: {raw}. See docs/data.md for download instructions.")
    loaders = {"musique": load_musique, "twowiki": load_twowiki}
    if cfg.data.dataset not in loaders:
        sys.exit(f"dataset {cfg.data.dataset!r} is not supported; choose from {sorted(loaders)}")

    t0 = time.perf_counter()
    passages, questions, stats = loaders[cfg.data.dataset](
        raw, n=cfg.data.n_questions, seed=cfg.data.sample_seed, split=cfg.data.split,
        max_skip_frac=cfg.data.max_skip_frac,
    )
    out = Path(cfg.paths.processed_dir)
    save_corpus(out, passages, questions)
    elapsed = time.perf_counter() - t0

    manifest = {
        "dataset": cfg.data.dataset,
        "split": cfg.data.split,
        "source_file": raw.name,
        "source_sha256": file_sha256(raw),
        "source_version": SOURCES[cfg.data.dataset],
        "preprocessing": {
            "n_questions": cfg.data.n_questions, "sample_seed": cfg.data.sample_seed,
            "dedupe": "sha1(normalize(title) + '\\n' + normalize(text)); normalize = NFKC, collapse whitespace, lowercase",
            "max_skip_frac": cfg.data.max_skip_frac,
        },
        "counts": {
            "records": stats.n_records, "skipped": stats.n_skipped,
            "skip_reasons": stats.skip_reasons, "questions": stats.n_selected,
            "candidates": stats.n_candidates, "passages": stats.n_passages,
            "duplicates_merged": stats.n_duplicates,
            "gold_passages": sum(len(q.gold_ids) for q in questions),
            "hops": dict(sorted(Counter(q.hops for q in questions).items())),
            "hop_types": dict(sorted(Counter(q.meta["hop_type"] for q in questions).items())),
        },
        "corpus_hash": corpus_hash(out),
        "build_seconds": round(elapsed, 3),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("wrote %s (%d passages, %d questions, hash %s)", out, len(passages), len(questions),
             manifest["corpus_hash"])


if __name__ == "__main__":
    main()
