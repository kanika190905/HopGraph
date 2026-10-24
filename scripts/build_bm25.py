"""Build and persist the BM25 index.

    python scripts/build_bm25.py --config configs/musique_200.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hopgraph.artifacts import bm25_dir  # noqa: E402
from hopgraph.config import load_config  # noqa: E402
from hopgraph.evaluation.memory import RSSMonitor  # noqa: E402
from hopgraph.ingestion.corpus import load_corpus  # noqa: E402
from hopgraph.retrieval.bm25 import BM25Retriever  # noqa: E402
from hopgraph.runinfo import setup_logging  # noqa: E402

log = logging.getLogger("build_bm25")


def dir_size(d: Path) -> int:
    return sum(p.stat().st_size for p in d.rglob("*") if p.is_file())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.name)
    passages, _ = load_corpus(cfg.paths.processed_dir)
    out = bm25_dir(cfg)
    with RSSMonitor(limit_gb=cfg.eval.rss_limit_gb) as mon:
        t0 = time.perf_counter()
        r = BM25Retriever(cfg.bm25)
        r.index(passages)
        index_s = time.perf_counter() - t0
        r.save(out)
    meta = {"passages": len(passages), "index_seconds": round(index_s, 3),
            "peak_rss_mb": round(mon.peak_mb, 1), "disk_bytes": dir_size(out),
            "params": {"k1": cfg.bm25.k1, "b": cfg.bm25.b, "method": cfg.bm25.method,
                       "stopwords": cfg.bm25.stopwords, "stemmer": cfg.bm25.stemmer}}
    (out / "build_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("BM25 index -> %s: %s", out, meta)


if __name__ == "__main__":
    main()
