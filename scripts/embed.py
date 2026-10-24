"""Encode passages with BGE-small, cache float32 embeddings and persist a FAISS IndexFlatIP.

    python scripts/embed.py --config configs/musique_200.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hopgraph.artifacts import dense_dir  # noqa: E402
from hopgraph.config import load_config  # noqa: E402
from hopgraph.evaluation.memory import RSSMonitor  # noqa: E402
from hopgraph.evaluation.timing import pin_threads  # noqa: E402
from hopgraph.ingestion.corpus import load_corpus  # noqa: E402
from hopgraph.runinfo import setup_logging  # noqa: E402

log = logging.getLogger("embed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true", help="re-encode even if cached")
    args = ap.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.name)
    threads = pin_threads(cfg.eval.threads)

    out = dense_dir(cfg)
    if (out / "faiss.index").exists() and not args.force:
        log.info("dense index already cached at %s (use --force to rebuild)", out)
        return

    from hopgraph.retrieval.dense import DenseRetriever, STEncoder

    passages, _ = load_corpus(cfg.paths.processed_dir)
    with RSSMonitor(limit_gb=cfg.eval.rss_limit_gb) as mon:
        t0 = time.perf_counter()
        enc = STEncoder(cfg.dense)
        load_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        emb = enc.encode_passages([p.content for p in passages])
        encode_s = time.perf_counter() - t1
        t2 = time.perf_counter()
        r = DenseRetriever(enc, cfg.dense.top_k)
        r.build(passages, emb)
        faiss_s = time.perf_counter() - t2
        meta = {
            "model": cfg.dense.model, "revision": cfg.dense.revision,
            "passages": len(passages), "dim": int(emb.shape[1]), "dtype": str(emb.dtype),
            "model_load_seconds": round(load_s, 3), "encode_seconds": round(encode_s, 3),
            "passages_per_second": round(len(passages) / encode_s, 2),
            "faiss_build_seconds": round(faiss_s, 4), "threads": threads,
            "max_seq_length": cfg.dense.max_seq_length, "batch_size": cfg.dense.batch_size,
        }
        r.save(out, emb, meta)
    meta["peak_rss_mb"] = round(mon.peak_mb, 1)
    meta["embeddings_bytes"] = (out / "embeddings.npy").stat().st_size
    meta["faiss_bytes"] = (out / "faiss.index").stat().st_size
    r.save(out, emb, meta)
    log.info("dense index -> %s: %s", out, meta)


if __name__ == "__main__":
    main()
