"""Extract entities with spaCy, build the entity–passage CSR graph and persist it.

    python scripts/build_graph.py --config configs/musique_200.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hopgraph.artifacts import entity_emb_path, graph_dir, mentions_path  # noqa: E402
from hopgraph.config import load_config  # noqa: E402
from hopgraph.evaluation.memory import RSSMonitor  # noqa: E402
from hopgraph.evaluation.timing import pin_threads  # noqa: E402
from hopgraph.graph.build import (  # noqa: E402
    build_graph,
    collect_entities,
    synonym_pairs_from_embeddings,
)
from hopgraph.graph.entities import (  # noqa: E402
    ambiguity_stats,
    extract_entities,
    load_mentions,
    load_ner,
    save_mentions,
)
from hopgraph.ingestion.corpus import load_corpus  # noqa: E402
from hopgraph.runinfo import setup_logging  # noqa: E402

log = logging.getLogger("build_graph")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.name)
    threads = pin_threads(cfg.eval.threads)
    gcfg = cfg.graph
    passages, _ = load_corpus(cfg.paths.processed_dir)
    pids = [p.id for p in passages]
    timings: dict[str, float] = {}

    with RSSMonitor(limit_gb=cfg.eval.rss_limit_gb) as mon:
        mpath = mentions_path(cfg)
        if mpath.exists():
            cached_ids, mentions = load_mentions(mpath)
            if cached_ids != pids:
                raise RuntimeError(f"{mpath} was built for a different corpus; delete it")
            log.info("loaded cached mentions from %s", mpath)
            timings["ner_seconds_cached"] = json.loads(
                (mpath.with_suffix(".meta.json")).read_text(encoding="utf-8"))["ner_seconds"]
        else:
            t0 = time.perf_counter()
            nlp = load_ner(gcfg.spacy_model)
            timings["spacy_load_seconds"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            mentions = extract_entities(passages, nlp, gcfg.labels, gcfg.batch_size,
                                        gcfg.n_process, gcfg.include_titles)
            timings["ner_seconds"] = time.perf_counter() - t0
            save_mentions(mpath, pids, mentions)
            mpath.with_suffix(".meta.json").write_text(
                json.dumps({"ner_seconds": timings["ner_seconds"], "passages": len(pids)}),
                encoding="utf-8")
            log.info("NER over %d passages: %.1fs", len(pids), timings["ner_seconds"])

        keys, names, _, _ = collect_entities(mentions, gcfg.entity_key)

        # Entity-name embeddings: for synonym edges and fuzzy query linking.
        emb = None
        epath = entity_emb_path(cfg)
        need_emb = gcfg.synonym_tau is not None or cfg.linking.embed_tau is not None
        if need_emb:
            if epath.exists():
                emb = np.load(epath)
                if emb.shape[0] != len(keys):
                    raise RuntimeError(f"{epath} does not match the entity set; delete it")
            else:
                from hopgraph.retrieval.dense import STEncoder

                enc = STEncoder(cfg.dense)
                t0 = time.perf_counter()
                emb = enc.encode_passages(names)
                timings["entity_embed_seconds"] = time.perf_counter() - t0
                np.save(epath, emb)
                log.info("embedded %d entity names: %.1fs", len(names),
                         timings["entity_embed_seconds"])

        pairs = None
        if gcfg.synonym_tau is not None:
            t0 = time.perf_counter()
            pairs = synonym_pairs_from_embeddings(emb, gcfg.synonym_tau, gcfg.synonym_top_k)
            timings["synonym_seconds"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        g = build_graph(pids, mentions, gcfg.entity_key, gcfg.mention_weight, pairs)
        timings["graph_build_seconds"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        _ = g.transition
        timings["transition_seconds"] = time.perf_counter() - t0
        out = graph_dir(cfg)
        sizes = g.save(out)

    stats = g.stats()
    stats.update(ambiguity_stats(mentions))
    meta = {
        "graph_config": cfg.to_dict()["graph"],
        "stats": stats,
        "timings_seconds": {k: round(v, 4) for k, v in timings.items()},
        "file_bytes": sizes,
        "total_bytes": sum(sizes.values()),
        "entity_embeddings": str(epath) if emb is not None else None,
        "peak_rss_mb": round(mon.peak_mb, 1),
        "threads": threads,
    }
    (out / "build_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("graph -> %s\n%s", out, json.dumps(meta["stats"], indent=2))
    log.info("timings %s, bytes %d, peak RSS %.0f MB", meta["timings_seconds"],
             meta["total_bytes"], mon.peak_mb)


if __name__ == "__main__":
    main()
