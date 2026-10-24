"""Where cached artefacts live. Directory names embed a hash of the settings that
produced them, so changing a parameter can never silently reuse a stale index."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from hopgraph.config import Config


def _h(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:8]


def bm25_dir(cfg: Config) -> Path:
    b = cfg.bm25
    return Path(cfg.paths.index_dir) / f"bm25_{_h([b.k1, b.b, b.method, b.stopwords, b.stemmer])}"


def dense_dir(cfg: Config) -> Path:
    d = cfg.dense
    return Path(cfg.paths.index_dir) / f"dense_{_h([d.model, d.revision, d.max_seq_length])}"


def mentions_path(cfg: Config) -> Path:
    g = cfg.graph
    key = [g.spacy_model, sorted(g.labels), g.include_titles]
    return Path(cfg.paths.index_dir) / "graph" / f"mentions_{_h(key)}.jsonl"


def entity_emb_path(cfg: Config) -> Path:
    g, d = cfg.graph, cfg.dense
    key = [g.spacy_model, sorted(g.labels), g.include_titles, g.entity_key, d.model, d.revision]
    return Path(cfg.paths.index_dir) / "graph" / f"entity_emb_{_h(key)}.npy"


def graph_dir(cfg: Config) -> Path:
    g = asdict(cfg.graph)
    g.pop("batch_size")
    g.pop("n_process")
    g["labels"] = sorted(g["labels"])
    if cfg.graph.synonym_tau is not None:
        g["dense"] = [cfg.dense.model, cfg.dense.revision]
    return Path(cfg.paths.index_dir) / "graph" / f"graph_{_h(g)}"
