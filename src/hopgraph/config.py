"""Experiment configuration: YAML -> validated, nested dataclasses.

Unknown keys anywhere in the YAML raise ``ConfigError`` so a typo can never
silently fall back to a default.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


@dataclass
class DataConfig:
    dataset: str = "musique"  # musique | twowiki | fixture
    split: str = "dev"
    raw_path: str = "data/raw/musique/musique_ans_v1.0_dev.jsonl"
    n_questions: int | None = 200  # None = all
    sample_seed: int = 0
    max_skip_frac: float = 0.01


@dataclass
class PathsConfig:
    processed_dir: str = "data/processed/musique_200"
    index_dir: str = "indexes/musique_200"
    results_dir: str = "results"


@dataclass
class BM25Config:
    k1: float = 1.5
    b: float = 0.75
    method: str = "lucene"
    stopwords: str | None = "en"
    stemmer: str | None = "english"  # PyStemmer algorithm name, or None
    top_k: int = 100


@dataclass
class DenseConfig:
    model: str = "BAAI/bge-small-en-v1.5"
    revision: str | None = None
    # BGE v1.5 model card recommends this instruction for short-query -> passage retrieval.
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    batch_size: int = 64
    max_seq_length: int = 512
    top_k: int = 100


@dataclass
class FusionConfig:
    k: int = 60
    weights: list[float] = field(default_factory=lambda: [1.0, 1.0])  # [bm25, dense]
    depth: int = 100


@dataclass
class RerankConfig:
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    revision: str | None = None
    depth: int = 20
    batch_size: int = 32
    max_length: int = 512


@dataclass
class GraphConfig:
    spacy_model: str = "en_core_web_sm"
    labels: list[str] = field(
        default_factory=lambda: [
            "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT",
            "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "DATE",
        ]
    )
    include_titles: bool = True  # passage title as an entity mention
    entity_key: str = "name"  # name | name_label
    mention_weight: str = "binary"  # binary | count
    synonym_tau: float | None = None  # None disables synonym edges
    synonym_top_k: int = 10
    batch_size: int = 128
    n_process: int = 1


@dataclass
class PPRConfig:
    alpha: float = 0.85  # probability of following an edge (continue), 1-alpha = restart
    tol: float = 1e-6
    max_iter: int = 50


@dataclass
class LinkingConfig:
    use_ner: bool = True
    use_ngram: bool = True
    max_ngram: int = 6
    embed_tau: float | None = 0.85  # None disables embedding NN fallback
    use_noun_chunks: bool = True


@dataclass
class SeedingConfig:
    use_specificity: bool = True
    passage_seed_m: int = 5  # top-m hybrid passages used as seeds
    passage_seed_beta: float = 0.5  # share of seed mass on passages when entities exist
    always_passage_seed: bool = True  # False = passage seeds only as a fallback


@dataclass
class GraphFusionConfig:
    k: int = 60
    weights: list[float] = field(default_factory=lambda: [1.0, 1.0])  # [hybrid, graph]
    depth: int = 100
    passage_score: str = "direct"  # direct | entity_sum


@dataclass
class EvalConfig:
    ks: list[int] = field(default_factory=lambda: [2, 5, 10])
    systems: list[str] = field(
        default_factory=lambda: ["bm25", "dense", "hybrid", "hybrid_ce", "graph", "graph_ce"]
    )
    bootstrap_n: int = 10_000
    bootstrap_seed: int = 0
    latency_n: int = 200
    latency_repeats: int = 5
    warmup: int = 10
    threads: int = 8
    rss_limit_gb: float = 12.0


@dataclass
class Config:
    name: str = "musique_200"
    seed: int = 0
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    bm25: BM25Config = field(default_factory=BM25Config)
    dense: DenseConfig = field(default_factory=DenseConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    ppr: PPRConfig = field(default_factory=PPRConfig)
    linking: LinkingConfig = field(default_factory=LinkingConfig)
    seeding: SeedingConfig = field(default_factory=SeedingConfig)
    graph_fusion: GraphFusionConfig = field(default_factory=GraphFusionConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def _build(cls: type, raw: Any, where: str) -> Any:
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(raw).__name__}")
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(known)}")
    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        hint = hints[key]
        if isinstance(hint, type) and dataclasses.is_dataclass(hint):
            kwargs[key] = _build(hint, value, f"{where}.{key}")
        else:
            kwargs[key] = _check_type(value, hint, f"{where}.{key}")
    return cls(**kwargs)


def _check_type(value: Any, hint: Any, where: str) -> Any:
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _check_type(value, arg, where)
            except ConfigError:
                continue
        raise ConfigError(f"{where}: {value!r} does not match {hint}")
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{where}: expected a list, got {value!r}")
        (inner,) = args
        return [_check_type(v, inner, f"{where}[]") for v in value]
    if hint is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if hint in (int, float, str, bool):
        if isinstance(value, bool) and hint is not bool:
            raise ConfigError(f"{where}: expected {hint.__name__}, got bool")
        if not isinstance(value, hint):
            raise ConfigError(f"{where}: expected {hint.__name__}, got {value!r}")
        return value
    return value


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load YAML into a validated ``Config``. ``overrides`` uses dotted keys."""
    # utf-8-sig: tolerate the BOM that Windows editors / PowerShell 5 often write.
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig")) or {}
    for dotted, value in (overrides or {}).items():
        node = raw
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    cfg: Config = _build(Config, raw, "config")
    validate(cfg)
    return cfg


def config_from_dict(raw: dict[str, Any]) -> Config:
    cfg: Config = _build(Config, raw, "config")
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    if not 0.0 < cfg.ppr.alpha < 1.0:
        raise ConfigError("ppr.alpha must be in (0, 1)")
    if len(cfg.fusion.weights) != 2 or len(cfg.graph_fusion.weights) != 2:
        raise ConfigError("fusion weights must have exactly two entries")
    if cfg.graph.entity_key not in ("name", "name_label"):
        raise ConfigError("graph.entity_key must be 'name' or 'name_label'")
    if cfg.graph.mention_weight not in ("binary", "count"):
        raise ConfigError("graph.mention_weight must be 'binary' or 'count'")
    if cfg.graph_fusion.passage_score not in ("direct", "entity_sum"):
        raise ConfigError("graph_fusion.passage_score must be 'direct' or 'entity_sum'")
    if not 0.0 <= cfg.seeding.passage_seed_beta <= 1.0:
        raise ConfigError("seeding.passage_seed_beta must be in [0, 1]")
    allowed = {"bm25", "dense", "hybrid", "hybrid_ce", "graph", "graph_only", "graph_ce"}
    bad = set(cfg.eval.systems) - allowed
    if bad:
        raise ConfigError(f"eval.systems: unknown system(s) {sorted(bad)}")


def save_config(cfg: Config, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
