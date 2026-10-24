"""End-to-end retrieval on the hand-labelled tiny corpus, with stub models (no downloads)."""

import numpy as np
import pytest

from hopgraph.config import Config
from hopgraph.evaluation.metrics import evaluate_ranking
from hopgraph.graph.build import EntityGraph, build_graph
from hopgraph.graph.entities import extract_entities
from hopgraph.graph.linking import QueryLinker
from hopgraph.graph.retriever import GraphRetriever
from hopgraph.ingestion.musique import load_musique
from hopgraph.pipeline import Pipeline, system_latency_ms
from hopgraph.retrieval.bm25 import BM25Retriever
from hopgraph.retrieval.dense import DenseRetriever
from hopgraph.types import RankedList
from tests.conftest import StubReranker

SYSTEMS = ["bm25", "dense", "hybrid", "hybrid_ce", "graph", "graph_only", "graph_ce"]


@pytest.fixture
def setup(tiny_path, hash_encoder, stub_nlp, tmp_path):
    cfg = Config()
    cfg.bm25.top_k = cfg.dense.top_k = cfg.fusion.depth = cfg.graph_fusion.depth = 10
    cfg.rerank.depth = 5
    cfg.linking.embed_tau = None
    passages, questions, _ = load_musique(tiny_path)
    bm25 = BM25Retriever(cfg.bm25)
    bm25.index(passages)
    dense = DenseRetriever(hash_encoder, cfg.dense.top_k)
    dense.build(passages)
    mentions = extract_entities(passages, stub_nlp, ["ORG"], include_titles=True)
    g = build_graph([p.id for p in passages], mentions)
    linker = QueryLinker(g, cfg.linking, nlp=stub_nlp, labels=frozenset({"ORG"}))
    gr = GraphRetriever(g, linker, cfg.seeding, cfg.ppr, cfg.graph_fusion)
    pipe = Pipeline(cfg, {p.id: p for p in passages}, bm25, dense, StubReranker(), gr)
    return cfg, passages, questions, pipe, bm25, dense, g


def test_all_systems_return_full_valid_rankings(setup):
    cfg, passages, questions, pipe, *_ = setup
    ids = {p.id for p in passages}
    for q in questions:
        out = pipe.run(q.text, SYSTEMS)
        for s in SYSTEMS:
            ranked = out.rankings[s].ids()
            assert len(ranked) == len(set(ranked)), s
            assert set(ranked) <= ids
        for s in ("hybrid", "graph", "graph_only", "graph_ce", "hybrid_ce"):
            assert len(out.rankings[s]) == min(10, len(passages)), s
        assert out.info["seed_kind"] in {"entities+passages", "passages", "entities"}
        assert system_latency_ms(out.stage_ms, "graph") >= system_latency_ms(out.stage_ms, "hybrid")


def test_graph_reaches_bridge_passage(setup):
    """The second-hop passage ("Bong Joon-ho") shares no words with the question's
    country/capital wording but is one entity hop from "Parasite"."""
    cfg, passages, questions, pipe, *_ = setup
    q = next(q for q in questions if q.id == "2hop__1_2")
    out = pipe.run(q.text, ["graph_only"])
    m = evaluate_ranking(out.rankings["graph_only"].ids(), q.gold_ids, [5])
    assert m["recall@5"] == 1.0


def test_deterministic_rankings(setup):
    cfg, passages, questions, pipe, *_ = setup
    a = [pipe.run(q.text, SYSTEMS).rankings for q in questions]
    b = [pipe.run(q.text, SYSTEMS).rankings for q in questions]
    for ra, rb in zip(a, b, strict=True):
        for s in SYSTEMS:
            assert ra[s].items == rb[s].items


def test_index_save_load_equivalence(setup, tmp_path, hash_encoder):
    cfg, passages, questions, pipe, bm25, dense, g = setup
    bm25.save(tmp_path / "bm25")
    emb = dense.build(passages)
    dense.save(tmp_path / "dense", emb, {"note": "test"})
    g.save(tmp_path / "graph")
    bm25b = BM25Retriever.load(cfg.bm25, tmp_path / "bm25")
    denseb = DenseRetriever.load(tmp_path / "dense", hash_encoder, cfg.dense.top_k)
    gb = EntityGraph.load(tmp_path / "graph")
    np.testing.assert_array_equal(DenseRetriever.load_embeddings(tmp_path / "dense"), emb)
    for q in questions:
        assert bm25.search(q.text, 10).items == bm25b.search(q.text, 10).items
        assert dense.search(q.text, 10).items == denseb.search(q.text, 10).items
    assert (gb.A != g.A).nnz == 0
    cfg.bm25.k1 = 0.9
    with pytest.raises(ValueError, match="rebuild"):
        BM25Retriever.load(cfg.bm25, tmp_path / "bm25")


def test_no_entity_query_falls_back_to_hybrid(setup):
    cfg, passages, questions, pipe, *_ = setup
    cfg.seeding.passage_seed_m = 0  # no passage seeds and no entities -> empty seed
    out = pipe.run("zzz qqq", ["hybrid", "graph"])
    assert out.info["seed_kind"] == "empty"
    assert out.rankings["graph"].items == out.rankings["hybrid"].items


def test_seed_vector_normalized(setup):
    cfg, passages, questions, pipe, *_ = setup
    gr = pipe.graph
    link = gr.linker.link(questions[0].text)
    hybrid = RankedList([(passages[0].id, 0.03), (passages[1].id, 0.01)])
    s, kind = gr.seed_vector(link, hybrid)
    assert kind == "entities+passages"
    assert s.sum() == pytest.approx(1.0)
    assert s[: gr.g.n_passages].sum() == pytest.approx(cfg.seeding.passage_seed_beta)
