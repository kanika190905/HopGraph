import numpy as np
import pytest

from hopgraph.config import LinkingConfig
from hopgraph.graph.build import EntityGraph, build_graph, synonym_pairs_from_embeddings
from hopgraph.graph.entities import (
    Mention,
    ambiguity_stats,
    extract_entities,
    normalize_entity,
    title_entity,
)
from hopgraph.graph.linking import QueryLinker
from hopgraph.ingestion.musique import load_musique


@pytest.mark.parametrize("raw,norm", [
    ("The Beatles", "beatles"),
    ("Bong Joon-ho", "bong joon ho"),
    ("U.S.", "us"),
    ("Pichai's", "pichai"),
    ("  Salim–Sulaiman ", "salim sulaiman"),
    ("Ｓｅｏｕｌ", "seoul"),  # full-width -> NFKC
    ("Chak De! India", "chak de india"),
    ("Theodore Roosevelt", "theodore roosevelt"),  # only a leading "the " word is dropped
])
def test_normalize_entity(raw, norm):
    assert normalize_entity(raw) == norm


def test_title_entity_drops_disambiguator():
    assert title_entity("Green (Steve Hillage album)") == "green"
    assert title_entity("Parasite (2019 film)") == "parasite"


def _m(*names):
    return [Mention(n, "ORG", n) for n in names]


def test_csr_dimensions_and_index_consistency():
    mentions = [_m("a", "b"), _m("b", "c", "c"), []]
    g = build_graph(["p0", "p1", "p2"], mentions, mention_weight="count")
    assert (g.n_passages, g.n_entities, g.n_nodes) == (3, 3, 6)
    assert g.entity_keys == ["a", "b", "c"]
    assert g.A.shape == (6, 6) and g.PE.shape == (3, 3)
    assert g.PE[1, 2] == 2.0  # "c" mentioned twice in p1
    assert g.A[1, 3 + 2] == 2.0 and g.A[3 + 2, 1] == 2.0  # symmetric passage<->entity
    np.testing.assert_allclose(g.specificity, [1.0, 0.5, 1.0])
    T = g.transition
    assert T.dangling.tolist() == [False, False, True, False, False, False]  # p2 has no entities
    gb = build_graph(["p0", "p1", "p2"], mentions, mention_weight="binary")
    assert gb.PE[1, 2] == 1.0
    assert g.stats()["isolated_passages"] == 1


def test_synonym_edges_and_save_load(tmp_path):
    mentions = [_m("new york"), _m("new york city"), _m("paris")]
    emb = np.array([[1, 0], [0.95, 0.312], [0, 1]], dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    pairs = synonym_pairs_from_embeddings(emb, tau=0.9)
    assert [(a, b) for a, b, _ in pairs] == [(0, 1)]
    g = build_graph(["p0", "p1", "p2"], mentions, synonym_pairs=pairs)
    assert g.n_synonym_edges == 1
    assert g.A[3 + 0, 3 + 1] > 0.9 and g.A[3 + 1, 3 + 0] == g.A[3 + 0, 3 + 1]
    g.save(tmp_path)
    g2 = EntityGraph.load(tmp_path)
    assert (g2.A != g.A).nnz == 0 and (g2.PE != g.PE).nnz == 0
    assert g2.entity_keys == g.entity_keys and g2.passage_ids == g.passage_ids


def test_name_label_keys_split_entities():
    mentions = [[Mention("green", "PERSON", "Green")], [Mention("green", "TITLE", "Green")]]
    assert build_graph(["a", "b"], mentions, key_mode="name").n_entities == 1
    g = build_graph(["a", "b"], mentions, key_mode="name_label")
    assert g.n_entities == 2 and g.entities_by_name("green") == [0, 1]
    assert ambiguity_stats(mentions)["names_with_multiple_labels"] == 0  # TITLE ignored


def test_extract_entities_with_titles(tiny_path, stub_nlp):
    passages, _, _ = load_musique(tiny_path)
    mentions = extract_entities(passages, stub_nlp, ["ORG"], include_titles=True)
    assert len(mentions) == len(passages)
    seoul = next(i for i, p in enumerate(passages) if p.title == "Seoul")
    assert Mention("seoul", "TITLE", "Seoul") in mentions[seoul]
    # Numeric labels are always excluded.
    assert extract_entities(passages[:1], stub_nlp, ["CARDINAL"], include_titles=False) == [[]]


def _graph_for_linking():
    mentions = [_m("bong joon ho", "parasite"), _m("bong joon ho", "daegu"), _m("seoul")]
    return build_graph(["p0", "p1", "p2"], mentions)


def test_linking_paths(stub_nlp, hash_encoder):
    g = _graph_for_linking()
    cfg = LinkingConfig(embed_tau=None)
    linker = QueryLinker(g, cfg, nlp=stub_nlp, labels=frozenset({"ORG"}))
    r = linker.link("Who directed Parasite?")
    assert [g.entity_keys[e.entity] for e in r.entities] == ["parasite"]
    assert r.path in {"ner", "ner+ngram", "ngram"}
    # lowercase mention: NER stub misses it, n-gram dictionary match finds it.
    r = linker.link("where was bong joon ho born")
    assert [g.entity_keys[e.entity] for e in r.entities] == ["bong joon ho"]
    assert r.path == "ngram"
    # nothing to link -> empty result, caller falls back to passage seeds
    r = linker.link("what is the answer")
    assert r.entities == [] and r.path == "none"


def test_linking_embedding_fallback(stub_nlp, hash_encoder):
    g = _graph_for_linking()
    emb = hash_encoder.encode_passages(g.entity_names)
    cfg = LinkingConfig(use_ngram=False, embed_tau=0.99)
    linker = QueryLinker(g, cfg, nlp=stub_nlp, encoder=hash_encoder, entity_emb=emb,
                         labels=frozenset({"ORG"}))
    # "Daegu City" has no exact match; the hash encoder maps it close to "daegu" only if
    # the extra token is ignored, so with tau=0.99 it must NOT link...
    assert linker.link("Is Daegu City big?").entities == []
    # ...but an exact-content mention with different punctuation embeds identically.
    cfg2 = LinkingConfig(use_ngram=False, embed_tau=0.5)
    linker2 = QueryLinker(g, cfg2, nlp=stub_nlp, encoder=hash_encoder, entity_emb=emb,
                          labels=frozenset({"ORG"}))
    r = linker2.link("Tell me about Bong-Joon-Ho-Film")
    assert r.path in {"embed", "none"}
