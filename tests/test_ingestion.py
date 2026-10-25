import json

import pytest

from hopgraph.ingestion.corpus import (
    DataError,
    RawCandidate,
    dedupe,
    load_corpus,
    map_gold,
    passage_hash,
    save_corpus,
)
from hopgraph.ingestion.musique import load_musique


def test_tiny_corpus_dedupes_shared_paragraphs(tiny_path):
    passages, questions, stats = load_musique(tiny_path, n=None)
    assert stats.n_selected == 5 and stats.n_candidates == 20
    # "Parasite (2019 film)", "Bong Joon-ho" and "Stanford University" appear twice.
    assert stats.n_duplicates == 3 and len(passages) == 17
    by_id = {p.id: p for p in passages}
    q1 = next(q for q in questions if q.id == "2hop__1_2")
    q4 = next(q for q in questions if q.id == "2hop__8_9")
    assert {by_id[g].title for g in q1.gold_ids} == {"Parasite (2019 film)", "Bong Joon-ho"}
    assert set(q1.gold_ids) == set(q4.gold_ids)  # same canonical gold after dedupe
    q3 = next(q for q in questions if q.id.startswith("3hop"))
    assert q3.hops == 3 and len(q3.gold_ids) == 3


def test_sampling_is_seeded_and_nested(tiny_path):
    _, a, _ = load_musique(tiny_path, n=2, seed=1)
    _, b, _ = load_musique(tiny_path, n=4, seed=1)
    _, c, _ = load_musique(tiny_path, n=2, seed=1)
    assert [q.id for q in a] == [q.id for q in c]
    assert [q.id for q in a] == [q.id for q in b][:2]


def test_offset_gives_disjoint_heldout_slice(tiny_path):
    _, first, _ = load_musique(tiny_path, n=2, seed=1)
    _, rest, _ = load_musique(tiny_path, n=None, seed=1, offset=2)
    _, everything, _ = load_musique(tiny_path, n=None, seed=1)
    assert {q.id for q in first}.isdisjoint({q.id for q in rest})
    assert [q.id for q in first + rest] == [q.id for q in everything]
    with pytest.raises(DataError):
        load_musique(tiny_path, offset=5)


def test_hash_normalization():
    assert passage_hash("Seoul", "Seoul  is\nthe capital.") == passage_hash(" seoul", "Seoul is the capital.")
    assert passage_hash("Seoul", "x") != passage_hash("Busan", "x")


def test_dedupe_and_gold_mapping_fail_loudly():
    res = dedupe([RawCandidate("q::0", "A", "text", "s"), RawCandidate("q::1", "A", "text ", "s")])
    assert len(res.passages) == 1 and res.n_duplicates == 1
    assert map_gold(["q::0", "q::1"], res.raw_to_canonical, "q") == (res.passages[0].id,)
    with pytest.raises(DataError):
        map_gold(["q::9"], res.raw_to_canonical, "q")
    with pytest.raises(DataError):
        dedupe([RawCandidate("q::0", "A", "t", "s"), RawCandidate("q::0", "B", "u", "s")])


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) if isinstance(r, dict) else r for r in rows) + "\n",
                    encoding="utf-8")


def test_malformed_records_skipped_or_fail(tiny_path, tmp_path):
    good = [json.loads(line) for line in tiny_path.read_text(encoding="utf-8").splitlines()]
    bad_q = dict(good[0], id="2hop__x_y", question="  ")
    no_support = dict(good[1], id="2hop__z_w",
                      paragraphs=[dict(p, is_supporting=False) for p in good[1]["paragraphs"]])
    p = tmp_path / "m.jsonl"
    _write(p, good + [bad_q, no_support, "{not json"])
    with pytest.raises(DataError, match="max_skip_frac"):
        load_musique(p, max_skip_frac=0.01)
    passages, questions, stats = load_musique(p, max_skip_frac=0.5)
    assert stats.n_skipped == 3 and len(questions) == 5
    assert stats.skip_reasons == {"empty_question": 1, "no_supporting": 1, "bad_json": 1}


def test_empty_file_fails(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    with pytest.raises(DataError):
        load_musique(p)


def test_save_load_roundtrip(tiny_path, tmp_path):
    passages, questions, _ = load_musique(tiny_path)
    save_corpus(tmp_path, passages, questions)
    p2, q2 = load_corpus(tmp_path)
    assert p2 == passages
    assert [(q.id, q.gold_ids, q.hops) for q in q2] == [(q.id, q.gold_ids, q.hops) for q in questions]
