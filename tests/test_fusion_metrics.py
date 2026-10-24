import math

import pytest

from hopgraph.evaluation.metrics import (
    all_gold_at_k,
    evaluate_ranking,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from hopgraph.retrieval.fusion import fill_from, rrf
from hopgraph.types import RankedList


def rl(*ids):
    return RankedList([(pid, float(len(ids) - i)) for i, pid in enumerate(ids)])


def test_rrf_hand_computed():
    a = rl("x", "y", "z")
    b = rl("y", "w")
    fused = rrf([a, b], k=60)
    s = dict(fused.items)
    assert s["y"] == pytest.approx(1 / 62 + 1 / 61)
    assert s["x"] == pytest.approx(1 / 61)
    assert s["w"] == pytest.approx(1 / 62)
    assert s["z"] == pytest.approx(1 / 63)
    assert fused.ids() == ["y", "x", "w", "z"]  # w and ... ordering by score


def test_rrf_weights_and_depth():
    a = rl("x", "y", "z")
    b = rl("z", "y", "x")
    fused = rrf([a, b], k=0, weights=[2.0, 1.0])
    s = dict(fused.items)
    assert s["x"] == pytest.approx(2 / 1 + 1 / 3)
    assert s["z"] == pytest.approx(2 / 3 + 1 / 1)
    assert rrf([a, b], k=60, depth=1).ids() == ["x", "z"]  # tie broken by id
    with pytest.raises(ValueError):
        rrf([a, b], weights=[1.0])


def test_rrf_tie_break_is_deterministic():
    assert rrf([rl("b"), rl("a")]).ids() == ["a", "b"]


def test_fill_from():
    out = fill_from(rl("a", "b"), rl("b", "c", "d"), 3)
    assert out.ids() == ["a", "b", "c"]
    assert out.scores() == sorted(out.scores(), reverse=True)
    assert fill_from(rl("a", "b", "c"), rl("d"), 2).ids() == ["a", "b"]
    assert fill_from(RankedList([]), rl("x", "y"), 5).ids() == ["x", "y"]


def test_metrics_hand_labelled():
    ranked = ["p1", "g1", "p2", "g2", "p3"]
    gold = ["g1", "g2", "g3"]
    assert recall_at_k(ranked, gold, 2) == pytest.approx(1 / 3)
    assert recall_at_k(ranked, gold, 5) == pytest.approx(2 / 3)
    assert all_gold_at_k(ranked, gold, 5) == 0.0
    assert all_gold_at_k(["g1", "g2", "g3"], gold, 3) == 1.0
    assert reciprocal_rank(ranked, gold) == pytest.approx(0.5)
    assert reciprocal_rank(["a", "b"], gold) == 0.0
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(ranked, gold, 5) == pytest.approx(dcg / idcg)
    m = evaluate_ranking(ranked, gold, [2, 5])
    assert set(m) == {"recall@2", "allgold@2", "ndcg@2", "recall@5", "allgold@5", "ndcg@5", "rr"}


def test_metrics_short_list_and_duplicates_in_gold():
    assert recall_at_k(["g1"], ["g1", "g1"], 10) == 1.0
    with pytest.raises(ValueError):
        recall_at_k(["a"], [], 1)


def test_ranked_list_from_array_ties():
    import numpy as np

    scores = np.array([0.0, 2.0, 1.0, 2.0, 1.0])
    out = RankedList.from_array(scores, ["e", "d", "c", "b", "a"], 3)
    assert out.ids() == ["b", "d", "a"]
