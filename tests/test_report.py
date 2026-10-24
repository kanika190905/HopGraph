import pytest

from hopgraph.evaluation.report import oracle_router, random_router


def _rec(qid, base_r5, graph_r5):
    return {"qid": qid, "hops": 2, "systems": {
        "hybrid": {"metrics": {"recall@5": base_r5}},
        "graph": {"metrics": {"recall@5": graph_r5}},
    }}


def _lat(qid, base_ms, graph_ms):
    return {"qid": qid, "system_ms": {"hybrid": base_ms, "graph": graph_ms},
            "stage_ms": {"ppr": graph_ms - base_ms}}


def test_oracle_router_hand_computed():
    records = [_rec("a", 0.5, 1.0), _rec("b", 0.5, 0.5), _rec("c", 1.0, 0.5), _rec("d", 0.0, 0.5)]
    lat = [_lat("a", 10, 30), _lat("b", 10, 30), _lat("c", 10, 30)]  # "d" not timed
    o = oracle_router(records, lat, "hybrid", "graph")
    assert o["n_quality"] == 4 and o["n_latency"] == 3
    assert o["graph_helps_frac"] == 0.5  # a and d
    assert o["graph_hurts_frac"] == 0.25 and o["tie_frac"] == 0.25
    assert o["recall@5_oracle"] == pytest.approx((1.0 + 0.5 + 1.0 + 0.5) / 4)
    # timed subset a, b, c: graph only for a -> (30 + 10 + 10) / 3
    assert o["mean_ms_oracle"] == pytest.approx(50 / 3)
    assert o["oracle_latency_saving_vs_always_graph"] == pytest.approx(1 - (50 / 3) / 30)
    assert o["graph_stage_share_of_always_graph"] == pytest.approx(20 / 30)


def test_random_router_bounds():
    records = [_rec(str(i), 0.0, 1.0) for i in range(100)]
    r = random_router(records, "hybrid", "graph", rate=0.3, n=500, seed=0)
    assert r["recall@5_mean"] == pytest.approx(0.3, abs=0.02)
    assert random_router(records, "hybrid", "graph", rate=0.0, n=10)["recall@5_mean"] == 0.0
