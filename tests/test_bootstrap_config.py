import numpy as np
import pytest

from hopgraph.config import Config, ConfigError, config_from_dict, load_config
from hopgraph.evaluation.bootstrap import holm, paired_bootstrap


def test_identical_systems_zero_diff():
    a = np.random.default_rng(0).random(100)
    r = paired_bootstrap(a, a, n=2000, seed=1)
    assert r.diff == 0.0 and r.ci_low == 0.0 and r.ci_high == 0.0
    assert r.p_value == 1.0


def test_constant_shift_detected():
    rng = np.random.default_rng(0)
    b = rng.random(200)
    a = b + 0.1
    r = paired_bootstrap(a, b, n=2000, seed=0)
    assert r.diff == pytest.approx(0.1)
    assert r.ci_low == pytest.approx(0.1) and r.ci_high == pytest.approx(0.1)
    assert r.p_value < 0.01


def test_bootstrap_ci_brackets_diff_and_is_seeded():
    rng = np.random.default_rng(3)
    a, b = rng.random(150), rng.random(150)
    r1 = paired_bootstrap(a, b, n=3000, seed=7)
    r2 = paired_bootstrap(a, b, n=3000, seed=7)
    assert r1 == r2
    assert r1.ci_low <= r1.diff <= r1.ci_high


def test_bootstrap_rejects_unpaired():
    with pytest.raises(ValueError):
        paired_bootstrap([1.0, 2.0], [1.0])


def test_holm():
    # Classic example: p = [0.01, 0.04, 0.03] -> [0.03, 0.06, 0.06]
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.9, 0.8]) == pytest.approx([1.0, 1.0])


def test_unknown_config_key_raises(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("name: x\nppr:\n  alpha: 0.5\n  dampning: 3\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="dampning"):
        load_config(p)


def test_config_types_and_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("name: x\nppr:\n  alpha: 0.5\n", encoding="utf-8")
    cfg = load_config(p, {"ppr.max_iter": 7, "graph.synonym_tau": 0.9})
    assert cfg.ppr.alpha == 0.5 and cfg.ppr.max_iter == 7 and cfg.graph.synonym_tau == 0.9
    with pytest.raises(ConfigError):
        config_from_dict({"ppr": {"max_iter": "fifty"}})
    with pytest.raises(ConfigError):
        config_from_dict({"ppr": {"alpha": 1.5}})
    with pytest.raises(ConfigError):
        config_from_dict({"eval": {"systems": ["bm25", "chatbot"]}})


def test_config_hash_stable():
    assert Config().hash() == Config().hash()
    assert config_from_dict({"ppr": {"alpha": 0.5}}).hash() != Config().hash()
