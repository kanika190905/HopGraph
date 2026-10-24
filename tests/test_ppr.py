import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp

from hopgraph.graph.ppr import ppr, transition_matrix


def _random_graph(rng: np.random.Generator, directed: bool):
    n = int(rng.integers(5, 30))
    p = float(rng.uniform(0.05, 0.35))
    G = nx.gnp_random_graph(n, p, seed=int(rng.integers(1 << 30)), directed=directed)
    for u, v in G.edges():
        G[u][v]["weight"] = float(rng.uniform(0.1, 3.0))
    # Force at least one isolated (dangling) node and, usually, a second component.
    G.add_node(n)
    if rng.random() < 0.7:
        G.add_edge(n + 1, n + 2, weight=1.0)
        if directed:
            G.add_edge(n + 2, n + 1, weight=1.0)
    return G


def _adjacency(G) -> sp.csr_matrix:
    return nx.to_scipy_sparse_array(G, nodelist=sorted(G.nodes()), weight="weight", format="csr")


@pytest.mark.parametrize("trial", range(24))
def test_matches_networkx(trial):
    rng = np.random.default_rng(trial)
    directed = trial % 2 == 1
    G = _random_graph(rng, directed)
    nodes = sorted(G.nodes())
    seeds = rng.random(len(nodes)) * (rng.random(len(nodes)) < 0.4)
    if seeds.sum() == 0:
        seeds[0] = 1.0
    alpha = float(rng.choice([0.5, 0.85, 0.95]))
    pers = {n: float(s) for n, s in zip(nodes, seeds, strict=True)}
    ref = nx.pagerank(G, alpha=alpha, personalization=pers, dangling=pers, tol=1e-13,
                      max_iter=10_000, weight="weight")
    res = ppr(transition_matrix(_adjacency(G)), seeds, alpha=alpha, tol=1e-12, max_iter=10_000)
    assert res.converged
    np.testing.assert_allclose(res.scores, [ref[n] for n in nodes], atol=1e-9)


def test_mass_conserved_with_dangling_nodes():
    # 0 -> 1 -> 2 (2 is dangling), 3 isolated.
    A = sp.csr_matrix(np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]], float))
    T = transition_matrix(A)
    assert T.dangling.tolist() == [False, False, True, True]
    res = ppr(T, np.array([1.0, 0, 0, 0]), alpha=0.85, tol=1e-12, max_iter=1000)
    assert res.scores.sum() == pytest.approx(1.0, abs=1e-12)
    assert res.scores[3] == 0.0  # unreachable from the seed, never seeded


def test_disconnected_component_gets_no_mass():
    A = sp.csr_matrix(np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], float))
    res = ppr(transition_matrix(A), np.array([1.0, 0, 0, 0]), alpha=0.85, tol=1e-12, max_iter=1000)
    assert res.scores[2] == 0.0 and res.scores[3] == 0.0
    assert res.scores[:2].sum() == pytest.approx(1.0)


def test_column_stochastic_orientation():
    A = sp.csr_matrix(np.array([[0, 2, 1], [1, 0, 0], [0, 1, 0]], float))
    M = transition_matrix(A).M.toarray()
    np.testing.assert_allclose(M.sum(axis=0), [1, 1, 1])
    assert M[1, 0] == pytest.approx(2 / 3)  # walker at 0 moves to 1 with prob 2/3


def test_iteration_cap_and_flags():
    A = sp.csr_matrix(np.array([[0, 1], [1, 0]], float))
    res = ppr(transition_matrix(A), np.array([1.0, 0.0]), alpha=0.99, tol=1e-15, max_iter=3)
    assert not res.converged and res.n_iter == 3


def test_seed_validation():
    T = transition_matrix(sp.csr_matrix(np.eye(2)))
    with pytest.raises(ValueError):
        ppr(T, np.zeros(2))
    with pytest.raises(ValueError):
        ppr(T, np.array([1.0, -1.0]))
    with pytest.raises(ValueError):
        ppr(T, np.ones(3))


def test_deterministic():
    rng = np.random.default_rng(0)
    A = sp.random(50, 50, density=0.1, random_state=1, format="csr")
    s = rng.random(50)
    T = transition_matrix(A)
    a = ppr(T, s).scores
    b = ppr(T, s).scores
    assert np.array_equal(a, b)
