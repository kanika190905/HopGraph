"""Personalized PageRank by sparse power iteration.

Orientation: ``A[u, v]`` is the weight of edge u -> v. A walker at u moves to v
with probability ``A[u, v] / d_u`` where ``d_u = sum_v A[u, v]``. The
column-stochastic transition matrix is therefore ``M = (D^{-1} A)^T``, i.e.
``M[v, u] = A[u, v] / d_u``, and

    p_{t+1} = alpha * (M p_t + (sum_{u dangling} p_t[u]) * s) + (1 - alpha) * s

with ``p_0 = s``. Dangling nodes (``d_u = 0``) send their mass back to the seed
vector, so ``sum(p) = 1`` at every step. This is the same fixed point as
``networkx.pagerank(G, alpha, personalization=s, dangling=s)``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

log = logging.getLogger(__name__)


@dataclass
class Transition:
    M: sp.csr_matrix  # column-stochastic (except dangling columns, which are all-zero)
    dangling: np.ndarray  # bool mask of nodes with zero out-weight


def transition_matrix(A: sp.spmatrix) -> Transition:
    A = sp.csr_matrix(A, dtype=np.float64)
    if A.shape[0] != A.shape[1]:
        raise ValueError("adjacency must be square")
    if A.nnz and A.data.min() < 0:
        raise ValueError("edge weights must be non-negative")
    out_w = np.asarray(A.sum(axis=1)).ravel()
    dangling = out_w == 0
    inv = np.zeros_like(out_w)
    inv[~dangling] = 1.0 / out_w[~dangling]
    P = sp.diags(inv) @ A  # row-stochastic for non-dangling rows
    return Transition(M=sp.csr_matrix(P.T), dangling=dangling)


@dataclass
class PPRResult:
    scores: np.ndarray
    n_iter: int
    converged: bool
    l1_delta: float


def ppr(
    T: Transition, seeds: np.ndarray, alpha: float = 0.85, tol: float = 1e-6, max_iter: int = 50,
) -> PPRResult:
    """Run PPR from a seed vector (any non-negative weights; normalized here)."""
    s = np.asarray(seeds, dtype=np.float64)
    n = T.M.shape[0]
    if s.shape != (n,):
        raise ValueError(f"seed vector has shape {s.shape}, expected ({n},)")
    if (s < 0).any():
        raise ValueError("seed weights must be non-negative")
    total = s.sum()
    if total <= 0:
        raise ValueError("seed vector is all zeros")
    s = s / total
    p = s.copy()
    delta = float("inf")
    for it in range(1, max_iter + 1):
        dangling_mass = p[T.dangling].sum() if T.dangling.any() else 0.0
        p_next = alpha * (T.M @ p + dangling_mass * s) + (1.0 - alpha) * s
        delta = float(np.abs(p_next - p).sum())
        p = p_next
        if delta < tol:
            return PPRResult(p, it, True, delta)
    return PPRResult(p, max_iter, False, delta)
