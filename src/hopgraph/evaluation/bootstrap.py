"""Paired bootstrap confidence intervals and Holm correction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class BootstrapResult:
    mean_a: float
    mean_b: float
    diff: float  # mean(a) - mean(b), on the original sample
    ci_low: float
    ci_high: float
    p_value: float  # two-sided bootstrap p-value for H0: diff = 0
    n: int
    n_resamples: int
    seed: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def paired_bootstrap(
    a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray, n: int = 10_000, seed: int = 0,
    alpha: float = 0.05, chunk: int = 1_000,
) -> BootstrapResult:
    """Resample questions with replacement (same indices for both systems).

    Returns the observed difference mean(a) - mean(b) and a percentile CI.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or x.size == 0:
        raise ValueError("a and b must be non-empty 1-D arrays of equal length (paired)")
    d = x - y
    rng = np.random.default_rng(seed)
    m = d.size
    boot = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk):
        size = min(chunk, n - start)
        idx = rng.integers(0, m, size=(size, m))
        boot[start:start + size] = d[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    # Two-sided p-value: how often the resampled difference falls on the other side of 0.
    p_le = (np.count_nonzero(boot <= 0) + 1) / (n + 1)
    p_ge = (np.count_nonzero(boot >= 0) + 1) / (n + 1)
    p = min(1.0, 2 * min(p_le, p_ge))
    return BootstrapResult(
        mean_a=float(x.mean()), mean_b=float(y.mean()), diff=float(d.mean()),
        ci_low=float(lo), ci_high=float(hi), p_value=float(p), n=m, n_resamples=n, seed=seed,
    )


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm–Bonferroni step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p)
    m = p.size
    adjusted = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()
