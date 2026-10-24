"""Thread pinning and per-stage wall-clock timing."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import numpy as np

_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def pin_threads(n: int) -> dict[str, object]:
    """Pin BLAS/OpenMP/torch/faiss thread counts. Call before heavy imports where possible."""
    for var in _THREAD_VARS:
        os.environ[var] = str(n)
    info: dict[str, object] = {var: n for var in _THREAD_VARS}
    try:
        import torch

        torch.set_num_threads(n)
        info["torch_threads"] = torch.get_num_threads()
    except ImportError:
        pass
    try:
        import faiss

        faiss.omp_set_num_threads(n)
        info["faiss_threads"] = faiss.omp_get_max_threads()
    except ImportError:
        pass
    return info


class StageTimer:
    """Accumulates seconds per named stage: ``with timer("ppr"): ...``."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}

    @contextmanager
    def __call__(self, stage: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[stage] = self.seconds.get(stage, 0.0) + time.perf_counter() - t0

    def ms(self) -> dict[str, float]:
        return {k: v * 1000.0 for k, v in self.seconds.items()}


def latency_summary(ms: Sequence[float]) -> dict[str, float]:
    a = np.asarray(ms, dtype=np.float64)
    if a.size == 0:
        return {"mean_ms": float("nan"), "p50_ms": float("nan"), "p95_ms": float("nan")}
    return {
        "mean_ms": float(a.mean()),
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
    }
