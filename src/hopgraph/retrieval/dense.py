"""Baseline B: dense retrieval with BGE-small + FAISS ``IndexFlatIP``.

Embeddings are L2-normalized, so the inner product equals cosine similarity:
``sim(q, d) = e_q . e_d = cos(e_q, e_d)`` when ``||e_q|| = ||e_d|| = 1``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np

from hopgraph.config import DenseConfig
from hopgraph.types import Passage, RankedList

log = logging.getLogger(__name__)


class Encoder(Protocol):
    """Anything that maps texts to an (n, dim) float32 array of unit vectors."""

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray: ...
    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...


class STEncoder:
    """Sentence-Transformers encoder (BGE). Query instruction only on queries, per model card."""

    def __init__(self, cfg: DenseConfig, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.cfg = cfg
        self.model = SentenceTransformer(cfg.model, revision=cfg.revision, device=device)
        self.model.max_seq_length = cfg.max_seq_length

    def _encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        emb = self.model.encode(
            list(texts), batch_size=batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        return np.ascontiguousarray(emb, dtype=np.float32)

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.cfg.batch_size)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([self.cfg.query_prefix + t for t in texts], self.cfg.batch_size)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(x / norms, dtype=np.float32)


class DenseRetriever:
    def __init__(self, encoder: Encoder | None, top_k: int = 100):
        self.encoder = encoder
        self.top_k = top_k
        self.index: faiss.Index | None = None
        self.ids: list[str] = []

    def build(self, passages: Sequence[Passage], embeddings: np.ndarray | None = None) -> np.ndarray:
        self.ids = [p.id for p in passages]
        if embeddings is None:
            assert self.encoder is not None
            embeddings = self.encoder.encode_passages([p.content for p in passages])
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if embeddings.shape[0] != len(self.ids):
            raise ValueError(f"{embeddings.shape[0]} embeddings for {len(self.ids)} passages")
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError("passage embeddings are not unit-normalized; IP != cosine")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        return embeddings

    def search_vectors(self, qemb: np.ndarray, k: int | None = None) -> list[RankedList]:
        assert self.index is not None
        k = min(k or self.top_k, self.index.ntotal)
        scores, idx = self.index.search(np.ascontiguousarray(qemb, dtype=np.float32), k)
        out = []
        for srow, irow in zip(scores, idx, strict=True):
            out.append(RankedList.from_scores(
                (self.ids[i], float(s)) for s, i in zip(srow, irow, strict=True) if i >= 0
            ))
        return out

    def search(self, query: str, k: int | None = None) -> RankedList:
        assert self.encoder is not None
        return self.search_vectors(self.encoder.encode_queries([query]), k)[0]

    def save(self, out_dir: str | Path, embeddings: np.ndarray, meta: dict) -> None:
        assert self.index is not None
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "embeddings.npy", embeddings.astype(np.float32))
        faiss.write_index(self.index, str(out / "faiss.index"))
        (out / "passage_ids.json").write_text(json.dumps(self.ids), encoding="utf-8")
        (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, in_dir: str | Path, encoder: Encoder | None, top_k: int = 100) -> DenseRetriever:
        d = Path(in_dir)
        r = cls(encoder, top_k)
        index = faiss.read_index(str(d / "faiss.index"))
        r.index = index
        r.ids = json.loads((d / "passage_ids.json").read_text(encoding="utf-8"))
        if index.ntotal != len(r.ids):
            raise ValueError("FAISS index size does not match passage_ids.json")
        return r

    @staticmethod
    def load_embeddings(in_dir: str | Path) -> np.ndarray:
        """Memory-mapped read of the cached float32 embedding matrix."""
        return np.load(Path(in_dir) / "embeddings.npy", mmap_mode="r")
