"""Entity–passage graph construction and persistence.

Node layout: passages occupy indices ``[0, P)``, entities ``[P, P + E)``.
Edges are undirected (stored symmetrically in ``A``):

* passage–entity mention edges (binary or mention-count weight);
* optional entity–entity synonym edges (cosine similarity of name embeddings >= tau).
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from hopgraph.graph.entities import Mention, entity_key
from hopgraph.graph.ppr import Transition, transition_matrix

log = logging.getLogger(__name__)


@dataclass
class EntityGraph:
    passage_ids: list[str]
    entity_keys: list[str]
    entity_names: list[str]  # normalized surface name (== key when entity_key == "name")
    entity_labels: list[dict[str, int]]
    PE: sp.csr_matrix  # (P, E) mention weights
    A: sp.csr_matrix  # (P+E, P+E) symmetric adjacency
    n_synonym_edges: int = 0
    _T: Transition | None = field(default=None, repr=False)
    _name_index: dict[str, list[int]] | None = field(default=None, repr=False)

    @property
    def n_passages(self) -> int:
        return len(self.passage_ids)

    @property
    def n_entities(self) -> int:
        return len(self.entity_keys)

    @property
    def n_nodes(self) -> int:
        return self.n_passages + self.n_entities

    @property
    def transition(self) -> Transition:
        if self._T is None:
            self._T = transition_matrix(self.A)
        return self._T

    @property
    def entity_passage_counts(self) -> np.ndarray:
        """Number of passages mentioning each entity."""
        return np.asarray((self.PE > 0).sum(axis=0)).ravel()

    @property
    def specificity(self) -> np.ndarray:
        """HippoRAG node specificity s_e = 1 / |passages mentioning e|."""
        counts = self.entity_passage_counts.astype(np.float64)
        return np.divide(1.0, counts, out=np.zeros_like(counts), where=counts > 0)

    def entities_by_name(self, name: str) -> list[int]:
        if self._name_index is None:
            idx: dict[str, list[int]] = {}
            for i, n in enumerate(self.entity_names):
                idx.setdefault(n, []).append(i)
            self._name_index = idx
        return self._name_index.get(name, [])

    def stats(self) -> dict[str, float | int]:
        counts = self.entity_passage_counts
        n_mention_edges = int(self.PE.nnz)
        return {
            "passages": self.n_passages,
            "entities": self.n_entities,
            "nodes": self.n_nodes,
            "mention_edges": n_mention_edges,
            "synonym_edges": self.n_synonym_edges,
            "undirected_edges": n_mention_edges + self.n_synonym_edges,
            "adjacency_nnz": int(self.A.nnz),
            "isolated_passages": int((np.asarray((self.PE > 0).sum(axis=1)).ravel() == 0).sum()),
            "singleton_entities": int((counts == 1).sum()),
            "max_entity_degree": int(counts.max()) if counts.size else 0,
            "mean_entities_per_passage": float(n_mention_edges / max(self.n_passages, 1)),
        }

    # ------------------------------------------------------------ persistence

    def save(self, out_dir: str | Path) -> dict[str, int]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        sp.save_npz(out / "adjacency.npz", self.A)
        sp.save_npz(out / "passage_entity.npz", self.PE)
        with (out / "entities.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for k, n, lab in zip(self.entity_keys, self.entity_names, self.entity_labels, strict=True):
                f.write(json.dumps({"key": k, "name": n, "labels": lab}, ensure_ascii=False) + "\n")
        meta = {"passage_ids": self.passage_ids, "n_synonym_edges": self.n_synonym_edges}
        (out / "graph_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        return {p.name: p.stat().st_size for p in out.iterdir() if p.is_file()}

    @classmethod
    def load(cls, in_dir: str | Path) -> EntityGraph:
        d = Path(in_dir)
        meta = json.loads((d / "graph_meta.json").read_text(encoding="utf-8"))
        keys, names, labels = [], [], []
        with (d / "entities.jsonl").open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                keys.append(r["key"])
                names.append(r["name"])
                labels.append(r["labels"])
        g = cls(
            passage_ids=meta["passage_ids"], entity_keys=keys, entity_names=names,
            entity_labels=labels, PE=sp.load_npz(d / "passage_entity.npz").tocsr(),
            A=sp.load_npz(d / "adjacency.npz").tocsr(), n_synonym_edges=meta["n_synonym_edges"],
        )
        g.check()
        return g

    def check(self) -> None:
        P, E = self.n_passages, self.n_entities
        if self.PE.shape != (P, E):
            raise ValueError(f"PE shape {self.PE.shape} != ({P}, {E})")
        if self.A.shape != (P + E, P + E):
            raise ValueError(f"A shape {self.A.shape} != ({P + E}, {P + E})")
        if abs(self.A - self.A.T).sum() > 1e-9:
            raise ValueError("adjacency is not symmetric")
        if self.A[:P, :P].nnz:
            raise ValueError("unexpected passage-passage edges")


def build_graph(
    passage_ids: Sequence[str],
    mentions: Sequence[list[Mention]],
    key_mode: str = "name",
    mention_weight: str = "binary",
    synonym_pairs: Sequence[tuple[int, int, float]] | None = None,
) -> EntityGraph:
    """Build the graph from per-passage mentions.

    ``synonym_pairs`` are (entity_i, entity_j, weight) triples, indices into the
    entity order produced by :func:`collect_entities` for the same inputs.
    """
    if len(passage_ids) != len(mentions):
        raise ValueError("one mention list per passage required")
    keys, names, labels, key_index = collect_entities(mentions, key_mode)
    P, E = len(passage_ids), len(keys)

    rows, cols, vals = [], [], []
    for pi, ms in enumerate(mentions):
        c = Counter(key_index[entity_key(m, key_mode)] for m in ms)
        for ei, cnt in c.items():
            rows.append(pi)
            cols.append(ei)
            vals.append(1.0 if mention_weight == "binary" else float(cnt))
    PE = sp.csr_matrix((vals, (rows, cols)), shape=(P, E), dtype=np.float64)

    blocks = sp.bmat([[None, PE], [PE.T, None]], format="csr")
    blocks = sp.csr_matrix(blocks, shape=(P + E, P + E))
    n_syn = 0
    if synonym_pairs:
        srow: list[int] = []
        scol: list[int] = []
        sval: list[float] = []
        for i, j, w in synonym_pairs:
            if i == j:
                continue
            srow += [P + i, P + j]
            scol += [P + j, P + i]
            sval += [w, w]
            n_syn += 1
        S = sp.csr_matrix((sval, (srow, scol)), shape=(P + E, P + E))
        S.sum_duplicates()
        S.data = np.minimum(S.data, 1.0)  # duplicates of the same pair must not stack
        blocks = (blocks + S).tocsr()
    g = EntityGraph(
        passage_ids=list(passage_ids), entity_keys=keys, entity_names=names, entity_labels=labels,
        PE=PE, A=blocks, n_synonym_edges=n_syn,
    )
    g.check()
    return g


def collect_entities(
    mentions: Sequence[list[Mention]], key_mode: str
) -> tuple[list[str], list[str], list[dict[str, int]], dict[str, int]]:
    """Deterministic entity order: first appearance in passage order."""
    keys: list[str] = []
    names: list[str] = []
    labels: list[Counter] = []
    index: dict[str, int] = {}
    for ms in mentions:
        for m in ms:
            k = entity_key(m, key_mode)
            if k not in index:
                index[k] = len(keys)
                keys.append(k)
                names.append(m.name)
                labels.append(Counter())
            labels[index[k]][m.label] += 1
    return keys, names, [dict(c) for c in labels], index


def synonym_pairs_from_embeddings(
    emb: np.ndarray, tau: float, top_k: int = 10
) -> list[tuple[int, int, float]]:
    """Entity pairs with cosine similarity >= tau among each entity's top_k neighbours."""
    import faiss

    x = np.ascontiguousarray(emb, dtype=np.float32)
    index = faiss.IndexFlatIP(x.shape[1])
    index.add(x)
    k = min(top_k + 1, x.shape[0])
    sims, nbrs = index.search(x, k)
    pairs: dict[tuple[int, int], float] = {}
    for i in range(x.shape[0]):
        for s, j in zip(sims[i], nbrs[i], strict=True):
            if j < 0 or j == i or s < tau:
                continue
            a, b = (i, int(j)) if i < j else (int(j), i)
            pairs[(a, b)] = max(pairs.get((a, b), 0.0), float(s))
    return [(a, b, w) for (a, b), w in sorted(pairs.items())]
