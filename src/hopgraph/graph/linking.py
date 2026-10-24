"""Query entity linking with logged fallbacks.

Order of attempts:
1. spaCy NER mentions in the query -> exact normalized-name lookup.
2. Dictionary n-gram match of the normalized query against entity names
   (longest match first, non-overlapping, stopword-only n-grams skipped).
3. NER mentions without an exact match -> nearest entity-name embedding (cos >= tau).
4. Only if nothing is linked yet: noun chunks -> exact lookup, then embedding match.
5. Nothing linked: the caller seeds PPR from first-stage passages only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from hopgraph.config import LinkingConfig
from hopgraph.graph.build import EntityGraph
from hopgraph.graph.entities import NUMERIC_LABELS, is_valid_name, normalize_entity

log = logging.getLogger(__name__)


def _stopwords() -> frozenset[str]:
    try:
        from spacy.lang.en.stop_words import STOP_WORDS

        return frozenset(STOP_WORDS)
    except ImportError:  # pragma: no cover - spaCy is a core dependency
        return frozenset({"the", "a", "an", "of", "in", "who", "what", "which", "is", "was"})


STOPWORDS = _stopwords()


@dataclass
class LinkedEntity:
    entity: int
    sim: float
    method: str  # ner | ngram | embed | noun_chunk | noun_chunk_embed
    mention: str


@dataclass
class LinkResult:
    entities: list[LinkedEntity] = field(default_factory=list)

    @property
    def path(self) -> str:
        if not self.entities:
            return "none"
        return "+".join(sorted({e.method for e in self.entities}))


class QueryLinker:
    def __init__(
        self,
        graph: EntityGraph,
        cfg: LinkingConfig,
        nlp=None,  # full spaCy pipeline (needs parser for noun chunks)
        encoder=None,  # has encode_passages(list[str]) -> unit vectors
        entity_emb: np.ndarray | None = None,
        labels: frozenset[str] | None = None,
    ):
        self.g = graph
        self.cfg = cfg
        self.nlp = nlp
        self.encoder = encoder
        self.labels = (labels or frozenset()) - NUMERIC_LABELS
        self._index = None
        if entity_emb is not None and cfg.embed_tau is not None and encoder is not None:
            import faiss

            x = np.ascontiguousarray(entity_emb, dtype=np.float32)
            if x.shape[0] != graph.n_entities:
                raise ValueError("entity embeddings do not match graph entities")
            self._index = faiss.IndexFlatIP(x.shape[1])
            self._index.add(x)

    def _exact(self, name: str) -> list[int]:
        return self.g.entities_by_name(name) if is_valid_name(name) else []

    def _embed(self, mentions: list[str]) -> list[tuple[int, float]]:
        """One (entity, sim) per mention; (-1, 0.0) when unmatched or embedding is disabled."""
        if self._index is None or not mentions:
            return [(-1, 0.0)] * len(mentions)
        q = self.encoder.encode_passages(mentions)
        sims, idx = self._index.search(np.ascontiguousarray(q, dtype=np.float32), 1)
        tau = self.cfg.embed_tau
        return [(int(i[0]), float(s[0])) if i[0] >= 0 and s[0] >= tau else (-1, 0.0)
                for s, i in zip(sims, idx, strict=True)]

    def _ngrams(self, query: str) -> list[tuple[str, list[int]]]:
        toks = normalize_entity(query).split()
        used = [False] * len(toks)
        found = []
        for n in range(min(self.cfg.max_ngram, len(toks)), 0, -1):
            for i in range(len(toks) - n + 1):
                if any(used[i:i + n]):
                    continue
                gram = toks[i:i + n]
                if all(t in STOPWORDS for t in gram):
                    continue
                name = " ".join(gram)
                ents = self._exact(name)
                if ents:
                    found.append((name, ents))
                    for j in range(i, i + n):
                        used[j] = True
        return found

    def link(self, query: str) -> LinkResult:
        res = LinkResult()
        seen: dict[int, LinkedEntity] = {}

        def add(e: int, sim: float, method: str, mention: str) -> None:
            if e not in seen or seen[e].sim < sim:
                seen[e] = LinkedEntity(e, sim, method, mention)

        doc = self.nlp(query) if self.nlp is not None else None
        unmatched: list[str] = []
        if doc is not None and self.cfg.use_ner:
            for ent in doc.ents:
                if ent.label_ not in self.labels:
                    continue
                name = normalize_entity(ent.text)
                hits = self._exact(name)
                for e in hits:
                    add(e, 1.0, "ner", ent.text)
                if not hits and is_valid_name(name):
                    unmatched.append(ent.text)
        if self.cfg.use_ngram:
            for name, ents in self._ngrams(query):
                for e in ents:
                    add(e, 1.0, "ngram", name)
        for text, (e, sim) in zip(unmatched, self._embed(unmatched), strict=True):
            if e >= 0:
                add(e, sim, "embed", text)
        if not seen and doc is not None and self.cfg.use_noun_chunks:
            chunks = []
            try:
                chunks = [c.text for c in doc.noun_chunks]
            except ValueError:  # pipeline without a parser
                chunks = []
            leftover = []
            for c in chunks:
                hits = self._exact(normalize_entity(c))
                for e in hits:
                    add(e, 1.0, "noun_chunk", c)
                if not hits:
                    leftover.append(c)
            for text, (e, sim) in zip(leftover, self._embed(leftover), strict=True):
                if e >= 0:
                    add(e, sim, "noun_chunk_embed", text)
        res.entities = sorted(seen.values(), key=lambda x: x.entity)
        return res
