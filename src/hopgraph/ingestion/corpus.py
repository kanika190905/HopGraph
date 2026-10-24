"""Corpus construction: deduplication, gold-ID mapping and JSONL persistence."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from hopgraph.types import Passage, Question

log = logging.getLogger(__name__)

_WS = re.compile(r"\s+")


class DataError(ValueError):
    """Corrupted input, empty corpus, or a gold passage that cannot be mapped."""


def normalize_for_hash(s: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFKC", s)).strip().lower()


def passage_hash(title: str, text: str) -> str:
    """Stable content ID: same normalized title + text -> same passage ID."""
    key = normalize_for_hash(title) + "\n" + normalize_for_hash(text)
    return "p" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:15]


@dataclass
class RawCandidate:
    """A candidate paragraph as it appears in one benchmark record."""

    raw_id: str  # e.g. "<question_id>::<paragraph_idx>"
    title: str
    text: str
    source: str


@dataclass
class DedupeResult:
    passages: list[Passage]
    raw_to_canonical: dict[str, str]
    n_raw: int
    n_duplicates: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_duplicates = self.n_raw - len(self.passages)


def dedupe(candidates: Iterable[RawCandidate]) -> DedupeResult:
    """Merge identical candidate paragraphs into one canonical passage.

    First occurrence wins for the stored (un-normalized) title/text, so output
    order and content are deterministic for a fixed input order.
    """
    by_id: dict[str, Passage] = {}
    mapping: dict[str, str] = {}
    n_raw = 0
    for c in candidates:
        n_raw += 1
        if c.raw_id in mapping:
            raise DataError(f"duplicate raw candidate id {c.raw_id!r}")
        pid = passage_hash(c.title, c.text)
        if pid not in by_id:
            by_id[pid] = Passage(id=pid, title=c.title.strip(), text=c.text.strip(), source=c.source)
        mapping[c.raw_id] = pid
    return DedupeResult(passages=list(by_id.values()), raw_to_canonical=mapping, n_raw=n_raw)


def map_gold(raw_gold: Sequence[str], mapping: dict[str, str], qid: str) -> tuple[str, ...]:
    """Map raw gold candidate IDs to canonical IDs; fail loudly on any miss."""
    out: list[str] = []
    for rid in raw_gold:
        if rid not in mapping:
            raise DataError(f"question {qid}: gold candidate {rid!r} has no canonical passage")
        pid = mapping[rid]
        if pid not in out:
            out.append(pid)
    if not out:
        raise DataError(f"question {qid}: no gold passages")
    return tuple(out)


def validate_corpus(passages: Sequence[Passage], questions: Sequence[Question]) -> None:
    if not passages:
        raise DataError("empty corpus")
    if not questions:
        raise DataError("no questions")
    ids = {p.id for p in passages}
    if len(ids) != len(passages):
        raise DataError("passage IDs are not unique")
    for q in questions:
        if not q.gold_ids:
            raise DataError(f"question {q.id} has no gold IDs")
        missing = [g for g in q.gold_ids if g not in ids]
        if missing:
            raise DataError(f"question {q.id}: gold IDs {missing} not in corpus")


# ---------------------------------------------------------------- persistence


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_corpus(out_dir: Path, passages: Sequence[Passage], questions: Sequence[Question]) -> None:
    write_jsonl(
        out_dir / "passages.jsonl",
        ({"id": p.id, "title": p.title, "text": p.text, "source": p.source} for p in passages),
    )
    write_jsonl(
        out_dir / "questions.jsonl",
        (
            {
                "id": q.id, "text": q.text, "gold_ids": list(q.gold_ids), "hops": q.hops,
                "split": q.split, "answer": q.answer, "meta": q.meta,
            }
            for q in questions
        ),
    )


def load_passages(path: Path) -> list[Passage]:
    return [Passage(r["id"], r["title"], r["text"], r.get("source", "")) for r in read_jsonl(path)]


def load_questions(path: Path) -> list[Question]:
    return [
        Question(
            id=r["id"], text=r["text"], gold_ids=tuple(r["gold_ids"]), hops=int(r["hops"]),
            split=r["split"], answer=r.get("answer", ""), meta=r.get("meta", {}),
        )
        for r in read_jsonl(path)
    ]


def load_corpus(processed_dir: str | Path) -> tuple[list[Passage], list[Question]]:
    d = Path(processed_dir)
    passages = load_passages(d / "passages.jsonl")
    questions = load_questions(d / "questions.jsonl")
    validate_corpus(passages, questions)
    return passages, questions


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def corpus_hash(processed_dir: str | Path) -> str:
    """Content hash of the processed corpus; part of every run ID."""
    d = Path(processed_dir)
    h = hashlib.sha256()
    for name in ("passages.jsonl", "questions.jsonl"):
        h.update(file_sha256(d / name).encode())
    return h.hexdigest()[:12]
