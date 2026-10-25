"""MuSiQue-Ans loader.

Record schema (verified on musique_ans_v1.0_dev.jsonl, 2,417 records):
``id, paragraphs[idx, title, paragraph_text, is_supporting], question,
question_decomposition[id, question, answer, paragraph_support_idx], answer,
answer_aliases, answerable``. The hop count is the leading digit of the ID
prefix (``2hop``, ``3hop1``, ``4hop3`` ...).
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path

from hopgraph.ingestion.corpus import DataError, RawCandidate, dedupe, map_gold, validate_corpus
from hopgraph.types import Passage, Question

log = logging.getLogger(__name__)

_HOPS = re.compile(r"^(\d)hop")


@dataclass
class LoadStats:
    n_records: int = 0
    n_skipped: int = 0
    n_selected: int = 0
    n_candidates: int = 0
    n_passages: int = 0
    n_duplicates: int = 0
    skip_reasons: dict[str, int] | None = None


def _record_problem(rec: dict) -> str | None:
    if not isinstance(rec.get("id"), str) or not rec["id"]:
        return "missing_id"
    if not isinstance(rec.get("question"), str) or not rec["question"].strip():
        return "empty_question"
    paras = rec.get("paragraphs")
    if not isinstance(paras, list) or not paras:
        return "no_paragraphs"
    for p in paras:
        if not isinstance(p, dict) or not str(p.get("paragraph_text", "")).strip():
            return "empty_paragraph"
        if "idx" not in p or "is_supporting" not in p:
            return "malformed_paragraph"
    if not any(p["is_supporting"] for p in paras):
        return "no_supporting"
    if not _HOPS.match(rec["id"]):
        return "unparseable_hops"
    return None


def load_musique(
    path: str | Path,
    n: int | None = None,
    seed: int = 0,
    split: str = "dev",
    max_skip_frac: float = 0.01,
    offset: int = 0,
) -> tuple[list[Passage], list[Question], LoadStats]:
    """Load ``n`` questions (seeded sample) and their merged, deduplicated corpus.

    Sampling is a seeded permutation of the valid records, then positions
    ``[offset, offset + n)``. With offset 0, smaller subsets are nested inside larger
    ones; a non-zero offset gives a disjoint held-out slice of the same permutation.
    """
    path = Path(path)
    stats = LoadStats(skip_reasons={})
    assert stats.skip_reasons is not None
    # Pass 1: validate every record (for the skip-rate check) but keep only line numbers,
    # so sampling a few hundred questions from the 241 MB train file stays memory-light.
    valid: list[int] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            stats.n_records += 1
            reason: str | None
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                reason = "bad_json"
            else:
                reason = _record_problem(rec)
            if reason:
                stats.n_skipped += 1
                stats.skip_reasons[reason] = stats.skip_reasons.get(reason, 0) + 1
                log.warning("skipping line %d: %s", line_no, reason)
                continue
            valid.append(line_no)

    if stats.n_records == 0:
        raise DataError(f"{path}: no records")
    skip_frac = stats.n_skipped / stats.n_records
    if skip_frac > max_skip_frac:
        raise DataError(
            f"{path}: skipped {stats.n_skipped}/{stats.n_records} ({skip_frac:.2%}) records "
            f"> max_skip_frac={max_skip_frac:.2%}; reasons={stats.skip_reasons}"
        )

    order = list(range(len(valid)))
    random.Random(seed).shuffle(order)
    if not 0 <= offset < len(order):
        raise DataError(f"offset {offset} out of range for {len(order)} valid records")
    chosen = [valid[i] for i in order[offset:(offset + n) if n is not None else None]]
    # Pass 2: parse only the chosen lines, then restore the sampled order.
    wanted = set(chosen)
    by_line: dict[int, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line_no in wanted:
                by_line[line_no] = json.loads(line)
    selected = [by_line[ln] for ln in chosen]
    stats.n_selected = len(selected)

    candidates: list[RawCandidate] = []
    raw_gold: dict[str, list[str]] = {}
    for rec in selected:
        qid = rec["id"]
        raw_gold[qid] = []
        for p in rec["paragraphs"]:
            rid = f"{qid}::{p['idx']}"
            candidates.append(
                RawCandidate(raw_id=rid, title=p.get("title", ""), text=p["paragraph_text"],
                             source=f"musique:{split}")
            )
            if p["is_supporting"]:
                raw_gold[qid].append(rid)
    stats.n_candidates = len(candidates)

    result = dedupe(candidates)
    stats.n_passages = len(result.passages)
    stats.n_duplicates = result.n_duplicates

    questions: list[Question] = []
    for rec in selected:
        qid = rec["id"]
        gold = map_gold(raw_gold[qid], result.raw_to_canonical, qid)
        if len(gold) != len(raw_gold[qid]):
            log.warning("question %s: %d gold paragraphs collapsed to %d after dedupe",
                        qid, len(raw_gold[qid]), len(gold))
        m = _HOPS.match(qid)
        assert m is not None
        questions.append(
            Question(
                id=qid, text=rec["question"].strip(), gold_ids=gold, hops=int(m.group(1)),
                split=split, answer=rec.get("answer", ""),
                meta={"hop_type": qid.split("__")[0],
                      "n_candidates": len(rec["paragraphs"]),
                      "decomposition": [d.get("question", "") for d in rec.get("question_decomposition", [])]},
            )
        )

    validate_corpus(result.passages, questions)
    log.info(
        "musique %s: %d records, %d skipped, %d selected, %d candidates -> %d passages (%d dups)",
        split, stats.n_records, stats.n_skipped, stats.n_selected, stats.n_candidates,
        stats.n_passages, stats.n_duplicates,
    )
    return result.passages, questions, stats
