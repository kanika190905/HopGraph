"""2WikiMultiHopQA loader (official release, ``data/dev.json`` from github.com/Alab-NII/2wikimultihop).

Record schema (verified on dev.json, 12,576 records): ``_id, type, question, context
[[title, [sentences]]] (10 paragraphs), supporting_facts [[title, sent_id]], evidences, answer``.
Gold passages are the context paragraphs whose title appears in ``supporting_facts``
(2 titles, or 4 for ``bridge_comparison``). The dev file repeats a title within one
context 523 times; those repeats always carry identical text and are never gold, so
content-hash deduplication merges them. ``hops`` is the number of gold paragraphs.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from hopgraph.ingestion.corpus import DataError, RawCandidate, dedupe, map_gold, validate_corpus
from hopgraph.ingestion.musique import LoadStats
from hopgraph.types import Passage, Question

log = logging.getLogger(__name__)


def _record_problem(rec: object) -> str | None:
    if not isinstance(rec, dict):
        return "not_a_record"
    if not isinstance(rec.get("_id"), str) or not rec["_id"]:
        return "missing_id"
    if not isinstance(rec.get("question"), str) or not rec["question"].strip():
        return "empty_question"
    ctx = rec.get("context")
    if not isinstance(ctx, list) or not ctx:
        return "no_paragraphs"
    for item in ctx:
        if (not isinstance(item, list) or len(item) != 2 or not isinstance(item[1], list)
                or not " ".join(item[1]).strip()):
            return "malformed_paragraph"
    sup = rec.get("supporting_facts")
    if not isinstance(sup, list) or not sup:
        return "no_supporting"
    titles = {t for t, _ in ctx}
    if not {s[0] for s in sup} <= titles:
        return "supporting_title_not_in_context"
    return None


def load_twowiki(
    path: str | Path, n: int | None = None, seed: int = 0, split: str = "dev",
    max_skip_frac: float = 0.01, offset: int = 0,
) -> tuple[list[Passage], list[Question], LoadStats]:
    """Seeded sample (positions ``[offset, offset + n)``) plus its merged, deduplicated corpus."""
    path = Path(path)
    stats = LoadStats(skip_reasons={})
    assert stats.skip_reasons is not None
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise DataError(f"{path}: not valid JSON ({e})") from e
    if not isinstance(records, list) or not records:
        raise DataError(f"{path}: expected a non-empty JSON array")
    valid = []
    for i, rec in enumerate(records):
        stats.n_records += 1
        reason = _record_problem(rec)
        if reason:
            stats.n_skipped += 1
            stats.skip_reasons[reason] = stats.skip_reasons.get(reason, 0) + 1
            log.warning("skipping record %d: %s", i, reason)
            continue
        valid.append(rec)
    skip_frac = stats.n_skipped / stats.n_records
    if skip_frac > max_skip_frac:
        raise DataError(f"{path}: skipped {skip_frac:.2%} of records > {max_skip_frac:.2%}; "
                        f"reasons={stats.skip_reasons}")

    order = list(range(len(valid)))
    random.Random(seed).shuffle(order)
    if not 0 <= offset < len(order):
        raise DataError(f"offset {offset} out of range for {len(order)} valid records")
    selected = [valid[i] for i in order[offset:(offset + n) if n is not None else None]]
    stats.n_selected = len(selected)

    candidates: list[RawCandidate] = []
    raw_gold: dict[str, list[str]] = {}
    for rec in selected:
        qid = rec["_id"]
        gold_titles = {s[0] for s in rec["supporting_facts"]}
        raw_gold[qid] = []
        for idx, (title, sents) in enumerate(rec["context"]):
            rid = f"{qid}::{idx}"
            candidates.append(RawCandidate(rid, title, " ".join(sents), f"2wiki:{split}"))
            if title in gold_titles:
                raw_gold[qid].append(rid)
    stats.n_candidates = len(candidates)
    result = dedupe(candidates)
    stats.n_passages = len(result.passages)
    stats.n_duplicates = result.n_duplicates

    questions = []
    for rec in selected:
        qid = rec["_id"]
        gold = map_gold(raw_gold[qid], result.raw_to_canonical, qid)
        n_titles = len({s[0] for s in rec["supporting_facts"]})
        if len(gold) != n_titles:
            raise DataError(f"question {qid}: {n_titles} supporting titles but {len(gold)} gold passages")
        questions.append(Question(
            id=qid, text=rec["question"].strip(), gold_ids=gold, hops=len(gold), split=split,
            answer=rec.get("answer", ""),
            meta={"hop_type": rec.get("type", ""), "n_candidates": len(rec["context"])},
        ))
    validate_corpus(result.passages, questions)
    log.info("2wiki %s: %d records, %d skipped, %d selected, %d candidates -> %d passages (%d dups)",
             split, stats.n_records, stats.n_skipped, stats.n_selected, stats.n_candidates,
             stats.n_passages, stats.n_duplicates)
    return result.passages, questions, stats
