"""LLM-free entity extraction (spaCy NER + passage titles) and entity-name normalization."""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from hopgraph.types import Passage

log = logging.getLogger(__name__)

# Number-like labels are excluded by default (they create hub nodes, not bridges).
NUMERIC_LABELS = frozenset({"CARDINAL", "ORDINAL", "PERCENT", "QUANTITY", "MONEY", "TIME"})

# Coarse label groups, kept as metadata and used when entity_key == "name_label".
COARSE = {
    "PERSON": "PERSON",
    "ORG": "ORG",
    "NORP": "GROUP",
    "GPE": "PLACE", "LOC": "PLACE", "FAC": "PLACE",
    "WORK_OF_ART": "WORK", "PRODUCT": "WORK",
    "EVENT": "EVENT",
    "LAW": "MISC", "LANGUAGE": "MISC",
    "DATE": "DATE",
    "TITLE": "TITLE",
}

_WS = re.compile(r"\s+")
_JOIN = re.compile(r"[.'’`]")  # removed without a space: "U.S." -> "us", "O'Neil" -> "oneil"
_NON_WORD = re.compile(r"[^\w\s]|_")  # every other punctuation mark becomes a space
_POSSESSIVE = re.compile(r"['’]s$")
_TRAILING_PAREN = re.compile(r"\s*\([^()]*\)\s*$")


def normalize_entity(s: str) -> str:
    """NFKC, lowercase, strip possessive and punctuation, drop a leading "the", collapse spaces."""
    s = unicodedata.normalize("NFKC", s).lower().strip()
    s = _POSSESSIVE.sub("", s)
    s = _JOIN.sub("", s)
    s = _NON_WORD.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    if s.startswith("the "):
        s = s[4:]
    return s


def title_entity(title: str) -> str:
    """Wikipedia-style title without its disambiguator: "Green (Steve Hillage album)" -> "green"."""
    return normalize_entity(_TRAILING_PAREN.sub("", title))


def is_valid_name(norm: str) -> bool:
    return len(norm) >= 2 and any(ch.isalnum() for ch in norm)


@dataclass(frozen=True)
class Mention:
    name: str  # normalized
    label: str  # coarse label
    surface: str  # original text


def entity_key(m: Mention, mode: str) -> str:
    return m.name if mode == "name" else f"{m.name}|{m.label}"


def load_ner(model: str = "en_core_web_sm"):
    """spaCy pipeline with only the NER component (its tok2vec is internal in *_sm)."""
    import spacy

    return spacy.load(model, exclude=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])


def mentions_from_doc(doc, labels: frozenset[str]) -> list[Mention]:
    out = []
    for ent in doc.ents:
        if ent.label_ not in labels:
            continue
        name = normalize_entity(ent.text)
        if is_valid_name(name):
            out.append(Mention(name, COARSE.get(ent.label_, "MISC"), ent.text))
    return out


def extract_entities(
    passages: Sequence[Passage],
    nlp,
    labels: Iterable[str],
    batch_size: int = 128,
    n_process: int = 1,
    include_titles: bool = True,
) -> list[list[Mention]]:
    """Mentions per passage, in passage order. Titles are added as TITLE mentions."""
    label_set = frozenset(labels) - NUMERIC_LABELS
    out: list[list[Mention]] = []
    texts = (p.content for p in passages)
    for p, doc in zip(passages, nlp.pipe(texts, batch_size=batch_size, n_process=n_process),
                      strict=True):
        ms = mentions_from_doc(doc, label_set)
        if include_titles and p.title:
            t = title_entity(p.title)
            if is_valid_name(t):
                ms.append(Mention(t, "TITLE", p.title))
        out.append(ms)
    return out


def save_mentions(path: Path, passage_ids: Sequence[str], mentions: Sequence[list[Mention]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for pid, ms in zip(passage_ids, mentions, strict=True):
            f.write(json.dumps({"id": pid, "m": [[m.name, m.label, m.surface] for m in ms]},
                               ensure_ascii=False) + "\n")


def load_mentions(path: Path) -> tuple[list[str], list[list[Mention]]]:
    ids, out = [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ids.append(r["id"])
            out.append([Mention(*m) for m in r["m"]])
    return ids, out


def ambiguity_stats(mentions: Sequence[list[Mention]]) -> dict[str, float]:
    """How often one normalized name carries more than one coarse NER label."""
    labels: dict[str, Counter] = {}
    for ms in mentions:
        for m in ms:
            if m.label != "TITLE":
                labels.setdefault(m.name, Counter())[m.label] += 1
    n = len(labels)
    multi = sum(1 for c in labels.values() if len(c) > 1)
    return {"ner_names": n, "names_with_multiple_labels": multi,
            "frac_multi_label": multi / n if n else 0.0}
