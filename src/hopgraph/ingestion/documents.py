"""User documents (.txt / .md) -> overlapping word-window passages.

Each passage ID is ``<doc_id>#<chunk_no>``; ``doc_id`` is the file path relative to the
input root, so IDs are stable across runs. The title is the first Markdown heading (or the
file stem), so title entities work the same way as for benchmark passages.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from hopgraph.ingestion.corpus import DataError
from hopgraph.types import Passage

_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", re.M)
SUFFIXES = (".txt", ".md")


def chunk_words(text: str, size: int = 150, overlap: int = 30) -> list[str]:
    """Split on whitespace into windows of ``size`` words, consecutive windows sharing ``overlap``."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    words = text.split()
    if not words:
        return []
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def load_documents(
    root: str | Path, size: int = 150, overlap: int = 30, suffixes: Iterable[str] = SUFFIXES,
) -> list[Passage]:
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in set(suffixes))
    if not files:
        raise DataError(f"no {'/'.join(suffixes)} files under {root}")
    passages = []
    for f in files:
        text = f.read_text(encoding="utf-8-sig")
        m = _HEADING.search(text) if f.suffix.lower() == ".md" else None
        title = m.group(1).strip() if m else f.stem
        doc_id = f.relative_to(root).as_posix()
        for i, chunk in enumerate(chunk_words(text, size, overlap)):
            passages.append(Passage(id=f"{doc_id}#{i}", title=title, text=chunk, source=f"doc:{doc_id}"))
    if not passages:
        raise DataError(f"documents under {root} contain no text")
    return passages
