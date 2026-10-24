import pytest

from hopgraph.ingestion.corpus import DataError
from hopgraph.ingestion.documents import chunk_words, load_documents


def test_chunk_words_overlap_and_coverage():
    words = [f"w{i}" for i in range(25)]
    chunks = chunk_words(" ".join(words), size=10, overlap=3)
    assert [c.split()[0] for c in chunks] == ["w0", "w7", "w14"]
    assert chunks[-1].split()[-1] == "w24"
    assert chunks[0].split()[-3:] == chunks[1].split()[:3]
    assert chunk_words("   ", 10, 2) == []
    assert chunk_words("a b c", 10, 2) == ["a b c"]
    with pytest.raises(ValueError):
        chunk_words("a", 5, 5)


def test_load_documents_ids_and_titles(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "notes.md").write_text("# Parasite\n\n" + "word " * 40, encoding="utf-8")
    (tmp_path / "sub" / "plain.txt").write_text("Seoul is the capital.", encoding="utf-8")
    (tmp_path / "skip.pdf").write_bytes(b"%PDF")
    ps = load_documents(tmp_path, size=20, overlap=5)
    ids = [p.id for p in ps]
    assert ids[0] == "notes.md#0" and ids[-1] == "sub/plain.txt#0"
    assert ps[0].title == "Parasite" and ps[-1].title == "plain"
    assert len(ids) == len(set(ids))
    with pytest.raises(DataError):
        load_documents(tmp_path / "sub", suffixes=(".md",))
