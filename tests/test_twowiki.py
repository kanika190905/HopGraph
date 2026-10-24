import pytest

from hopgraph.ingestion.corpus import DataError
from hopgraph.ingestion.twowiki import load_twowiki
from tests.conftest import FIXTURES

TINY = FIXTURES / "tiny_corpus" / "twowiki_tiny.json"


def test_twowiki_gold_and_dedupe():
    with pytest.raises(DataError, match="skipped"):
        load_twowiki(TINY)  # 1 of 3 records malformed > 1%
    passages, questions, stats = load_twowiki(TINY, max_skip_frac=0.5)
    assert stats.n_skipped == 1 and stats.skip_reasons == {"empty_question": 1}
    # 7 candidates: duplicate "Seoul" within a1 and "Bong Joon-ho" across a1/b2 merge.
    assert stats.n_candidates == 7 and len(passages) == 5
    by_id = {p.id: p for p in passages}
    a1 = next(q for q in questions if q.id == "a1")
    assert {by_id[g].title for g in a1.gold_ids} == {"Parasite (2019 film)", "Bong Joon-ho"}
    assert a1.hops == 2 and a1.meta["hop_type"] == "compositional"
    assert by_id[a1.gold_ids[0]].text == "Parasite is a 2019 film. It was directed by Bong Joon-ho."
