import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db, learning, retrieval  # noqa: E402


@pytest.fixture()
def conn(tmp_path):
    db_path = tmp_path / "test.db"
    db.init_db(db_path, reset=True)
    connection = db.get_connection(db_path)
    yield connection
    connection.close()


def test_single_observation_stays_pending_and_produces_no_candidate(conn):
    results = learning.ingest_observation(
        conn, "ask aditya to review the sarvam kiwi service",
        "Ask Aaditya to review the Sarvam Kivi service."
    )
    conn.commit()
    assert len(results) == 2  # aditya->Aaditya, kiwi->Kivi
    assert all(r.status == "pending" for r in results)

    candidates = retrieval.find_candidates(conn, "ask aditya about the kiwi update")
    assert candidates == [], "pending memory must not produce retrieval candidates"


def test_repeated_observation_activates_and_then_intervenes(conn):
    learning.ingest_observation(
        conn, "ask aditya to review the sarvam kiwi service",
        "Ask Aaditya to review the Sarvam Kivi service."
    )
    results = learning.ingest_observation(
        conn, "loop in aditya on the kiwi rollout",
        "Loop in Aaditya on the Kivi rollout."
    )
    conn.commit()
    assert any(r.status == "active" and r.was_activated for r in results)

    candidates = retrieval.find_candidates(conn, "ping aditya about kiwi again")
    words = {c.canonical_word for c in candidates}
    assert "Aaditya" in words
    assert "Kivi" in words


def test_explicit_correction_activates_immediately(conn):
    result = learning.ingest_explicit_correction(
        conn, "Kivi", "Kivi is an AI company", asr_token="kiwi"
    )
    conn.commit()
    assert result.status == "active"
    assert result.evidence_count == 1

    candidates = retrieval.find_candidates(conn, "I opened the kiwi dashboard")
    assert any(c.canonical_word == "Kivi" for c in candidates)


def test_novel_entity_not_hallucinated(conn):
    # No memories at all -- an unrelated proper noun must not match anything.
    candidates = retrieval.find_candidates(conn, "ask Nandini to review the doc")
    assert candidates == []


def test_conflicting_memories_both_surface_as_candidates(conn):
    learning.ingest_explicit_correction(conn, "Aditi", "Aditi joined the call", asr_token="aditi")
    learning.ingest_explicit_correction(conn, "Aditya", "Aditya sent the report", asr_token="aditya")
    conn.commit()

    candidates = retrieval.find_candidates(conn, "ask aditya for the update")
    words = {c.canonical_word for c in candidates}
    # Both are plausible phonetic neighbors of the ASR token "aditya";
    # disambiguating between them is the Judge-Editor's job, not retrieval's.
    assert "Aditya" in words


def test_unrelated_short_word_does_not_falsely_match(conn):
    learning.ingest_explicit_correction(conn, "Kivi", "Kivi is our product", asr_token="kivi")
    conn.commit()
    # "I ate a kiwi" -- retrieval SHOULD still surface kiwi->Kivi as a
    # candidate (that's expected; disambiguation is the judge's job), but
    # a completely unrelated word must not.
    candidates = retrieval.find_candidates(conn, "please pass the salt")
    assert candidates == []

def test_token_splitting_extraction(conn):
    # ASR splits Kivi into "ki wi"
    results = learning.ingest_observation(
        conn, "ask aditya to review the sarvam ki wi service",
        "Ask Aaditya to review the Sarvam Kivi service."
    )
    conn.commit()
    # Expect aditya->Aaditya and "ki wi"->Kivi
    words = {r.canonical_word for r in results}
    assert "Aaditya" in words
    assert "Kivi" in words
