"""
Lifecycle 1: Observation -> Learning.

Two ways evidence enters the system:

1. implicit_observation: we're shown an (asr_text, formatted_text) pair
   (formatted_text stands in for what the user ultimately kept/approved).
   Token-level diffs are extracted and treated as *low-confidence* signals
   -- a single occurrence creates a 'pending' memory that cannot yet affect
   output. Only repeated, independent observations of the same correction
   promote it to 'active'.

2. explicit_correction: the user directly tells the system "this word
   should be spelled X" (e.g. via a manual dictionary entry). This is
   high-confidence by construction -- there's no ambiguity to accumulate
   evidence against -- so it activates immediately.

This asymmetry is the core defense against learning ASR noise as fact:
inferred corrections must repeat before they're trusted; declared
corrections are trusted at once because a human vouched for them directly.
"""
import sqlite3
from dataclasses import dataclass

from . import config
from .diff import extract_corrections, TokenCorrection
from .phonetic import compute_phonetic_keys
from .tokenize_utils import normalize


@dataclass
class LearningResult:
    canonical_word: str
    dictionary_id: int
    evidence_count: int
    status: str          # 'pending' | 'active'
    was_new: bool
    was_activated: bool  # True only on the transition pending -> active


def _upsert_dictionary_entry(
    conn: sqlite3.Connection,
    canonical_word: str,
    source_type: str,
    asr_token: str = None
) -> LearningResult:
    keys = compute_phonetic_keys(canonical_word)
    row = conn.execute(
        "SELECT id, evidence_count, status FROM user_dictionary WHERE canonical_word = ?",
        (canonical_word,),
    ).fetchone()

    activate_immediately = (
        source_type == "explicit_correction"
        and config.EXPLICIT_CORRECTION_ACTIVATES_IMMEDIATELY
    )

    if row is None:
        status = "active" if activate_immediately else "pending"
        cur = conn.execute(
            """INSERT INTO user_dictionary
               (canonical_word, primary_hash, secondary_hash,
                evidence_count, status)
               VALUES (?, ?, ?, 1, ?)""",
            (canonical_word, keys.primary, keys.secondary, status),
        )
        return LearningResult(
            canonical_word=canonical_word,
            dictionary_id=cur.lastrowid,
            evidence_count=1,
            status=status,
            was_new=True,
            was_activated=(status == "active"),
        )

    new_count = row["evidence_count"] + 1
    was_pending = row["status"] == "pending"
    should_activate = activate_immediately or (
        was_pending and new_count >= config.EVIDENCE_THRESHOLD_FOR_ACTIVATION
    )
    new_status = "active" if should_activate else row["status"]

    conn.execute(
        """UPDATE user_dictionary
           SET evidence_count = ?, status = ?,
               last_seen_at = CURRENT_TIMESTAMP
           WHERE id = ?""",
        (new_count, new_status, row["id"]),
    )
    return LearningResult(
        canonical_word=canonical_word,
        dictionary_id=row["id"],
        evidence_count=new_count,
        status=new_status,
        was_new=False,
        was_activated=(was_pending and new_status == "active"),
    )


def _log_evidence(
    conn: sqlite3.Connection,
    dictionary_id: int,
    correction: TokenCorrection,
    source_type: str,
) -> None:
    conn.execute(
        """INSERT INTO memory_evidence
           (dictionary_id, context_snippet, source_type, asr_token)
           VALUES (?, ?, ?, ?)""",
        (dictionary_id, correction.context_snippet, source_type, correction.asr_token),
    )


def ingest_observation(
    conn: sqlite3.Connection, asr_text: str, formatted_text: str
) -> list[LearningResult]:
    """Implicit, low-confidence path: diff two transcripts, learn from
    whatever token-level corrections survive the plausibility gate."""
    corrections = extract_corrections(asr_text, formatted_text)
    results = []
    for c in corrections:
        result = _upsert_dictionary_entry(
            conn, c.corrected_token, "implicit_observation", asr_token=c.asr_token
        )
        _log_evidence(conn, result.dictionary_id, c, "implicit_observation")
        results.append(result)
    return results


def ingest_explicit_correction(
    conn: sqlite3.Connection,
    canonical_word: str,
    context_snippet: str,
    asr_token: str | None = None,
) -> LearningResult:
    """High-confidence path: the user directly declares a correction.
    Activates immediately regardless of prior evidence count."""
    result = _upsert_dictionary_entry(
        conn, canonical_word, "explicit_correction", asr_token=asr_token
    )
    correction = TokenCorrection(
        asr_token=asr_token or canonical_word,
        corrected_token=canonical_word,
        context_snippet=context_snippet,
    )
    _log_evidence(conn, result.dictionary_id, correction, "explicit_correction")
    return result
