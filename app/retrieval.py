"""
Lifecycle 2, retrieval half: given a new incoming ASR transcript, find
which of its tokens have a plausible active-memory replacement.

Candidate generation performs a fuzzy phonetic hash lookup against the
active dictionary, then passes survivors directly to the LLM Judge.
No magic edit-distance ratios — the LLM is the precision layer.
"""
import sqlite3
from dataclasses import dataclass

from . import config
from .phonetic import edit_distance, compute_phonetic_keys, squash_repeats, phonetic_spelling_normalize
from .tokenize_utils import tokenize, normalize, punctuation_boundaries


@dataclass
class Candidate:
    token: str                 # the raw ASR token as it appeared
    token_index: int           # position in the tokenized ASR text
    dictionary_id: int
    canonical_word: str
    evidence_count: int
    edit_dist: int
    context_snippets: list[str]


def _fetch_context_snippets(conn: sqlite3.Connection, dictionary_id: int) -> list[str]:
    rows = conn.execute(
        """SELECT context_snippet FROM memory_evidence
           WHERE dictionary_id = ?
           ORDER BY created_at DESC LIMIT ?""",
        (dictionary_id, config.MAX_CONTEXT_SNIPPETS),
    ).fetchall()
    return [r["context_snippet"] for r in rows]


def find_candidates(conn: sqlite3.Connection, asr_text: str, max_ngram: int = 2) -> list[Candidate]:
    tokens = tokenize(asr_text)
    candidates: list[Candidate] = []
    punct_bounds = punctuation_boundaries(asr_text)
    
    # Pre-load the active dictionary to do robust in-memory fuzzy retrieval
    active_dict = conn.execute(
        "SELECT id, canonical_word, primary_hash, secondary_hash, evidence_count "
        "FROM user_dictionary WHERE status = 'active'"
    ).fetchall()

    for window_size in range(1, max_ngram + 1):
        for idx in range(len(tokens) - window_size + 1):
            # Never form bigrams across punctuation boundaries
            if window_size > 1:
                spans_punctuation = any(
                    (idx + k) in punct_bounds
                    for k in range(window_size - 1)
                )
                if spans_punctuation:
                    continue

            ngram_tokens = tokens[idx:idx + window_size]
            raw_token_str = " ".join(ngram_tokens)
            collapsed_str = raw_token_str.replace(" ", "")

            # Only strictly filter stopwords on unigrams. 
            if window_size == 1 and normalize(raw_token_str) in config.STOPWORDS:
                continue
                
            if len(collapsed_str) < config.MIN_TOKEN_LENGTH_FOR_MEMORY:
                continue

            keys = compute_phonetic_keys(collapsed_str)
            search_hashes = [h for h in (keys.primary, keys.secondary) if h]
            if not search_hashes:
                continue
                
            seen_dict_ids = set()
            
            for row in active_dict:
                dict_id = row["id"]
                if dict_id in seen_dict_ids:
                    continue
                    
                canonical = row["canonical_word"]
                if canonical == raw_token_str:
                    continue
                    
                # Fuzzy hash check: Edit distance between phonetic codes <= 2
                row_hashes = [h for h in (row["primary_hash"], row["secondary_hash"]) if h]
                hash_match = False
                for sh in search_hashes:
                    for rh in row_hashes:
                        if edit_distance(sh, rh) <= 2:
                            hash_match = True
                            break
                    if hash_match:
                        break
                        
                if not hash_match:
                    continue
                
                # Character distance checks (on ghost squashed strings)
                norm_str = squash_repeats(phonetic_spelling_normalize(normalize(collapsed_str)))
                norm_canonical = squash_repeats(phonetic_spelling_normalize(normalize(canonical.replace(" ", ""))))
                char_dist = edit_distance(norm_str, norm_canonical)
                
                # Reject if it exceeds the absolute cap OR the relative ratio
                if char_dist > config.MAX_CHAR_EDIT_DISTANCE:
                    continue
                longer_len = max(len(norm_str), len(norm_canonical))
                if longer_len > 0 and (char_dist / longer_len) > config.MAX_EDIT_DISTANCE_RATIO:
                    continue
                    
                seen_dict_ids.add(dict_id)

                candidates.append(
                    Candidate(
                        token=raw_token_str,
                        token_index=idx,
                        dictionary_id=dict_id,
                        canonical_word=canonical,
                        evidence_count=row["evidence_count"],
                        edit_dist=edit_distance(collapsed_str, canonical.replace(" ", "")),
                        context_snippets=_fetch_context_snippets(conn, dict_id),
                    )
                )

    return candidates
