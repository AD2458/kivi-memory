-- Kivi phonetic memory schema.
-- Deviates from the drafted report in one place: user_dictionary stores
-- THREE phonetic codes per word (metaphone / soundex / match_rating_codex)
-- instead of a single "Double Metaphone" hash, because no single jellyfish
-- algorithm reliably collapses acoustically-similar pairs (verified against
-- the assignment's own "Kivi"/"kiwi" example). Retrieval matches on ANY of
-- the three, then edit distance decides plausibility.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS user_dictionary (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_word      TEXT NOT NULL,
    primary_hash        TEXT NOT NULL,
    secondary_hash      TEXT NOT NULL,
    evidence_count      INTEGER NOT NULL DEFAULT 1,
    status              TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'active', 'rejected')),
    created_at          DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_seen_at        DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    -- one canonical spelling per user per dictionary; the same *sound* can
    -- map to multiple canonical spellings (that's how we model the
    -- Aditi / Aditya conflicting-memory case), so we only dedupe on the
    -- exact spelling, not the phonetic key.
    UNIQUE (canonical_word)
);

CREATE INDEX IF NOT EXISTS idx_primary_hash ON user_dictionary(primary_hash);
CREATE INDEX IF NOT EXISTS idx_secondary_hash ON user_dictionary(secondary_hash);
CREATE INDEX IF NOT EXISTS idx_status    ON user_dictionary(status);

CREATE TABLE IF NOT EXISTS memory_evidence (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    dictionary_id   INTEGER NOT NULL,
    context_snippet TEXT NOT NULL,
    source_type     TEXT NOT NULL
                    CHECK (source_type IN ('explicit_correction', 'implicit_observation')),
    asr_token       TEXT NOT NULL,   -- the raw token this evidence was extracted from
    created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dictionary_id) REFERENCES user_dictionary(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_evidence_dict ON memory_evidence(dictionary_id);

CREATE TABLE IF NOT EXISTS intervention_logs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id          TEXT NOT NULL,
    input_asr           TEXT NOT NULL,
    target_token_index  INTEGER,
    target_token        TEXT,
    candidate_memory_id INTEGER,
    candidate_word       TEXT,
    did_intervene        BOOLEAN,
    llm_reasoning        TEXT,
    output_text          TEXT,
    execution_latency_ms REAL,
    token_cost           INTEGER,
    timestamp             DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (candidate_memory_id) REFERENCES user_dictionary(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON intervention_logs(timestamp);

CREATE TABLE IF NOT EXISTS exact_overrides (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asr_token TEXT UNIQUE NOT NULL,
    canonical_word TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

