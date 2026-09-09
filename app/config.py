"""Tunable constants. Kept in one place so the eval harness can reference
the exact same values the pipeline uses (no drift between what we test
and what we ship)."""

# Evidence-gating: how many independent implicit observations before a
# 'pending' memory is promoted to 'active' and allowed to actually
# intervene in output.
EVIDENCE_THRESHOLD_FOR_ACTIVATION = 2

# An explicit user correction (as opposed to an inferred ASR-vs-formatted
# discrepancy) is trusted immediately -- the user directly told us the
# right spelling, there's no ambiguity to accumulate evidence against.
EXPLICIT_CORRECTION_ACTIVATES_IMMEDIATELY = True
# Maximum character-level edit distance (after Indian-English normalization)
# between an ASR token and a canonical word. This is a simple absolute cap
# to prevent short phonetic hashes from matching half the dictionary.
# Valid corrections are typically 0-2 edits; garbage matches are 5+.
MAX_CHAR_EDIT_DISTANCE = 3
MAX_EDIT_DISTANCE_RATIO = 0.5

# Words too short are noisy for phonetic hashing and too common to be
# personal entities worth memorizing.
MIN_TOKEN_LENGTH_FOR_MEMORY = 3

# Basic stopword set for tokenization during retrieval (not exhaustive by
# design -- retrieval only needs to skip tokens that would never be a
# personal entity anyway; over-filtering risks skipping real candidates).
STOPWORDS = {
    "a", "an", "the", "to", "of", "in", "on", "at", "for", "and", "or",
    "is", "are", "was", "were", "be", "been", "it", "this", "that",
    "with", "as", "by", "from", "please", "can", "you", "i", "we",
}

# How many recent context snippets to attach when presenting a candidate
# to the Judge-Editor / to the reviewer UI.
MAX_CONTEXT_SNIPPETS = 3
