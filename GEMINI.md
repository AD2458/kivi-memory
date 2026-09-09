Here is the detailed `GEMINI.md` file capturing our complete architectural design, technical decisions, and constraints for the Kivi Backend-Focused Full Stack Task.

# Kivi Phonetic Memory System - Architecture & Implementation Design

## 1. Overall Goal of the Project

To design and build a narrowly scoped, word-level "phonetic memory" system that acts as an intermediary between raw ASR output and final formatted text. The system must learn personalized vocabulary (entities, names, specific terms) through ordinary user corrections and reliably apply them to future ASR outputs based on acoustic similarity and semantic context. The primary mandate is to build the "smallest [system] that makes Kivi feel as though it has met this person before," deliberately avoiding overengineering.

## 2. Exact System Architecture Designed

The system implements an **Evidence-Gated Phonetic Retrieval-Augmented Generation (RAG)** pipeline.
It relies on deterministic phonetic hashing for instantaneous database lookups (bypassing semantic vector databases entirely) and utilizes a localized "Judge-Editor" LLM pattern strictly for binary semantic validation of retrieved candidates.

## 3. Components, Modules, and Responsibilities

* **Demonstration UI (Streamlit):** Provides a local, interactive surface to manually inject observations (learning), process new transcripts, inspect the internal database state, and view intervention trace logs.
* **Learning Engine (FastAPI Background):** Ingests raw ASR and corrected output, executes character-level diffing to extract target word pairs, calculates string distances, and updates the database state machine (pending vs. active).
* **Phonetic Hashing & Pre-Processing Module:** Normalizes transcripts for Indian-English acoustic quirks, then generates primary and secondary acoustic hashes using the Double Metaphone algorithm.
* **Relational Memory Store (SQLite):** Stores canonical words, their phonetic hashes, historical sentence context, and evidence counts in a lightweight local file.
* **Intervention Pipeline (FastAPI Sync):** Scans incoming transcripts for active phonetic hashes, retrieves contextual candidates, constructs the Judge-Editor LLM prompt, applies deterministic string replacements, and logs the outcome.
* **Evaluation Harness (Python Script):** An automated pipeline that runs the `seed_data.json` test cases, calculates precision/recall metrics, and generates a formatted Markdown report.

## 4. Data Flow Between Components

**Lifecycle 1: Observation & Learning (Async)**

1. System receives a raw ASR transcript and a user-corrected transcript.
2. `difflib.SequenceMatcher` performs a character-level diff to isolate the altered substring mapping (e.g., `"ki wi"` -> `"Kivi"`).
3. Levenshtein distance validates that the change is an acoustic/typographical correction, not a full semantic rewrite.
4. The Pre-Processor normalizes characters (e.g., `w` -> `v`), and the Double Metaphone algorithm generates the phonetic hash.
5. The system upserts the mapping to the `user_dictionary` table. If new, `status='pending'` and `evidence_count=1`. If existing, it increments `evidence_count`. If `evidence_count` hits the threshold (e.g., 2), `status` upgrades to `'active'`.
6. The surrounding sentence is saved to `memory_evidence` for future LLM context.

**Lifecycle 2: Processing & Intervention (Sync)**

1. System receives a new raw ASR transcript.
2. Retrieval queries the SQLite database for all *active* phonetic hashes for the user. A fast substring/Aho-Corasick search flags matches in the transcript.
3. For a match, the database returns the canonical word and its recent `memory_evidence` snippets.
4. The LLM (Judge-Editor) receives the ASR sentence, the candidate word, and historical context. It returns a strict JSON: `{"intervene": boolean, "reasoning": "..."}`.
5. If `true`, backend performs a deterministic string replacement. If `false`, it leaves the text alone.
6. The outcome, LLM reasoning, and latency are committed to `intervention_logs`.

## 5. Algorithms and Approaches

* **Double Metaphone:** Generates two distinct phonetic codes (primary and secondary) per word, making it highly resilient to non-Western names and varying ASR pronunciations (e.g., "Jose" yielding both English `JS` and Spanish `HS`).
* **Character-Level Diffing (Gestalt Pattern Matching):** Uses Python's `difflib.SequenceMatcher` during the learning phase to handle ASR token-splitting errors (mapping `"ki wi"` to `"Kivi"` seamlessly).
* **Damerau-Levenshtein Distance:** Validates acoustic similarity by counting character edits (including transpositions) between raw ASR and corrected terms.
* **Indian-English Pre-Processing Heuristic:** A lightweight regex step before phonetic hashing that normalizes common ASR confusion for Indian accents (e.g., `w`->`v`, `aa`->`a`, `ee`->`i`) to ensure words like "Aaditya" and "aditya" hash identically.
* **State Machine Evidence Accumulation:** Requires multiple observations before promoting a memory from `pending` to `active`, preventing the system from permanently learning ASR hallucinations.
* **Judge-Editor LLM Pattern:** Constrains the LLM to a binary decision based on historical context, preventing standard generative hallucinations and preserving the user's surrounding transcript.

## 6. APIs, Models, Databases, and Tools

* **Backend:** Python 3.11+, FastAPI (for async performance during LLM calls).
* **Database:** SQLite in WAL (Write-Ahead Logging) mode.
* **Phonetics:** `doublemetaphone` or `Fuzzy` package (specifically for Double Metaphone, explicitly *not* `jellyfish`'s standard Metaphone).
* **String Matching:** `jellyfish` (strictly for rapid Levenshtein/Damerau-Levenshtein calculations).
* **LLM:** OpenAI API (`gpt-4o-mini` with structured JSON outputs).
* **Frontend:** Streamlit.

## 7. Database Schema

* **`user_dictionary`**: `id`, `canonical_word`, `phonetic_hash` (Indexed), `evidence_count`, `confidence_score` (Bounded 0.0-1.0 for time decay/conflict resolution), `status` (`pending`, `active`), `created_at`, `last_seen_at`.
* **`memory_evidence`**: `id`, `dictionary_id` (FK), `context_snippet`, `source_type`, `created_at`.
* **`intervention_logs`**: `id`, `input_asr`, `target_token`, `candidate_memory`, `did_intervene`, `llm_reasoning`, `execution_latency_ms`, `token_cost`, `timestamp`.

## 8. Important Implementation Decisions & Reasoning

* **No Vector Databases:** Semantic embeddings cluster by meaning, not sound. They fail to map "kiwi" to "Kivi". Exact SQL phonetic hash lookups are $O(1)$, deterministic, and logically superior for acoustic errors.
* **Async LLM Calls in Hot Path:** To minimize latency blocking, if a transcript triggers multiple phonetic matches, FastAPI's `asyncio.gather()` must be used to execute the LLM Judge validations concurrently.
* **Confidence Score Mechanics:** Provides a mathematical bound (0.0-1.0) to facilitate "forgetting" (decaying scores based on `last_seen_at`) and resolving conflicts between two active memories with identical phonetic hashes.
* **Frictionless Learning:** The system extracts observations silently via background diffing rather than requiring the user to fill out a static dictionary form.

## 9. Constraints and Explicit "Do NOT Do" Decisions

* **NO Vector DBs (Chroma/FAISS):** Overkill and mathematically incorrect for acoustic mapping.
* **NO Heavy Agent Frameworks (LangChain/Letta/Mem0):** Abstracts away control flow, causes prompt bloat, and makes it impossible to debug false interventions deterministically.
* **NO Docker or Cloud Databases (Supabase/Postgres):** The evaluation requires the reviewing agent to run the code easily. Local Python + SQLite is explicitly preferred for absolute reproducibility.




## 10. Planned Experiments & Evaluation Methodology

The evaluation script must process `seed_data.json` and measure:

* **Metrics:** Intervention Precision (True Positives / Total Interventions), Intervention Recall, False Intervention Rate (FIR - must be heavily penalized), Named Entity Word Error Rate (NE-WER), P50/P95 Latency, Token Cost, Database Growth.


* **Test Case Categories:**
* *True Positive:* Correctly replacing "ask aditya" -> "Ask Aaditya".
* *True Negative (Context Mismatch):* Leaving "I ate a kiwi" alone despite phonetic match to "Kivi".
* *Pending Memory:* Ignoring a phonetic match because `evidence_count` has not met the threshold.
* *Conflicting Memories:* Disambiguating two similar phonetic matches based on semantic context snippets.
* *Novel Entities:* Refusing to hallucinate a correction for a completely unseen proper noun.