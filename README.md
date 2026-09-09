# Kivi Phonetic Memory System

An **Evidence-Gated Phonetic Retrieval-Augmented Generation (RAG)** pipeline designed to act as an intermediary between raw ASR (Speech-to-Text) output and final formatted text. 

The system learns personalized vocabulary (entities, names, specific terms) through ordinary user corrections and reliably applies them to future ASR outputs based on acoustic similarity and semantic context. It deliberately avoids heavy vector databases, instead relying on lightning-fast phonetic math and a localized LLM Judge.

---

## 🏗️ Core Architecture

The system is built on a ruthless minimalist philosophy: **No Vector DBs, No Heavy Agent Frameworks.**

1. **Phonetic Hashing (Double Metaphone):** 
   Instead of using semantic embeddings (which cluster by meaning and fail to map "kiwi" to "Kivi"), the system generates acoustic codes for words. This allows for instantaneous, deterministic $O(1)$ database lookups for words that *sound* the same.
2. **Mathematical Security (Damerau-Levenshtein):** 
   To prevent short phonetic hashes from hallucinating wild matches, the system calculates the raw edit distance between the spoken word and the database entity. If the distance is too high (e.g., > 3 edits), it is instantly rejected before the LLM ever sees it.
3. **The LLM Judge-Editor:** 
   Matches that pass the phonetic and edit-distance filters are sent to a lightweight LLM (Groq `compound-mini`). The LLM acts purely as a binary semantic judge: it checks if the surrounding sentence context matches the specific nature of the candidate entity (e.g., distinguishing "Jason" the human from "JSON" the data format).
4. **Batched Prompting:** 
   If a single sentence contains multiple phonetic matches, the system stacks them into a single, numbered prompt. This drastically reduces API latency, avoids rate limits, and processes the entire sentence in a single LLM round-trip.

---

## 🗄️ System Components (Streamlit UI)

* **🎙️ Process ASR:** The hot-path testing ground. Paste raw transcripts to see the engine detect, judge, and replace phonetic mis-transcriptions in real-time.
* **🧠 Learning Engine:** 
  * *Implicit Observation:* Learns passively when a user corrects a transcript. Pushes the word to the database in a `pending` state until it sees the correction happen again.
  * *Explicit Correction:* A manual override to forcefully inject a known entity directly into the `active` state (e.g., onboarding a new coworker's name).
* **📚 User Dictionary:** The internal SQLite state machine. Displays all active and pending phonetic memories. Includes a multi-select search and delete tool for easy pruning.
* **⚙️ Intervention Logs:** A complete audit trail. Every evaluated word is logged with a unique 8-character `request_id`, tracking the LLM's exact reasoning, the token index, API latency, and whether it intervened.

---

## 🧠 Advanced Edge-Case Handling

### 1. Entity Clashes (The Tie-Breaker)
If the user says `"aditi"`, the phonetic hash (`ATT`) might pull both `"Aditi"` and `"Aaditya"` from the database. The system sends both to the LLM. If the LLM approves both, the Python backend sorts the approvals by **Edit Distance**. Since `"aditi"` is 0 edits away from `"Aditi"` but 3 edits away from `"Aaditya"`, the system applies `"Aditi"` and safely locks the index, dropping the loser.

### 2. Capitalization & Grammar Awareness
The LLM is strictly instructed to dynamically adapt the candidate to the sentence's grammar. If the database holds `"Kivi"` but the transcript says `"the kiwis servers"`, the LLM will output `"Kivi's"` to preserve possession, preventing rigid string-replacement errors.

### 3. Pure Math > Heuristics
The system relies entirely on the raw Double Metaphone algorithm and Levenshtein edit distance. We explicitly stripped out "Indian-English" regex normalizers (like forcing `ee` to `i`) because they artificially crushed distinct words together. The pure math is natively capable of handling vowel transliterations and `v/w` swaps flawlessly.

---

## 🚀 Setup & Execution

**Prerequisites:** Python 3.11+

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Environment Variables:**
   Ensure you have your LLM API key configured (e.g., `GROQ_API_KEY`) in your `.env` or system environment.
3. **Run the Dashboard:**
   ```bash
   streamlit run app_ui.py
   ```
4. **Database:** 
   The system will automatically initialize `kivi.db` locally using Write-Ahead Logging (WAL) for safe, concurrent async operations.
