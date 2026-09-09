# Kivi Phonetic Memory System

An **Evidence-Gated Phonetic Retrieval-Augmented Generation (RAG)** pipeline designed to act as an intermediary between raw ASR (Speech-to-Text) output and final formatted text. 

The system learns personalized vocabulary (entities, names, specific terms) through ordinary user corrections and reliably applies them to future ASR outputs based on acoustic similarity and semantic context. It deliberately avoids heavy vector databases, instead relying on lightning-fast phonetic math and a localized LLM Judge.

---

## 🏗️ Core Architecture

The system is built on a ruthless minimalist philosophy: **No Vector DBs, No Heavy Agent Frameworks.**

1. **Direct Overrides (Zero-Latency):** 
   A hardcoded alias table that bypasses the phonetic engine entirely. If the ASR reads "keevee", it instantly converts it to "Kivi" before the rest of the sentence is processed.
2. **Phonetic Hashing (Double Metaphone):** 
   The system generates acoustic codes for words. This allows for instantaneous, deterministic $O(1)$ database lookups for words that *sound* exactly the same.
3. **The LLM Judge-Editor:** 
   Matches that pass the exact phonetic hash are sent to a lightweight LLM (Groq `compound-mini`). The LLM acts purely as a binary semantic judge: it checks if the surrounding sentence context matches the specific nature of the candidate entity.
4. **Batched Prompting:** 
   If a single sentence contains multiple phonetic matches, the system stacks them into a single, numbered prompt. This drastically reduces API latency, avoids rate limits, and processes the entire sentence in a single LLM round-trip.
5. **Mathematical Security (Tie-Breaker):** 
   If the LLM approves multiple competing candidates (e.g. "Aditi" vs "Aaditya"), the system resolves the tie using raw Damerau-Levenshtein edit distance, locking in the closest spelling.

---

## 🗄️ System Components (Streamlit UI)

* **🎙️ Process ASR:** The hot-path testing ground. Paste raw transcripts to see the engine detect, judge, and replace phonetic mis-transcriptions in real-time.
* **🧠 Learning Engine:** 
  * *Implicit Observation:* Learns passively when a user corrects a transcript.
  * *Explicit Correction:* Manually force a phonetic entity into the dictionary.
  * *Direct Overrides:* Hardcode specific zero-latency ASR string replacements.
* **📚 User Dictionary:** The internal SQLite state machine. Displays all active and pending phonetic memories. Includes a multi-select search and delete tool for easy pruning.
* **⚙️ Intervention Logs:** A complete audit trail. Every evaluated word is logged with a unique 8-character `request_id`, tracking the LLM's exact reasoning, the token index, API latency, and whether it intervened.

---

## 🧠 Advanced Edge-Case Handling

### 1. Entity Clashes (The Tie-Breaker)
If the user says `"aditi"`, the phonetic hash (`ATT`) might pull both `"Aditi"` and `"Aaditya"` from the database. The system sends both to the LLM. If the LLM approves both, the Python backend sorts the approvals by **Edit Distance**. Since `"aditi"` is 0 edits away from `"Aditi"` but 3 edits away from `"Aaditya"`, the system applies `"Aditi"` and safely locks the index, dropping the loser.

### 2. Math Ghosts (On-The-Fly Normalization)
Instead of permanently mutating the dictionary with heavy linguistic heuristics, the system uses "Math Ghosts". Right before checking edit distance, it safely transforms both the ASR word and the database word by replacing `ee->i`, `oo->u`, `ph->f`, and squashing all consecutive repeating characters (e.g. `keevee` -> `kivi`). This allows the math check to easily pass complex ASR typos, while the LLM and the final output still receive the perfectly intact, original tokens.

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
