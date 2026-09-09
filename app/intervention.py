"""
Lifecycle 2, Intervention Pipeline:
Scans incoming transcripts for active phonetic hashes, retrieves contextual
candidates, constructs the Judge-Editor LLM prompt, applies deterministic
string replacements, and logs the outcome.
"""
import os
import re
import time
import json
import asyncio
import sqlite3
from openai import AsyncOpenAI

from . import retrieval
from .retrieval import Candidate

# Using Groq for lightning-fast free inference
client = AsyncOpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    timeout=5.0,
    max_retries=0
)
MODEL_NAME = "groq/compound-mini"

SYSTEM_PROMPT = """You are a phonetic-entity classifier. You will receive a numbered list of potential mis-transcriptions.
For each item, decide if the highlighted token is a mis-transcription (including capitalization or slight spelling errors) of the candidate entity based on the context window.
  
Rules:
- true: The surrounding context aligns with the SPECIFIC nature of the Candidate entity.
- false: The context implies a COMPLETELY DIFFERENT entity type OR the context uses the literal dictionary meaning.
- If intervene is true, you must provide the grammatically correct form of the candidate entity in the "replacement" field. Apply an 's (possessive) or s (plural) to the candidate entity if the sentence grammar requires it.

Examples:
Text: "ask <<mark>> to join" | Entity: Marc -> {"intervene": true, "replacement": "Marc", "reasoning": "person being asked"}
Text: "leave a <<mark>> here" | Entity: Marc -> {"intervene": false, "replacement": null, "reasoning": "literal physical mark"}
Text: "ask <<jason>> to join" | Entity: JSON -> {"intervene": false, "replacement": null, "reasoning": "context implies human, not data"}
Text: "save in <<jason>> format" | Entity: JSON -> {"intervene": true, "replacement": "JSON", "reasoning": "implies data format"}
Text: "check the <<roses>> logs" | Entity: ROSE -> {"intervene": true, "replacement": "ROSE's", "reasoning": "possessive software system"}
Text: "she bought a <<rose>>" | Entity: ROSE -> {"intervene": false, "replacement": null, "reasoning": "literal flower"}
  
Reply with a strict JSON array of objects, one for each item:
[
  {"id": 0, "intervene": true, "replacement": "Marc", "reasoning": "person taking action"},
  {"id": 1, "intervene": false, "replacement": null, "reasoning": "context implies human, not data"},
  {"id": 2, "intervene": true, "replacement": "JSON", "reasoning": "implies data format"}
]
Output ONLY the JSON array. Do not include markdown blocks or thinking tags."""

def build_batch_prompt(asr_text: str, candidates: list[Candidate]) -> str:
    from .tokenize_utils import _WORD_RE
    token_matches = list(_WORD_RE.finditer(asr_text))
    
    prompt = "Evaluate the following items:\n\n"
    
    for i, c in enumerate(candidates):
        num_tokens = len(_WORD_RE.findall(c.token))
        start_char = token_matches[c.token_index].start()
        end_char = token_matches[c.token_index + num_tokens - 1].end()
        
        # Highlight exactly WHICH occurrence we want the LLM to judge
        highlighted_asr = asr_text[:start_char] + "<<" + asr_text[start_char:end_char] + ">>" + asr_text[end_char:]
        
        # Extract sliding window
        tokens = highlighted_asr.split()
        target_idx = -1
        for j, t in enumerate(tokens):
            if "<<" in t or ">>" in t:
                target_idx = j
                break
                
        if target_idx != -1:
            start = max(0, target_idx - 5)
            end = min(len(tokens), target_idx + 6)
            window = " ".join(tokens[start:end])
        else:
            window = highlighted_asr
            
        context_lines = "\n".join(f"- {ctx}" for ctx in c.context_snippets) or "None"
        
        prompt += f"Item {i}:\n"
        prompt += f"Snippet: ... {window} ...\n"
        prompt += f"Candidate entity: {c.canonical_word}\n"
        prompt += f"Known context: {context_lines}\n\n"
        
    return prompt

async def judge_candidates_batch(asr_text: str, candidates: list[Candidate]) -> list[tuple[Candidate, dict]]:
    if not candidates:
        return []
        
    start_time = time.time()
    prompt = build_batch_prompt(asr_text, candidates)

    try:
        import re
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT}, 
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=1000
        )
        raw_content = response.choices[0].message.content.strip()
        
        # Safely extract the JSON array
        match = re.search(r'\[.*\]', raw_content, re.DOTALL)
        if match:
            json_str = match.group(0)
            results_list = json.loads(json_str)
        else:
            results_list = []
            
        # Map back to candidates
        judgments = []
        for i, c in enumerate(candidates):
            # Find the result for this ID, default to False if not found
            res = next((r for r in results_list if r.get("id") == i), None)
            if res:
                result_dict = {
                    "intervene": res.get("intervene", False), 
                    "reasoning": res.get("reasoning", ""),
                    "replacement": res.get("replacement")
                }
            else:
                result_dict = {"intervene": False, "reasoning": "Batch missing ID"}
                
            result_dict["latency_ms"] = (time.time() - start_time) * 1000 / len(candidates)
            judgments.append((c, result_dict))
            
        return judgments
    except Exception as e:
        # Fallback if the whole batch crashes
        judgments = []
        for c in candidates:
            judgments.append((c, {"intervene": False, "reasoning": f"LLM Error: {str(e)[:50]}", "latency_ms": 0}))
        return judgments


import secrets

def _log_intervention(conn: sqlite3.Connection, request_id: str, asr_text: str, output_text: str, 
                      candidate: Candidate, result: dict):
    conn.execute(
        """INSERT INTO intervention_logs 
           (request_id, input_asr, target_token_index, target_token, candidate_memory_id, candidate_word, 
            did_intervene, llm_reasoning, output_text, execution_latency_ms, token_cost)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (request_id, asr_text, candidate.token_index, candidate.token, candidate.dictionary_id, candidate.canonical_word,
         result.get("intervene", False), result.get("reasoning", ""), 
         output_text, result.get("latency_ms", 0.0), result.get("cost", 0))
    )


async def process_transcript(conn: sqlite3.Connection, asr_text: str) -> str:
    """
    The main hot-path function.
    Finds candidates, concurrently judges them via LLM, applies replacements,
    and returns the finalized text.
    """
    request_id = secrets.token_hex(4)
    
    # 1. Direct Overrides (Zero-latency exact match replacements)
    overrides = conn.execute("SELECT asr_token, canonical_word FROM exact_overrides").fetchall()
    for row in overrides:
        asr_token = row["asr_token"]
        canonical_word = row["canonical_word"]
        # Use regex for whole-word replacement to avoid breaking substrings (e.g. 'cat' in 'category')
        # \b doesn't work well with punctuation sometimes, but it's standard for word boundaries
        asr_text = re.sub(rf'\b{re.escape(asr_token)}\b', canonical_word, asr_text, flags=re.IGNORECASE)
    
    candidates = retrieval.find_candidates(conn, asr_text)
    
    if not candidates:
        return asr_text

    from .tokenize_utils import _WORD_RE
    token_matches = list(_WORD_RE.finditer(asr_text))

    judgments = await judge_candidates_batch(asr_text, candidates)

    output_text = asr_text
    
    # Sort backwards by token index so character slices remain valid as we edit from right-to-left
    # Secondary sort by edit distance (ascending) to prefer perfect matches over sloppy bigrams on overlap.
    # Tertiary sort by token length (descending) to prefer bigrams ONLY if edit distance is tied.
    # Note: We negate edit_dist because reverse=True applies to the entire tuple.
    judgments.sort(key=lambda x: (x[0].token_index, -x[0].edit_dist, len(x[0].token)), reverse=True)
    
    modified_indexes = set()
    
    for candidate, result in judgments:
        did_intervene = result.get("intervene", False)
        
        if did_intervene:
            num_tokens = len(_WORD_RE.findall(candidate.token))
            token_range = set(range(candidate.token_index, candidate.token_index + num_tokens))
            
            # Prevent overlapping replacements (e.g. replacing 'ki' then 'ki wi')
            if token_range & modified_indexes:
                continue
                
            modified_indexes.update(token_range)
            
            # Extract exact character offsets from the original string
            start_char = token_matches[candidate.token_index].start()
            end_char = token_matches[candidate.token_index + num_tokens - 1].end()
            
            # Use the LLM's grammatically correct replacement, fallback to canonical
            replacement_word = result.get("replacement") or candidate.canonical_word
            
            # Apply exact slice replacement
            output_text = output_text[:start_char] + replacement_word + output_text[end_char:]
            
        _log_intervention(conn, request_id, asr_text, output_text, candidate, result)

    conn.commit()
    return output_text
