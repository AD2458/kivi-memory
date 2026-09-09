import re

asr_text = "Did aditya check the logs? If it's a kiwi-related issue, aditya needs to fix kiwi, not the frontend."
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
token_matches = list(_WORD_RE.finditer(asr_text))

# Mocking the LLM judgments (replacing both aditya's and both kiwi's)
judgments = [
    # (token_index, num_tokens, new_word)
    (1, 1, "Aaditya"), # first aditya
    (8, 1, "kivi"),    # first kiwi (wait, user said "fix kivi the frontend". so 2nd kiwi got replaced?)
    (11, 1, "Aaditya"), # second aditya
    (15, 1, "kivi")     # second kiwi
]

# Sort backwards
judgments.sort(key=lambda x: (x[0], 1), reverse=True)

output_text = asr_text
modified_indexes = set()

for idx, num_tokens, canonical in judgments:
    token_range = set(range(idx, idx + num_tokens))
    if token_range & modified_indexes:
        continue
    modified_indexes.update(token_range)
    
    start_char = token_matches[idx].start()
    end_char = token_matches[idx + num_tokens - 1].end()
    
    print(f"Replacing at {start_char}:{end_char} -> {canonical}")
    output_text = output_text[:start_char] + canonical + output_text[end_char:]
    print(f"Current: {output_text}")

print("FINAL:", output_text)
