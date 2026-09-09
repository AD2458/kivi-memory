"""
Align two transcripts (raw ASR vs. a corrected version -- either the
formatted-output proxy or an explicit user correction) and isolate
token-level substitutions that look like phonetic/spelling corrections
rather than unrelated grammar or word-choice changes.

Word-level alignment via difflib.SequenceMatcher. Extracts 'replace' opcodes,
supporting multi-word token-splitting corrections (e.g., "ki wi" -> "Kivi")
by collapsing whitespace during the phonetic plausibility check.
"""
from dataclasses import dataclass
from difflib import SequenceMatcher

from .tokenize_utils import tokenize, normalize
from . import config


@dataclass(frozen=True)
class TokenCorrection:
    asr_token: str
    corrected_token: str
    context_snippet: str  # the corrected sentence, for evidence storage


def extract_corrections(asr_text: str, corrected_text: str) -> list[TokenCorrection]:
    asr_tokens = tokenize(asr_text)
    corrected_tokens = tokenize(corrected_text)

    asr_norm = [normalize(t) for t in asr_tokens]
    corrected_norm = [normalize(t) for t in corrected_tokens]

    matcher = SequenceMatcher(a=asr_norm, b=corrected_norm, autojunk=False)
    results: list[TokenCorrection] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":
            continue
        
        asr_span = asr_tokens[i1:i2]
        corrected_span = corrected_tokens[j1:j2]

        if len(asr_span) == len(corrected_span):
            # Equal token counts (N→N): learn each word individually.
            # e.g. "aditya shatriya" → "Aaditya Kshatriya" becomes
            #       aditya→Aaditya AND shatriya→Kshatriya
            for asr_word, corrected_word in zip(asr_span, corrected_span):
                if len(normalize(corrected_word)) < config.MIN_TOKEN_LENGTH_FOR_MEMORY:
                    continue
                # Skip if they are actually the same word (just casing matched during normalize)
                if normalize(asr_word) == normalize(corrected_word):
                    continue
                results.append(
                    TokenCorrection(
                        asr_token=asr_word,
                        corrected_token=corrected_word,
                        context_snippet=corrected_text,
                    )
                )
        else:
            # Unequal token counts (N→M): genuine token-split correction.
            # e.g. "ki wi" → "Kivi" (2→1), must stay as a single entry.
            asr_tok = " ".join(asr_span)
            corrected_tok = " ".join(corrected_span)
            corrected_collapsed = corrected_tok.replace(" ", "")

            if len(normalize(corrected_collapsed)) < config.MIN_TOKEN_LENGTH_FOR_MEMORY:
                continue

            results.append(
                TokenCorrection(
                    asr_token=asr_tok,
                    corrected_token=corrected_tok,
                    context_snippet=corrected_text,
                )
            )

    return results
