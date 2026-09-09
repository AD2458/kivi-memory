import re

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_PUNCT_RE = re.compile(r"[,.\?!;:\-]")


def tokenize(text: str) -> list[str]:
    """Extract word tokens, stripping punctuation. Keeps original casing
    (callers decide when to lowercase) since casing itself is sometimes
    part of what a memory should restore, e.g. proper nouns."""
    return _WORD_RE.findall(text)


def punctuation_boundaries(text: str) -> set[int]:
    """Return the set of token indices where a punctuation mark appears
    between token[i] and token[i+1]. A bigram should never be formed
    across these boundaries."""
    tokens = list(_WORD_RE.finditer(text))
    boundaries = set()
    for i in range(len(tokens) - 1):
        gap = text[tokens[i].end():tokens[i + 1].start()]
        if _PUNCT_RE.search(gap):
            boundaries.add(i)
    return boundaries


def normalize(word: str) -> str:
    return word.lower().strip()
