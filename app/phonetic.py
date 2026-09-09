"""
Phonetic encoding and distance utilities.

Uses Double Metaphone to generate primary and secondary phonetic codes.
"""
import re
from dataclasses import dataclass
import jellyfish
from doublemetaphone import doublemetaphone


def squash_repeats(word: str) -> str:
    """Squash consecutive identical characters to single characters (e.g. keevee -> keve)."""
    return re.sub(r'(.)\1+', r'\1', word.lower())

def phonetic_spelling_normalize(word: str) -> str:
    """Targeted fixes for common ASR phonetic misspellings before edit distance math."""
    w = word.lower()
    w = w.replace('ee', 'i')
    w = w.replace('oo', 'u')
    w = w.replace('ph', 'f')
    return w


@dataclass(frozen=True)
class PhoneticKeys:
    primary: str
    secondary: str

    def as_tuple(self) -> tuple[str, str]:
        return (self.primary, self.secondary)


def compute_phonetic_keys(word: str) -> PhoneticKeys:
    """Compute the set of phonetic codes we index a word under."""
    w = word.strip().lower()
    primary, secondary = doublemetaphone(w)
    return PhoneticKeys(
        primary=primary or "",
        secondary=secondary or "",
    )


def edit_distance(a: str, b: str) -> int:
    """Damerau-Levenshtein distance (handles adjacent transpositions,
    e.g. 'aditya' <-> 'aidtya', which plain Levenshtein overweights)."""
    return jellyfish.damerau_levenshtein_distance(a.lower(), b.lower())


def phonetic_similarity_score(a: str, b: str) -> float:
    """
    0.0 (no relation) .. 1.0 (identical). Cheap secondary signal used to
    rank multiple candidates, not to gate a decision on its own.
    """
    return jellyfish.jaro_winkler_similarity(a.lower(), b.lower())


def is_plausible_phonetic_match(
    token: str,
    canonical: str,
    max_edit_distance_ratio: float = 0.4,
    max_absolute_distance: int = 3,
) -> bool:
    """
    The correctness gate for 'is this ASR token plausibly a mangling of
    this canonical dictionary word'. Two bounds are applied together:

    - relative: edit distance can't exceed `max_edit_distance_ratio` of the
      longer word's length (so short words need near-exact matches, long
      words tolerate a bit more drift).
    - absolute: a hard ceiling regardless of length, so we never treat a
      12-character coincidence as phonetically related just because the
      ratio math allows it.

    This is intentionally conservative: false negatives (missing a
    correction) are cheap, false interventions (mangling a word the user
    meant) are expensive and erode trust -- matches the brief's emphasis
    on precision over recall.
    """
    if not token or not canonical:
        return False
        
    norm_token = token.lower()
    norm_canonical = canonical.lower()
    
    dist = edit_distance(norm_token, norm_canonical)
    if dist == 0:
        return True
    if dist > max_absolute_distance:
        return False
    longer = max(len(norm_token), len(norm_canonical))
    if longer == 0:
        return False
    return (dist / longer) <= max_edit_distance_ratio


def keys_overlap(a: PhoneticKeys, b: PhoneticKeys) -> bool:
    """True if any of the phonetic codes agree between two words."""
    a_keys = {a.primary, a.secondary} - {""}
    b_keys = {b.primary, b.secondary} - {""}
    return bool(a_keys & b_keys)
