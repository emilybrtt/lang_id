# src/preprocessing.py
import re
import numpy as np

ALPHABET = list("abcdefghijklmnopqrstuvwxyz")

def clean(text: str) -> str:
    """Lowercase, keep only a-z, strip the rest."""
    text = text.lower()
    text = re.sub(r"[^a-z]", "", text)  # drop non-letters
    return text

def to_freq_vector(text: str) -> np.ndarray:
    """
    Returns a 26-dim normalized frequency vector.
    Each element = count(letter) / total_letters.
    Zero vector if text is empty.
    """
    cleaned = clean(text)
    n = len(cleaned)
    if n == 0:
        return np.zeros(26)
    vec = np.array([cleaned.count(c) for c in ALPHABET], dtype=float)
    return vec / n   # L1 normalization → sums to 1.0

# Quick sanity check
if __name__ == "__main__":
    v = to_freq_vector("Hello World!")
    print(dict(zip(ALPHABET, v.round(4))))