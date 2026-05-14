import re
import numpy as np

ALPHABET = list("abcdefghijklmnopqrstuvwxyz")


def clean(text: str) -> str:
    text = text.lower() # minuscula
    text = re.sub(r"[^a-z]", "", text)  # só o q for letra a-z
    return text


def to_freq_vector(text: str) -> np.ndarray:
    cleaned = clean(text)
    n = len(cleaned)
    if n == 0:
        return np.zeros(26)
    vec = np.array([cleaned.count(c) for c in ALPHABET], dtype=float)
    return vec / n  # normaliza


# check pra ver se tá funcionando
if __name__ == "__main__":
    v = to_freq_vector("Hello World!")
    print(dict(zip(ALPHABET, v.round(4))))
