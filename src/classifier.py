import numpy as np, pandas as pd
from preprocessing import to_freq_vector


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """cos(θ) = (a · b) / (‖a‖ ‖b‖)"""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


class LanguageClassifier:
    def __init__(self, ref_path="data/reference_vectors.csv"):
        df = pd.read_csv(ref_path, index_col=0)
        # chaves = códigos de idioma, valores = arrays numpy 26-dim
        self.refs = {col: df[col].values for col in df.columns}

    def classify(self, text: str, metric="cosine") -> dict:
        """Retorna dict: {lang: score}, ordenado melhor primeiro."""
        v = to_freq_vector(text)
        scores = {}
        for lang, ref in self.refs.items():
            if metric == "cosine":
                scores[lang] = cosine_similarity(v, ref)
            else:
                scores[lang] = -euclidean_distance(v, ref)  # negativo → maior=melhor
        return dict(sorted(scores.items(), key=lambda x: -x[1]))

    def predict(self, text: str, metric="cosine") -> str:
        return next(iter(self.classify(text, metric)))


if __name__ == "__main__":
    clf = LanguageClassifier()
    samples = [
        ("O gato dormia sobre o tapete.", "pt"),
        ("The quick brown fox jumps.", "en"),
        ("El sol brilla con fuerza.", "es"),
    ]
    for text, truth in samples:
        pred_cos = clf.predict(text, "cosine")
        pred_euc = clf.predict(text, "euclidean")
        print(f"{truth} | cos→{pred_cos} | euc→{pred_euc} | '{text[:30]}'")
