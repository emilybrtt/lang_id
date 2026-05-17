import os

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from .classifier import LanguageClassifier


def load_test_set(test_dir="data/test") -> list[tuple[str, str]]:
    import os

    samples = []
    for fname in os.listdir(test_dir):
        lang = fname.replace(".txt", "")
        with open(os.path.join(test_dir, fname), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append((line, lang))
    return samples


def evaluate(clf: LanguageClassifier):
    samples = load_test_set()
    texts, truths = zip(*samples)
    preds_cos = [clf.predict(t, "cosine") for t in texts]
    preds_euc = [clf.predict(t, "euclidean") for t in texts]

    print(f"Cosine accuracy:    {accuracy_score(truths, preds_cos):.3f}")
    print(f"Euclidean accuracy: {accuracy_score(truths, preds_euc):.3f}")

    # Matriz de confusão para cosseno
    langs = sorted(set(truths))
    cm = confusion_matrix(truths, preds_cos, labels=langs)
    disp = ConfusionMatrixDisplay(cm, display_labels=langs)
    disp.plot(cmap="Blues")
    plt.tight_layout()
    plt.savefig("report/confusion_matrix.pdf")
    plt.show()


def accuracy_vs_length(clf: LanguageClassifier, test_dir="data/test"):
    samples = load_test_set(test_dir)
    lengths = [10, 20, 50, 100, 200, 500]
    accs = []
    for L in lengths:
        truncated = [(t[:L], g) for t, g in samples if len(t) >= L]
        if not truncated:
            continue
        texts, truths = zip(*truncated)
        preds = [clf.predict(t) for t in texts]
        accs.append(accuracy_score(truths, preds))

    plt.figure(figsize=(6, 3))
    plt.plot(lengths[: len(accs)], accs, marker="o")
    plt.xlabel("Text length (characters)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs input length")
    plt.tight_layout()
    plt.savefig("report/accuracy_vs_length.pdf")
    plt.show()


if __name__ == "__main__":
    clf = LanguageClassifier()
    evaluate(clf)
    accuracy_vs_length(clf)
