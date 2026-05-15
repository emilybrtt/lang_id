import os, numpy as np, pandas as pd
from preprocessing import to_freq_vector


def build_reference_vectors(train_dir="data/train") -> pd.DataFrame:
    rows = {}
    for fname in os.listdir(train_dir):
        lang = fname.replace(".txt", "")
        with open(os.path.join(train_dir, fname), encoding="utf-8") as f:
            corpus = f.read()
        # Dividir em chunks de 500 caracteres → calcular média dos vetores
        # Isso evita viés de documento longo e fornece uma estimativa média
        chunks = [corpus[i : i + 500] for i in range(0, len(corpus), 500)]
        vecs = [to_freq_vector(c) for c in chunks if len(c) > 50]
        rows[lang] = np.mean(vecs, axis=0)

    df = pd.DataFrame(rows, index=list("abcdefghijklmnopqrstuvwxyz"))
    df.to_csv("data/reference_vectors.csv")
    return df


if __name__ == "__main__":
    df = build_reference_vectors()
    print(df.T)  # linhas=idiomas, colunas=letras
