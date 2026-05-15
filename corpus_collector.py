import requests
import time
import os
import re

LANGUAGES = ["pt", "en", "es", "de", "fr", "it"]
OUTPUT_DIR = "data/train"
TARGET_CHARS = 300_000  # per language (reduce if slow)

HEADERS = {
    "User-Agent": "ProjetoUniversitarioAlgebraLinear_v2 (contato: debrittoemily@gmail.com) Python/3.10"
}

def safe_request(url, params):
    """Gerencia as requisições e trata o erro 429 (Too Many Requests)"""
    wait_time = 5  # Começa esperando 5s se der erro
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=15)
            if r.status_code == 429:
                print(f"\n[429] Wikipedia bloqueou por excesso. Esperando {wait_time}s...")
                time.sleep(wait_time)
                wait_time *= 2  # Aumenta o tempo de espera para a próxima tentativa
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"\n[WARN] Erro na tentativa {attempt+1}: {e}")
            time.sleep(2)
    return None

def get_random_page_ids(lang: str, n: int = 10) -> list[int]:
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {"action": "query", "list": "random", "rnlimit": n, "rnnamespace": 0, "format": "json"}
    data = safe_request(url, params)
    return [p["id"] for p in data["query"]["random"]] if data else []

def get_page_text(lang: str, page_id: int) -> str:
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query", "prop": "extracts", "pageids": page_id,
        "explaintext": True, "exsectionformat": "plain", "format": "json"
    }
    data = safe_request(url, params)
    if data and "query" in data:
        pages = data["query"]["pages"]
        return next(iter(pages.values())).get("extract", "")
    return ""


def clean_wiki_text(text: str) -> str:
    # Remove section headers, extra whitespace
    text = re.sub(r"==+[^=]+==+", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def fetch_corpus(lang: str, target_chars: int = TARGET_CHARS) -> str:
    print(f"Fetching {lang}... ", end="", flush=True)
    corpus = ""
    
    while len(corpus) < target_chars:
        # 1. Pede 20 IDs de uma vez
        ids = get_random_page_ids(lang, n=20)
        if not ids:
            time.sleep(5)
            continue
            
        # 2. Transforma a lista de IDs em uma string: "123|456|789..."
        ids_string = "|".join(map(str, ids))
        
        # 3. Faz UMA única requisição para pegar o texto de todas as 20
        url = f"https://{lang}.wikipedia.org/w/api.php"
        params = {
            "action": "query", "prop": "extracts", "pageids": ids_string,
            "explaintext": True, "exsectionformat": "plain", "format": "json"
        }
        
        data = safe_request(url, params)
        if data and "query" in data:
            for page in data["query"]["pages"].values():
                text = page.get("extract", "")
                if text:
                    corpus += clean_wiki_text(text) + "\n\n"
        
        print(f"{len(corpus):,} chars...", end=" ", flush=True)
        time.sleep(2) # Pausa obrigatória entre lotes de 20 páginas
        
        if len(corpus) >= target_chars:
            break

    return corpus

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for lang in LANGUAGES:
        out_path = os.path.join(OUTPUT_DIR, f"{lang}.txt")

        # Skip if already collected
        if os.path.exists(out_path):
            size = os.path.getsize(out_path)
            if size > 100_000:
                print(f"Skipping {lang} (already exists, {size:,} bytes)")
                continue

        corpus = fetch_corpus(lang)

        if len(corpus) < 10_000:
            print(f"  [ERROR] Too little data for {lang}, skipping.")
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(corpus)
        print(f"  Saved to {out_path}")

    print("\nAll done! Check data/train/")