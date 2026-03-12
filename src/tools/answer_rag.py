import argparse
import contextlib
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def load_chunks(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def retrieve(query: str, index_dir: Path, model_name: str, top_k: int) -> List[dict]:
    chunks_path = index_dir / "chunks.jsonl"
    emb_path = index_dir / "embeddings.npy"

    if not chunks_path.exists() or not emb_path.exists():
        raise FileNotFoundError(f"Index files not found in {index_dir}. Run build_index first.")

    chunks = load_chunks(chunks_path)
    embeddings = np.load(emb_path)

    if len(chunks) != len(embeddings):
        raise RuntimeError("Index mismatch: chunks.jsonl and embeddings.npy sizes differ.")

    # Suppress noisy stderr logs from embedding backends.
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stderr(devnull):
            embedder = SentenceTransformer(model_name)
            q_emb = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]

    scores = embeddings @ q_emb
    top_idx = np.argsort(-scores)[:top_k]

    results: List[dict] = []
    for rank, idx in enumerate(top_idx, start=1):
        i = int(idx)
        row = chunks[i]
        row["score"] = float(scores[i])
        row["rank"] = rank
        results.append(row)
    return results


def build_prompt(question: str, contexts: List[dict]) -> str:
    blocks = []
    for c in contexts:
        blocks.append(
            f"[SOURCE {c['rank']}]\n"
            f"category: {c.get('category','')}\n"
            f"file: {c.get('source_path','')}\n"
            f"chunk_id: {c.get('chunk_id','')}\n"
            f"score: {c.get('score',0):.4f}\n"
            f"content:\n{c.get('text','')}"
        )

    joined = "\n\n".join(blocks)

    return (
        "Tu es un assistant RAG d'entreprise. Reponds uniquement a partir du contexte fourni.\n"
        "Si l'information n'est pas dans le contexte, dis clairement: 'Information non trouvee dans les documents fournis.'\n"
        "Reponse en francais, claire et concise, puis une section 'Sources' avec [SOURCE n].\n\n"
        f"Question:\n{question}\n\n"
        f"Contexte:\n{joined}\n"
    )


def query_ollama(model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Answer questions with local RAG + Ollama")
    parser.add_argument("question", help="Question utilisateur")
    parser.add_argument("--index-dir", type=Path, default=Path("data/index_business"), help="Index directory")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Embedding model name")
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL, help="Ollama model name")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved chunks")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature")
    parser.add_argument("--max-tokens", type=int, default=500, help="Maximum generated tokens")
    parser.add_argument("--show-context", action="store_true", help="Print retrieved chunks before answer")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_dir = args.index_dir.resolve()

    contexts = retrieve(
        query=args.question,
        index_dir=index_dir,
        model_name=args.embed_model,
        top_k=args.top_k,
    )

    if args.show_context:
        print("=== Retrieved Context ===")
        for c in contexts:
            preview = (c.get("text", "")[:250]).replace("\n", " ")
            print(f"[{c['rank']}] score={c['score']:.4f} | {c.get('category')} | {c.get('source_path')}")
            print(f"    {preview}")

    prompt = build_prompt(args.question, contexts)

    try:
        answer = query_ollama(
            model=args.ollama_model,
            prompt=prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            "Impossible de contacter Ollama. Verifie que Ollama est installe et lance (http://127.0.0.1:11434)."
        ) from exc

    print("=== Reponse ===")
    print(answer)


if __name__ == "__main__":
    main()
