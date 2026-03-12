import argparse
import contextlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_LLM_MODE = os.getenv("RAG_LLM_MODE", "extractive").strip().lower() or "extractive"
DEFAULT_RETRIEVAL_MODE = os.getenv("RAG_RETRIEVAL_MODE", "lexical").strip().lower() or "lexical"
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate").strip()
DEFAULT_OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "").strip()

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

STOPWORDS = {
    "a",
    "alors",
    "au",
    "aucun",
    "aussi",
    "autre",
    "aux",
    "avec",
    "avoir",
    "bon",
    "car",
    "ce",
    "cela",
    "ces",
    "ceux",
    "chaque",
    "ci",
    "comme",
    "comment",
    "dans",
    "des",
    "du",
    "dedans",
    "dehors",
    "depuis",
    "devrait",
    "doit",
    "donc",
    "dos",
    "droite",
    "de",
    "elle",
    "elles",
    "en",
    "encore",
    "essai",
    "est",
    "et",
    "eu",
    "fait",
    "faites",
    "fois",
    "font",
    "force",
    "haut",
    "hors",
    "ici",
    "il",
    "ils",
    "je",
    "juste",
    "la",
    "le",
    "les",
    "leur",
    "là",
    "ma",
    "maintenant",
    "mais",
    "mes",
    "mine",
    "moins",
    "mon",
    "mot",
    "même",
    "ni",
    "nommés",
    "notre",
    "nous",
    "nouveaux",
    "ou",
    "où",
    "par",
    "parce",
    "parole",
    "pas",
    "personnes",
    "peut",
    "peu",
    "pièce",
    "plupart",
    "pour",
    "pourquoi",
    "quand",
    "que",
    "quel",
    "quelle",
    "quelles",
    "quels",
    "qui",
    "sa",
    "sans",
    "ses",
    "seulement",
    "si",
    "sien",
    "son",
    "sont",
    "sous",
    "soyez",
    "sujet",
    "sur",
    "ta",
    "tandis",
    "tellement",
    "tels",
    "tes",
    "ton",
    "tous",
    "tout",
    "trop",
    "très",
    "tu",
    "valeur",
    "voie",
    "voient",
    "vont",
    "votre",
    "vous",
    "vu",
    "ça",
    "étaient",
    "état",
    "étions",
    "été",
    "être",
}


def load_chunks(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"\w+", text.lower())
    return [token for token in tokens if len(token) > 2 and token not in STOPWORDS]


@lru_cache(maxsize=2)
def _load_embedder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def _lexical_retrieve(query: str, chunks: List[dict], top_k: int) -> List[dict]:
    query_terms = _tokenize(query)
    if not query_terms:
        return []

    query_counts: dict[str, int] = {}
    for term in query_terms:
        query_counts[term] = query_counts.get(term, 0) + 1

    scored: List[tuple[float, int]] = []
    for idx, row in enumerate(chunks):
        text = row.get("text", "")
        tokens = _tokenize(text)
        if not tokens:
            continue

        token_counts: dict[str, int] = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        overlap = sum(min(token_counts.get(term, 0), count) for term, count in query_counts.items())
        if overlap == 0:
            continue

        density = overlap / max(len(tokens), 1)
        scored.append((overlap + density, idx))

    scored.sort(key=lambda item: item[0], reverse=True)

    results: List[dict] = []
    for rank, (score, idx) in enumerate(scored[:top_k], start=1):
        row = dict(chunks[idx])
        row["score"] = float(score)
        row["rank"] = rank
        results.append(row)
    return results


def retrieve(
    query: str,
    index_dir: Path,
    model_name: str,
    top_k: int,
    retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
) -> List[dict]:
    chunks_path = index_dir / "chunks.jsonl"
    emb_path = index_dir / "embeddings.npy"

    if not chunks_path.exists() or not emb_path.exists():
        raise FileNotFoundError(f"Index files not found in {index_dir}. Run build_index first.")

    chunks = load_chunks(chunks_path)
    embeddings = np.load(emb_path)

    if len(chunks) != len(embeddings):
        raise RuntimeError("Index mismatch: chunks.jsonl and embeddings.npy sizes differ.")

    if retrieval_mode == "lexical":
        fallback = _lexical_retrieve(query=query, chunks=chunks, top_k=top_k)
        if fallback:
            return fallback

    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stderr(devnull):
                embedder = _load_embedder(model_name)
                q_emb = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    except Exception:
        fallback = _lexical_retrieve(query=query, chunks=chunks, top_k=top_k)
        if fallback:
            return fallback
        raise

    scores = embeddings @ q_emb
    top_idx = np.argsort(-scores)[:top_k]

    results: List[dict] = []
    for rank, idx in enumerate(top_idx, start=1):
        i = int(idx)
        row = dict(chunks[i])
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


def query_ollama(
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    base_url: str = DEFAULT_OLLAMA_URL,
    api_key: str = DEFAULT_OLLAMA_API_KEY,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.post(base_url, json=payload, headers=headers, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()


def _extractive_answer(question: str, contexts: List[dict]) -> str:
    if not contexts:
        return "Information non trouvee dans les documents fournis."

    terms = set(_tokenize(question))
    ranked_sentences: List[tuple[int, float, str]] = []
    seen: set[str] = set()
    for context in contexts:
        text = context.get("text", "")
        for sentence in re.split(r"(?<=[.!?])\s+|\s+\|\s+", text):
            clean = sentence.strip()
            if len(clean) < 20:
                continue
            normalized = clean.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            overlap = len(terms & set(_tokenize(clean)))
            if overlap == 0 and ranked_sentences:
                continue
            ranked_sentences.append((overlap, float(context.get("score", 0.0)), clean))

    if ranked_sentences:
        ranked_sentences.sort(key=lambda item: (item[0], item[1], len(item[2])), reverse=True)
        return " ".join(item[2] for item in ranked_sentences[:2])

    return contexts[0].get("text", "")[:400] or "Information non trouvee dans les documents fournis."


def generate_answer(
    question: str,
    contexts: List[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    prompt: Optional[str] = None,
    llm_mode: str = DEFAULT_LLM_MODE,
) -> str:
    mode = (llm_mode or DEFAULT_LLM_MODE).strip().lower()
    if mode == "ollama":
        if prompt is None:
            prompt = build_prompt(question, contexts)
        return query_ollama(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    return _extractive_answer(question=question, contexts=contexts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Answer questions with local RAG")
    parser.add_argument("question", help="Question utilisateur")
    parser.add_argument("--index-dir", type=Path, default=Path("data/index_business"), help="Index directory")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Embedding model name")
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL, help="Ollama model name")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved chunks")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature")
    parser.add_argument("--max-tokens", type=int, default=500, help="Maximum generated tokens")
    parser.add_argument("--show-context", action="store_true", help="Print retrieved chunks before answer")
    parser.add_argument(
        "--llm-mode",
        default=DEFAULT_LLM_MODE,
        choices=["extractive", "ollama"],
        help="Answer generation mode",
    )
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
        answer = generate_answer(
            question=args.question,
            contexts=contexts,
            model=args.ollama_model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            prompt=prompt,
            llm_mode=args.llm_mode,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Impossible de contacter le service LLM configure ({DEFAULT_OLLAMA_URL})."
        ) from exc

    print("=== Reponse ===")
    print(answer)


if __name__ == "__main__":
    main()
