import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers.utils import logging as hf_logging

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
hf_logging.set_verbosity_error()

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_chunks(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search local RAG index")
    parser.add_argument("query", help="Question or search text")
    parser.add_argument("--index-dir", type=Path, default=Path("data/index"), help="Index directory")
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="SentenceTransformer model name")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_dir = args.index_dir.resolve()
    chunks_path = index_dir / "chunks.jsonl"
    emb_path = index_dir / "embeddings.npy"

    if not chunks_path.exists() or not emb_path.exists():
        raise FileNotFoundError(f"Index files not found in {index_dir}. Run build_index first.")

    chunks = load_chunks(chunks_path)
    embeddings = np.load(emb_path)

    if len(chunks) != len(embeddings):
        raise RuntimeError("Index corruption: chunks.jsonl and embeddings.npy size mismatch.")

    model = SentenceTransformer(args.model_name)
    q_emb = model.encode([args.query], normalize_embeddings=True, convert_to_numpy=True)[0]

    scores = embeddings @ q_emb
    top_idx = np.argsort(-scores)[: args.top_k]

    for rank, idx in enumerate(top_idx, start=1):
        row = chunks[int(idx)]
        score = float(scores[int(idx)])
        preview = row["text"][:300].replace("\n", " ")

        print(f"[{rank}] score={score:.4f} category={row.get('category')} file={row.get('source_path')}")
        print(f"    chunk_id={row.get('chunk_id')}")
        print(f"    preview={preview}")


if __name__ == "__main__":
    main()
