import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers.utils import logging as hf_logging

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
hf_logging.set_verbosity_error()

SUPPORTED_EXTENSIONS = {".pdf", ".xlsx", ".xlsm"}
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    source_path: str
    category: str
    file_type: str
    part: str
    char_count: int
    word_count: int


def iter_source_files(raw_dir: Path) -> List[Path]:
    files: List[Path] = []
    for p in raw_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(p)
    files.sort()
    return files


def extract_pdf_parts(path: Path) -> List[Tuple[str, str]]:
    parts: List[Tuple[str, str]] = []
    reader = PdfReader(str(path))
    for idx, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            parts.append((f"page_{idx}", text))
    return parts


def extract_excel_parts(path: Path) -> List[Tuple[str, str]]:
    parts: List[Tuple[str, str]] = []
    workbook = pd.ExcelFile(path)

    for sheet_name in workbook.sheet_names:
        df = workbook.parse(sheet_name=sheet_name, dtype=str)
        if df.empty:
            continue

        df = df.fillna("")
        lines: List[str] = []
        headers = [str(c).strip() for c in df.columns]

        for i, row in enumerate(df.itertuples(index=False), start=1):
            cells: List[str] = []
            for col_name, val in zip(headers, row):
                val_s = str(val).strip()
                if val_s:
                    cells.append(f"{col_name}: {val_s}")
            if cells:
                lines.append(f"row {i} | " + " | ".join(cells))

        if lines:
            parts.append((f"sheet_{sheet_name}", "\n".join(lines)))

    return parts


def split_words(text: str) -> List[str]:
    return re.findall(r"\S+", text)


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    words = split_words(text)
    if not words:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    step = chunk_size - chunk_overlap
    chunks: List[str] = []
    for start in range(0, len(words), step):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        if end == len(words):
            break
    return chunks


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def build_chunks(
    raw_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    min_chars: int,
    categories_filter: Optional[Set[str]] = None,
) -> List[ChunkRecord]:
    files = iter_source_files(raw_dir)
    all_chunks: List[ChunkRecord] = []

    for file_path in tqdm(files, desc="Extracting", unit="file"):
        rel = file_path.relative_to(raw_dir)
        category = rel.parts[0] if len(rel.parts) > 1 else "uncategorized"
        if categories_filter and category.lower() not in categories_filter:
            continue

        try:
            if file_path.suffix.lower() == ".pdf":
                parts = extract_pdf_parts(file_path)
                file_type = "pdf"
            else:
                parts = extract_excel_parts(file_path)
                file_type = "excel"
        except Exception as exc:
            print(f"[WARN] Failed to parse {file_path}: {type(exc).__name__}: {exc}")
            continue

        for part_name, raw_text in parts:
            text = normalize_text(raw_text)
            if len(text) < min_chars:
                continue

            for idx, chunk in enumerate(chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap), start=1):
                chunk_n = normalize_text(chunk)
                if len(chunk_n) < min_chars:
                    continue

                chunk_id = f"{rel.as_posix()}::{part_name}::chunk_{idx}"
                all_chunks.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        text=chunk_n,
                        source_path=str(file_path),
                        category=category,
                        file_type=file_type,
                        part=part_name,
                        char_count=len(chunk_n),
                        word_count=len(split_words(chunk_n)),
                    )
                )

    return all_chunks


def save_index(
    chunks: List[ChunkRecord],
    embeddings: np.ndarray,
    index_dir: Path,
    model_name: str,
    raw_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    min_chars: int,
) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = index_dir / "chunks.jsonl"
    emb_path = index_dir / "embeddings.npy"
    meta_path = index_dir / "index_meta.json"

    with chunks_path.open("w", encoding="utf-8") as f:
        for item in chunks:
            row = {
                "chunk_id": item.chunk_id,
                "text": item.text,
                "source_path": item.source_path,
                "category": item.category,
                "file_type": item.file_type,
                "part": item.part,
                "char_count": item.char_count,
                "word_count": item.word_count,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    np.save(emb_path, embeddings.astype(np.float32))

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "embedding_dim": int(embeddings.shape[1]) if embeddings.size else 0,
        "num_chunks": len(chunks),
        "raw_dir": str(raw_dir),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "min_chars": min_chars,
        "files": sorted({c.source_path for c in chunks}),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local vector index from data/raw")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Input document directory")
    parser.add_argument("--index-dir", type=Path, default=Path("data/index"), help="Output index directory")
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="SentenceTransformer model name")
    parser.add_argument("--chunk-size", type=int, default=700, help="Chunk size in words")
    parser.add_argument("--chunk-overlap", type=int, default=80, help="Chunk overlap in words")
    parser.add_argument("--min-chars", type=int, default=80, help="Minimum characters per chunk")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument(
        "--categories",
        default="",
        help="Optional comma-separated categories to index (example: RH,IT,Finance,Qualite,Securite)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    index_dir = args.index_dir.resolve()

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    print(f"[INFO] Reading documents from: {raw_dir}")
    categories_filter: Optional[Set[str]] = None
    if args.categories.strip():
        categories_filter = {x.strip().lower() for x in args.categories.split(",") if x.strip()}
        print(f"[INFO] Category filter enabled: {sorted(categories_filter)}")

    chunks = build_chunks(
        raw_dir=raw_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chars=args.min_chars,
        categories_filter=categories_filter,
    )

    if not chunks:
        print("[INFO] No chunks generated. Nothing to index.")
        return

    texts = [c.text for c in chunks]
    print(f"[INFO] Generated chunks: {len(texts)}")
    print(f"[INFO] Loading embedding model: {args.model_name}")
    model = SentenceTransformer(args.model_name)

    print("[INFO] Computing embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    save_index(
        chunks=chunks,
        embeddings=embeddings,
        index_dir=index_dir,
        model_name=args.model_name,
        raw_dir=raw_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chars=args.min_chars,
    )

    print(f"[OK] Index saved to: {index_dir}")
    print(f"[OK] chunks.jsonl: {index_dir / 'chunks.jsonl'}")
    print(f"[OK] embeddings.npy: {index_dir / 'embeddings.npy'}")
    print(f"[OK] index_meta.json: {index_dir / 'index_meta.json'}")


if __name__ == "__main__":
    main()
