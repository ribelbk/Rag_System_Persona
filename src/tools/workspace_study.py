import json
import re
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4
import xml.etree.ElementTree as ET

import pandas as pd
import numpy as np
from pypdf import PdfReader

from src.tools.answer_rag import DEFAULT_EMBED_MODEL, _lexical_retrieve, _load_embedder
from src.tools.build_index import chunk_text, normalize_text

WORKSPACE_DIR = Path("data/user_workspace")
UPLOADS_DIR = WORKSPACE_DIR / "uploads"
CHUNKS_PATH = WORKSPACE_DIR / "chunks.jsonl"
MANIFEST_PATH = WORKSPACE_DIR / "manifest.json"
SUPPORTED_UPLOAD_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".xlsx", ".xlsm"}

STOPWORDS = {
    "a",
    "ai",
    "ainsi",
    "alors",
    "au",
    "aux",
    "avec",
    "avoir",
    "ce",
    "ces",
    "comme",
    "dans",
    "de",
    "des",
    "du",
    "elle",
    "en",
    "est",
    "et",
    "etre",
    "il",
    "ils",
    "je",
    "la",
    "le",
    "les",
    "leur",
    "mais",
    "me",
    "mes",
    "mon",
    "ne",
    "nos",
    "notre",
    "nous",
    "ou",
    "par",
    "pas",
    "pour",
    "qu",
    "que",
    "qui",
    "sa",
    "se",
    "ses",
    "son",
    "sur",
    "te",
    "tes",
    "ton",
    "tu",
    "un",
    "une",
    "vos",
    "votre",
    "vous",
}


def ensure_workspace() -> None:
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    if not MANIFEST_PATH.exists():
        MANIFEST_PATH.write_text("[]", encoding="utf-8")
    if not CHUNKS_PATH.exists():
        CHUNKS_PATH.write_text("", encoding="utf-8")


def load_manifest() -> List[Dict[str, Any]]:
    ensure_workspace()
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def save_manifest(items: List[Dict[str, Any]]) -> None:
    MANIFEST_PATH.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def load_workspace_chunks() -> List[Dict[str, Any]]:
    ensure_workspace()
    rows: List[Dict[str, Any]] = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_workspace_chunks(rows: List[Dict[str, Any]]) -> None:
    ensure_workspace()
    with CHUNKS_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_workspace_chunks(rows: List[Dict[str, Any]]) -> None:
    ensure_workspace()
    with CHUNKS_PATH.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _extract_pdf(path: Path) -> str:
    parts: List[str] = []
    reader = PdfReader(str(path))
    for page in reader.pages:
        text = (page.extract_text() or "").strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def _extract_excel(path: Path) -> str:
    workbook = pd.ExcelFile(path)
    parts: List[str] = []
    for sheet_name in workbook.sheet_names:
        df = workbook.parse(sheet_name=sheet_name, dtype=str).fillna("")
        if df.empty:
            continue
        headers = [str(column).strip() for column in df.columns]
        for idx, row in enumerate(df.itertuples(index=False), start=1):
            cells = []
            for header, value in zip(headers, row):
                value_str = str(value).strip()
                if value_str:
                    cells.append(f"{header}: {value_str}")
            if cells:
                parts.append(f"Sheet {sheet_name} | row {idx} | " + " | ".join(cells))
    return "\n\n".join(parts)


def _excel_rows(path: Path) -> List[str]:
    workbook = pd.ExcelFile(path)
    rows: List[str] = []
    for sheet_name in workbook.sheet_names:
        df = workbook.parse(sheet_name=sheet_name, dtype=str).fillna("")
        if df.empty:
            continue
        headers = [str(column).strip() for column in df.columns]
        for idx, row in enumerate(df.itertuples(index=False), start=1):
            cells = []
            for header, value in zip(headers, row):
                value_str = str(value).strip()
                if value_str:
                    cells.append(f"{header}: {value_str}")
            if cells:
                rows.append(f"Sheet {sheet_name} | row {idx} | " + " | ".join(cells))
    return rows


def _extract_docx(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        xml_data = archive.read("word/document.xml")
    root = ET.fromstring(xml_data)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: List[str] = []
    for paragraph in root.findall(".//w:p", ns):
        texts = [node.text for node in paragraph.findall(".//w:t", ns) if node.text]
        joined = "".join(texts).strip()
        if joined:
            paragraphs.append(joined)
    return "\n".join(paragraphs)


def extract_text_from_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path).replace("\ufeff", "")
    if suffix in {".xlsx", ".xlsm"}:
        return _extract_excel(path).replace("\ufeff", "")
    if suffix == ".docx":
        return _extract_docx(path).replace("\ufeff", "")
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8-sig", errors="ignore").replace("\ufeff", "")
    raise ValueError(f"Unsupported file type: {suffix}")


def _doc_status(chunks: int) -> str:
    return "ready" if chunks > 0 else "empty"


def ingest_uploaded_file(file_name: str, category: str, saved_path: Path) -> Dict[str, Any]:
    ensure_workspace()
    doc_id = uuid4().hex[:12]
    chunk_rows: List[Dict[str, Any]] = []
    if saved_path.suffix.lower() in {".xlsx", ".xlsm"}:
        row_texts = [normalize_text(row) for row in _excel_rows(saved_path)]
        for idx, row_text in enumerate(row_texts, start=1):
            if len(row_text) < 20:
                continue
            chunk_rows.append(
                {
                    "chunk_id": f"{doc_id}::chunk_{idx}",
                    "doc_id": doc_id,
                    "doc_name": file_name,
                    "text": row_text,
                    "source_path": str(saved_path),
                    "category": category,
                    "file_type": saved_path.suffix.lower().lstrip("."),
                    "part": f"row_{idx}",
                    "score": 0.0,
                }
            )
    else:
        text = normalize_text(extract_text_from_file(saved_path))
        if not text:
            raise ValueError("No extractable text found in uploaded file.")

        chunk_texts = chunk_text(text, chunk_size=350, chunk_overlap=60)
        for idx, chunk in enumerate(chunk_texts, start=1):
            cleaned = normalize_text(chunk)
            if len(cleaned) < 40:
                continue
            chunk_rows.append(
                {
                    "chunk_id": f"{doc_id}::chunk_{idx}",
                    "doc_id": doc_id,
                    "doc_name": file_name,
                    "text": cleaned,
                    "source_path": str(saved_path),
                    "category": category,
                    "file_type": saved_path.suffix.lower().lstrip("."),
                    "part": f"chunk_{idx}",
                    "score": 0.0,
                }
            )

    if not chunk_rows:
        raise ValueError("No extractable text found in uploaded file.")

    append_workspace_chunks(chunk_rows)

    manifest = load_manifest()
    uploaded_at = datetime.now(timezone.utc).isoformat()
    record = {
        "doc_id": doc_id,
        "name": file_name,
        "category": category,
        "source_path": str(saved_path),
        "uploaded_at": uploaded_at,
        "size_bytes": saved_path.stat().st_size,
        "chunks": len(chunk_rows),
        "status": _doc_status(len(chunk_rows)),
    }
    manifest.insert(0, record)
    save_manifest(manifest)
    return record


def list_documents() -> List[Dict[str, Any]]:
    items = load_manifest()
    for item in items:
        item["size_label"] = human_size(item.get("size_bytes", 0))
    return items


def human_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size_bytes} B"


def _semantic_workspace_retrieve(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int,
    model_name: str = DEFAULT_EMBED_MODEL,
) -> List[Dict[str, Any]]:
    if not chunks:
        return []

    texts = [chunk.get("text", "") for chunk in chunks]
    if not any(texts):
        return []

    try:
        embedder = _load_embedder(model_name)
        chunk_embeddings = embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        query_embedding = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    except Exception:
        return []

    scores = np.asarray(chunk_embeddings @ query_embedding, dtype=np.float32)
    top_idx = np.argsort(-scores)[:top_k]

    results: List[Dict[str, Any]] = []
    for rank, idx in enumerate(top_idx, start=1):
        i = int(idx)
        row = dict(chunks[i])
        row["score"] = float(scores[i])
        row["rank"] = rank
        results.append(row)
    return results


def _merge_ranked_results(*result_sets: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    for weight, rows in enumerate(result_sets[::-1], start=1):
        boost = float(weight)
        for row in rows:
            chunk_id = str(row.get("chunk_id", ""))
            if not chunk_id:
                continue
            score = float(row.get("score", 0.0))
            rank = int(row.get("rank", 9999))
            combined = score + (1.0 / max(rank, 1)) * boost
            existing = merged.get(chunk_id)
            if existing is None or combined > float(existing.get("_combined_score", -1e9)):
                item = dict(row)
                item["_combined_score"] = combined
                merged[chunk_id] = item

    ordered = sorted(
        merged.values(),
        key=lambda row: (float(row.get("_combined_score", 0.0)), float(row.get("score", 0.0))),
        reverse=True,
    )[:top_k]

    results: List[Dict[str, Any]] = []
    for rank, row in enumerate(ordered, start=1):
        item = dict(row)
        item["rank"] = rank
        item["score"] = float(item.get("_combined_score", item.get("score", 0.0)))
        item.pop("_combined_score", None)
        results.append(item)
    return results


def workspace_retrieve(query: str, top_k: int = 5, doc_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    chunks = load_workspace_chunks()
    if doc_ids:
        allowed = set(doc_ids)
        chunks = [chunk for chunk in chunks if chunk.get("doc_id") in allowed]
    if not chunks:
        return []

    semantic = _semantic_workspace_retrieve(query=query, chunks=chunks, top_k=top_k * 2)
    lexical = _lexical_retrieve(query=query, chunks=chunks, top_k=top_k * 2)

    if semantic and lexical:
        return _merge_ranked_results(semantic, lexical, top_k=top_k)
    if semantic:
        return semantic[:top_k]
    return lexical[:top_k]


def _sentences_from_chunks(chunks: List[Dict[str, Any]]) -> List[str]:
    seen: set[str] = set()
    sentences: List[str] = []
    for chunk in chunks:
        for sentence in re.split(r"(?<=[.!?])\s+", chunk.get("text", "")):
            clean = sentence.strip()
            if len(clean) < 25:
                continue
            normalized = clean.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            sentences.append(clean)
    return sentences


def _keyword_score(sentences: List[str]) -> Counter:
    counts: Counter = Counter()
    for sentence in sentences:
        for token in re.findall(r"[A-Za-zÀ-ÿ0-9]{4,}", sentence.lower()):
            if token not in STOPWORDS:
                counts[token] += 1
    return counts


def build_summary(doc_ids: Optional[List[str]] = None, max_points: int = 5) -> Dict[str, Any]:
    chunks = workspace_retrieve("résumé cours notions importantes objectifs conclusion", top_k=10, doc_ids=doc_ids)
    if not chunks and doc_ids:
        all_chunks = load_workspace_chunks()
        chunks = [chunk for chunk in all_chunks if chunk.get("doc_id") in set(doc_ids)]
    sentences = _sentences_from_chunks(chunks)
    if not sentences:
        return {"title": "Résumé", "bullets": [], "paragraph": "Aucun contenu disponible pour générer un résumé."}

    keywords = _keyword_score(sentences)
    ranked = sorted(
        sentences,
        key=lambda sentence: sum(keywords.get(token, 0) for token in re.findall(r"[A-Za-zÀ-ÿ0-9]{4,}", sentence.lower())),
        reverse=True,
    )
    bullets = ranked[:max_points]
    paragraph = " ".join(ranked[:3])
    return {"title": "Résumé du cours / projet", "bullets": bullets, "paragraph": paragraph}


def build_flashcards(doc_ids: Optional[List[str]] = None, limit: int = 6) -> List[Dict[str, str]]:
    chunks = workspace_retrieve("définition concept important procédure étape méthode", top_k=12, doc_ids=doc_ids)
    if not chunks and doc_ids:
        all_chunks = load_workspace_chunks()
        chunks = [chunk for chunk in all_chunks if chunk.get("doc_id") in set(doc_ids)]
    sentences = _sentences_from_chunks(chunks)
    cards: List[Dict[str, str]] = []
    for sentence in sentences:
        parts = re.split(r":| est | signifie | correspond à ", sentence, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2:
            question = f"Explique: {parts[0].strip()}"
            answer = parts[1].strip()
        else:
            question = f"Que faut-il retenir ?"
            answer = sentence
        if len(answer) < 20:
            continue
        cards.append({"question": question, "answer": answer})
        if len(cards) >= limit:
            break
    if not cards:
        cards.append({"question": "Quel est le point principal du document ?", "answer": "Aucun contenu exploitable n'a été trouvé pour produire des fiches."})
    return cards


def build_quiz(doc_ids: Optional[List[str]] = None, limit: int = 5) -> List[Dict[str, Any]]:
    flashcards = build_flashcards(doc_ids=doc_ids, limit=max(limit + 2, 6))
    answers = [card["answer"] for card in flashcards]
    quiz: List[Dict[str, Any]] = []
    for idx, card in enumerate(flashcards[:limit], start=1):
        distractors = [answer for answer in answers if answer != card["answer"]][:3]
        while len(distractors) < 3:
            distractors.append("Information non présente dans le document.")
        options = [card["answer"], *distractors[:3]]
        quiz.append(
            {
                "id": idx,
                "question": card["question"],
                "options": options,
                "answer": card["answer"],
            }
        )
    return quiz
