from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from sentence_transformers import SentenceTransformer

from src.tools.answer_rag import DEFAULT_EMBED_MODEL, DEFAULT_OLLAMA_MODEL, generate_answer
from src.tools.workspace_study import load_workspace_chunks

QDRANT_WORKSPACE_DIR = Path("data/qdrant_workspace")
HAYSTACK_HOME_DIR = Path(".haystack_home")


def _require_haystack() -> tuple[Any, Any, Any, Any, Any, Any]:
    HAYSTACK_HOME_DIR.mkdir(parents=True, exist_ok=True)
    import os

    os.environ.setdefault("HAYSTACK_HOME", str(HAYSTACK_HOME_DIR.resolve()))
    os.environ.setdefault("HAYSTACK_TELEMETRY_ENABLED", "False")

    try:
        from haystack import Document, Pipeline
        from haystack.components.embedders import (
            SentenceTransformersDocumentEmbedder,
            SentenceTransformersTextEmbedder,
        )
        from haystack.components.writers import DocumentWriter
        from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
        from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
    except ImportError as exc:
        raise RuntimeError(
            "Haystack/Qdrant n'est pas installe. Installe `haystack-ai` et `qdrant-haystack`."
        ) from exc

    return (
        Document,
        Pipeline,
        SentenceTransformersDocumentEmbedder,
        SentenceTransformersTextEmbedder,
        DocumentWriter,
        QdrantDocumentStore,
        QdrantEmbeddingRetriever,
    )


def _embedding_dim(model_name: str) -> int:
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    return int(dim or 384)


def _workspace_rows(doc_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    rows = load_workspace_chunks()
    if doc_ids:
        allowed = set(doc_ids)
        rows = [row for row in rows if row.get("doc_id") in allowed]
    return rows


def _workspace_documents(doc_ids: Optional[List[str]] = None) -> List[Any]:
    (
        Document,
        _Pipeline,
        _SentenceTransformersDocumentEmbedder,
        _SentenceTransformersTextEmbedder,
        _DocumentWriter,
        _QdrantDocumentStore,
        _QdrantEmbeddingRetriever,
    ) = _require_haystack()

    documents: List[Any] = []
    for row in _workspace_rows(doc_ids=doc_ids):
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        documents.append(
            Document(
                content=text,
                meta={
                    "chunk_id": row.get("chunk_id"),
                    "doc_id": row.get("doc_id"),
                    "doc_name": row.get("doc_name"),
                    "source_path": row.get("source_path"),
                    "category": row.get("category"),
                    "part": row.get("part"),
                    "file_type": row.get("file_type"),
                },
            )
        )
    return documents


def build_workspace_haystack_store(
    doc_ids: Optional[List[str]] = None,
    model_name: str = DEFAULT_EMBED_MODEL,
    collection_name: str = "workspace_chunks",
) -> Any:
    (
        _Document,
        Pipeline,
        SentenceTransformersDocumentEmbedder,
        _SentenceTransformersTextEmbedder,
        DocumentWriter,
        QdrantDocumentStore,
        _QdrantEmbeddingRetriever,
    ) = _require_haystack()

    documents = _workspace_documents(doc_ids=doc_ids)
    if not documents:
        return None

    QDRANT_WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    document_store = QdrantDocumentStore(
        path=str(QDRANT_WORKSPACE_DIR.resolve()),
        index=collection_name,
        embedding_dim=_embedding_dim(model_name),
        recreate_index=True,
        return_embedding=True,
        wait_result_from_api=True,
    )

    indexing = Pipeline()
    indexing.add_component("embedder", SentenceTransformersDocumentEmbedder(model=model_name))
    indexing.add_component("writer", DocumentWriter(document_store=document_store))
    indexing.connect("embedder.documents", "writer.documents")
    indexing.run({"embedder": {"documents": documents}})
    return document_store


def haystack_workspace_retrieve(
    query: str,
    top_k: int = 8,
    doc_ids: Optional[List[str]] = None,
    model_name: str = DEFAULT_EMBED_MODEL,
    collection_name: str = "workspace_chunks",
) -> List[Dict[str, Any]]:
    (
        _Document,
        Pipeline,
        _SentenceTransformersDocumentEmbedder,
        SentenceTransformersTextEmbedder,
        _DocumentWriter,
        _QdrantDocumentStore,
        QdrantEmbeddingRetriever,
    ) = _require_haystack()

    document_store = build_workspace_haystack_store(
        doc_ids=doc_ids,
        model_name=model_name,
        collection_name=collection_name,
    )
    if document_store is None:
        return []

    query_pipeline = Pipeline()
    query_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(model=model_name))
    query_pipeline.add_component("retriever", QdrantEmbeddingRetriever(document_store=document_store))
    query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    result = query_pipeline.run(
        {
            "text_embedder": {"text": query},
            "retriever": {"top_k": top_k},
        }
    )

    documents = result.get("retriever", {}).get("documents", [])
    contexts: List[Dict[str, Any]] = []
    for rank, doc in enumerate(documents, start=1):
        meta = getattr(doc, "meta", {}) or {}
        score = float(getattr(doc, "score", 0.0) or 0.0)
        contexts.append(
            {
                "rank": rank,
                "score": score,
                "chunk_id": meta.get("chunk_id"),
                "doc_id": meta.get("doc_id"),
                "doc_name": meta.get("doc_name"),
                "source_path": meta.get("source_path"),
                "category": meta.get("category"),
                "part": meta.get("part"),
                "file_type": meta.get("file_type"),
                "text": getattr(doc, "content", "") or "",
            }
        )
    return contexts


def haystack_workspace_answer(
    question: str,
    top_k: int = 8,
    doc_ids: Optional[List[str]] = None,
    embed_model: str = DEFAULT_EMBED_MODEL,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    max_tokens: int = 700,
) -> Dict[str, Any]:
    contexts = haystack_workspace_retrieve(
        query=question,
        top_k=top_k,
        doc_ids=doc_ids,
        model_name=embed_model,
    )
    if not contexts:
        return {
            "answer": "Aucune information pertinente n'a ete trouvee dans tes documents personnels.",
            "model": ollama_model,
            "sources": [],
        }

    answer = generate_answer(
        question=question,
        contexts=contexts,
        model=ollama_model,
        temperature=0.0,
        max_tokens=max_tokens,
        llm_mode="ollama",
        anti_hallucination=True,
    )
    return {"answer": answer, "model": ollama_model, "sources": contexts}
