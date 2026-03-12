from pathlib import Path
from typing import Any, Dict, List, Optional
import re
from uuid import uuid4

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.tools.answer_rag import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_LLM_MODE,
    DEFAULT_OLLAMA_MODEL,
    NOT_FOUND_MESSAGE,
    generate_answer,
    retrieve,
)
from src.tools.workspace_study import (
    SUPPORTED_UPLOAD_EXTENSIONS,
    UPLOADS_DIR,
    build_flashcards,
    build_quiz,
    build_summary,
    human_size,
    ingest_uploaded_file,
    list_documents,
    workspace_retrieve,
)
from src.tools.haystack_qdrant import haystack_workspace_answer

BASE_DIR = Path(__file__).resolve().parents[2]
WEB_DIR = BASE_DIR / "src" / "web"

app = FastAPI(title="Local RAG API", version="1.3.0")

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, description="User question")
    index_dir: str = Field(default="data/index_business", description="Index directory")
    embed_model: str = Field(default=DEFAULT_EMBED_MODEL, description="Embedding model")
    ollama_model: str = Field(default=DEFAULT_OLLAMA_MODEL, description="Ollama model")
    top_k: int = Field(default=8, ge=1, le=20, description="Number of retrieved chunks")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Generation temperature")
    max_tokens: int = Field(default=900, ge=32, le=4096, description="Max generated tokens")
    include_sources: bool = Field(default=False, description="Include sources in API response")
    strict_answer: bool = Field(default=False, description="Force concise answer without source section")
    llm_mode: str = Field(default=DEFAULT_LLM_MODE, description="extractive or ollama")
    anti_hallucination: bool = Field(default=True, description="Reject unsupported answers")


class WorkspaceAskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(default=8, ge=1, le=20)
    include_sources: bool = True
    doc_ids: List[str] = Field(default_factory=list)
    anti_hallucination: bool = True


class StudyRequest(BaseModel):
    doc_ids: List[str] = Field(default_factory=list)


class SourceItem(BaseModel):
    rank: int
    score: float
    category: Optional[str] = None
    source_path: Optional[str] = None
    chunk_id: Optional[str] = None
    preview: str


class AskResponse(BaseModel):
    answer: str
    model: str
    sources: Optional[List[SourceItem]] = None


class UploadResponse(BaseModel):
    document: Dict[str, Any]


class LibraryResponse(BaseModel):
    documents: List[Dict[str, Any]]
    count: int


class SummaryResponse(BaseModel):
    title: str
    bullets: List[str]
    paragraph: str


class FlashcardsResponse(BaseModel):
    flashcards: List[Dict[str, str]]


class QuizResponse(BaseModel):
    quiz: List[Dict[str, Any]]


class EmployeeItem(BaseModel):
    nom: str
    prenom: str
    equipe: str


class EmployeesResponse(BaseModel):
    employees: List[EmployeeItem]
    count: int


@app.get("/")
def web_home():
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Web UI not found")
    return FileResponse(index_path)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _sanitize_filename(name: str) -> str:
    candidate = Path(name).name
    candidate = re.sub(r"[^A-Za-z0-9._ -]+", "_", candidate)
    return candidate or f"upload_{uuid4().hex[:8]}"


def _build_prompt(question: str, contexts: List[dict], strict_answer: bool) -> str:
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

    rules = [
        "Tu es un assistant RAG d'entreprise.",
        "Reponds uniquement avec les informations presentes dans le contexte.",
        f"Si l'information manque, reponds: {NOT_FOUND_MESSAGE}.",
        "Sois precis, factuel et complet sans inventer.",
        "Quand le contexte contient des details exacts, reutilise-les explicitement.",
        "N'invente aucune information et n'utilise aucune connaissance externe au contexte.",
    ]

    if strict_answer:
        rules.append("Donne une reponse courte, directe, sans section 'Sources' et sans commentaire meta.")
    else:
        rules.append("Donne une reponse detaillee et structuree. Si utile, utilise des puces ou de courts paragraphes.")
        rules.append("Ajoute a la fin une section 'Sources' en referencant les blocs [SOURCE n] utilises.")

    return (
        "\n".join(rules)
        + f"\n\nQuestion:\n{question}\n\nContexte:\n"
        + "\n\n".join(blocks)
        + "\n"
    )


def _to_sources(contexts: List[dict]) -> List[SourceItem]:
    sources: List[SourceItem] = []
    for c in contexts:
        sources.append(
            SourceItem(
                rank=int(c.get("rank", 0)),
                score=float(c.get("score", 0.0)),
                category=c.get("category"),
                source_path=c.get("source_path"),
                chunk_id=c.get("chunk_id"),
                preview=(c.get("text", "")[:300]).replace("\n", " "),
            )
        )
    return sources


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    index_dir = Path(req.index_dir).resolve()

    try:
        contexts = retrieve(
            query=question,
            index_dir=index_dir,
            model_name=req.embed_model,
            top_k=req.top_k,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"retrieval_error: {type(exc).__name__}: {exc}") from exc

    prompt = _build_prompt(question, contexts, strict_answer=req.strict_answer)

    try:
        answer = generate_answer(
            question=question,
            contexts=contexts,
            model=req.ollama_model,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            prompt=prompt,
            llm_mode=req.llm_mode,
            anti_hallucination=req.anti_hallucination,
        )
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail="llm_unreachable: verify your configured LLM endpoint is reachable",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"generation_error: {type(exc).__name__}: {exc}") from exc

    if req.include_sources:
        return AskResponse(answer=answer, model=req.ollama_model, sources=_to_sources(contexts))
    return AskResponse(answer=answer, model=req.ollama_model)


@app.get("/api/library", response_model=LibraryResponse)
def get_library() -> LibraryResponse:
    documents = list_documents()
    return LibraryResponse(documents=documents, count=len(documents))


@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    category: str = Form(default="cours"),
) -> UploadResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_UPLOAD_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"unsupported_file_type: allowed={allowed}")

    safe_name = _sanitize_filename(file.filename or f"document{suffix}")
    target = UPLOADS_DIR / f"{uuid4().hex[:8]}_{safe_name}"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(await file.read())

    try:
        document = ingest_uploaded_file(file_name=safe_name, category=category.strip() or "cours", saved_path=target)
    except Exception as exc:
        if target.exists():
            target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"upload_error: {type(exc).__name__}: {exc}") from exc

    document["size_label"] = human_size(document.get("size_bytes", 0))
    return UploadResponse(document=document)


@app.post("/api/workspace/ask", response_model=AskResponse)
def workspace_ask(req: WorkspaceAskRequest) -> AskResponse:
    contexts = workspace_retrieve(query=req.question, top_k=req.top_k, doc_ids=req.doc_ids or None)
    if not contexts:
        return AskResponse(
            answer="Aucune information pertinente n'a été trouvée dans tes documents personnels.",
            model="workspace-extractive",
            sources=[] if req.include_sources else None,
        )

    prompt = _build_prompt(req.question, contexts, strict_answer=False)
    answer = generate_answer(
        question=req.question,
        contexts=contexts,
        model=DEFAULT_OLLAMA_MODEL,
        temperature=0.0,
        max_tokens=700,
        prompt=prompt,
        llm_mode="ollama",
        anti_hallucination=req.anti_hallucination,
    )
    if req.include_sources:
        return AskResponse(answer=answer, model=DEFAULT_OLLAMA_MODEL, sources=_to_sources(contexts))
    return AskResponse(answer=answer, model=DEFAULT_OLLAMA_MODEL)


@app.post("/api/workspace/ask/haystack", response_model=AskResponse)
def workspace_ask_haystack(req: WorkspaceAskRequest) -> AskResponse:
    try:
        result = haystack_workspace_answer(
            question=req.question,
            top_k=req.top_k,
            doc_ids=req.doc_ids or None,
            embed_model=DEFAULT_EMBED_MODEL,
            ollama_model=DEFAULT_OLLAMA_MODEL,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail="llm_unreachable: verify your configured LLM endpoint is reachable",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"haystack_rag_error: {type(exc).__name__}: {exc}") from exc

    sources = _to_sources(result.get("sources", [])) if req.include_sources else None
    return AskResponse(
        answer=str(result.get("answer", "")),
        model=str(result.get("model", DEFAULT_OLLAMA_MODEL)),
        sources=sources,
    )


@app.post("/api/study/summary", response_model=SummaryResponse)
def study_summary(req: StudyRequest) -> SummaryResponse:
    summary = build_summary(doc_ids=req.doc_ids or None)
    return SummaryResponse(**summary)


@app.post("/api/study/flashcards", response_model=FlashcardsResponse)
def study_flashcards(req: StudyRequest) -> FlashcardsResponse:
    return FlashcardsResponse(flashcards=build_flashcards(doc_ids=req.doc_ids or None))


@app.post("/api/study/quiz", response_model=QuizResponse)
def study_quiz(req: StudyRequest) -> QuizResponse:
    return QuizResponse(quiz=build_quiz(doc_ids=req.doc_ids or None))


@app.post("/ask/employees", response_model=EmployeesResponse)
def ask_employees(index_dir: str = "data/index_business", embed_model: str = DEFAULT_EMBED_MODEL, top_k: int = 5) -> EmployeesResponse:
    try:
        contexts = retrieve(
            query="Noms prenoms equipes employes RH",
            index_dir=Path(index_dir).resolve(),
            model_name=embed_model,
            top_k=top_k,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"retrieval_error: {type(exc).__name__}: {exc}") from exc

    merged = "\n".join(c.get("text", "") for c in contexts if str(c.get("category", "")).lower() == "rh")

    pattern = re.compile(
        r"Nom:\s*([^|]+?)\s*\|\s*Pr[ée]nom:\s*([^|]+?)\s*\|\s*[ÉE]quipe:\s*([^|]+?)(?:\||$)",
        re.IGNORECASE,
    )
    rows = pattern.findall(merged)

    employees: List[EmployeeItem] = []
    seen = set()
    for nom, prenom, equipe in rows:
        key = (nom.strip().lower(), prenom.strip().lower(), equipe.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        employees.append(EmployeeItem(nom=nom.strip(), prenom=prenom.strip(), equipe=equipe.strip()))

    if not employees:
        pattern_alt = re.compile(
            r"Nom:\s*([^|\\n]+?)\\s*\\|\\s*Pr[ée]nom:\\s*([^|\\n]+?)\\s*\\|\\s*[ÉE]quipe:\\s*([^|\\n]+)",
            re.IGNORECASE,
        )
        rows_alt = pattern_alt.findall(merged)
        for nom, prenom, equipe in rows_alt:
            key = (nom.strip().lower(), prenom.strip().lower(), equipe.strip().lower())
            if key in seen:
                continue
            seen.add(key)
            employees.append(EmployeeItem(nom=nom.strip(), prenom=prenom.strip(), equipe=equipe.strip()))

    return EmployeesResponse(employees=employees, count=len(employees))
