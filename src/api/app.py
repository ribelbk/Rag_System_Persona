from pathlib import Path
from typing import List, Optional
import re

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.tools.answer_rag import (
    DEFAULT_LLM_MODE,
    DEFAULT_EMBED_MODEL,
    DEFAULT_OLLAMA_MODEL,
    generate_answer,
    retrieve,
)

BASE_DIR = Path(__file__).resolve().parents[2]
WEB_DIR = BASE_DIR / "src" / "web"

app = FastAPI(title="Local RAG API", version="1.2.0")

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, description="User question")
    index_dir: str = Field(default="data/index_business", description="Index directory")
    embed_model: str = Field(default=DEFAULT_EMBED_MODEL, description="Embedding model")
    ollama_model: str = Field(default=DEFAULT_OLLAMA_MODEL, description="Ollama model")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of retrieved chunks")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Generation temperature")
    max_tokens: int = Field(default=500, ge=32, le=4096, description="Max generated tokens")
    include_sources: bool = Field(default=False, description="Include sources in API response")
    strict_answer: bool = Field(default=True, description="Force concise answer without source section")
    llm_mode: str = Field(default=DEFAULT_LLM_MODE, description="extractive or ollama")


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
        "Si l'information manque, reponds: Information non trouvee dans les documents fournis.",
    ]

    if strict_answer:
        rules.append("Donne une reponse courte, directe, sans section 'Sources' et sans commentaire meta.")

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
