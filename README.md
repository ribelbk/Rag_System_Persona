# RAG Project - Local Corpus + Index Builder

Pipeline local pour:

1. telecharger des documents publics (`download_corpus.py`)
2. construire un index vectoriel local depuis `data/raw/` (`build_index.py`)
3. tester la recherche semantique (`search_index.py`)

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 1) Telechargement (optionnel)

```powershell
scripts\download_corpus.bat --seed-file seeds_business.json
```

Sorties:

- documents: `data/raw/<SOURCE>/`
- manifest: `data/manifest.jsonl`

## 2) Construire l'index RAG local

```powershell
scripts\build_index.bat
```

Options utiles:

```powershell
scripts\build_index.bat --chunk-size 700 --chunk-overlap 80 --min-chars 80
```

Indexer seulement les categories metier:

```powershell
scripts\build_index.bat --categories RH,IT,Finance,Qualite,Securite
```

Sorties index:

- `data/index/chunks.jsonl` (texte + metadonnees)
- `data/index/embeddings.npy` (vecteurs normalises)
- `data/index/index_meta.json` (config + stats)

## 3) Tester la recherche

```powershell
scripts\search_index.bat "quelle est la procedure incident securite ?" --top-k 5
```

## 4) Repondre avec Ollama (RAG complet)

Installer Ollama (Windows), puis verifier:

```powershell
ollama --version
```

Telecharger un modele:

```powershell
ollama pull llama3.1:8b
```

Lancer une question RAG:

```powershell
scripts\answer_rag.bat "Quelle est la procedure d'escalade en cas d'incident critique ?" --ollama-model llama3.1:8b --top-k 5
```

Option debug (voir le contexte recupere):

```powershell
scripts\answer_rag.bat "..." --show-context
```

## 5) API REST (question -> reponse)

Lancer l'API:

```powershell
scripts\run_api.bat
```

Interface web chatbot:

- Local: `http://127.0.0.1:8000/`
- Swagger: `http://127.0.0.1:8000/docs`

Endpoints:

- `GET http://127.0.0.1:8000/health`
- `POST http://127.0.0.1:8000/ask`
- `POST http://127.0.0.1:8000/api/workspace/ask`
- `POST http://127.0.0.1:8000/api/workspace/ask/haystack`
- `POST http://127.0.0.1:8000/ask/employees`

Exemple PowerShell:

```powershell
$body = @{
  question = "Quelle est la procedure d'escalade en cas d'incident critique ?"
  index_dir = "data/index_business"
  ollama_model = "llama3.1:8b"
  top_k = 5
  include_sources = $false
  strict_answer = $true
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/ask" -Method Post -ContentType "application/json" -Body $body
```

Exemple `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"Liste les employes et leurs equipes\",\"index_dir\":\"data/index_business\",\"ollama_model\":\"llama3.1:8b\",\"top_k\":5,\"include_sources\":false,\"strict_answer\":true}"

Extraction stricte des employes (JSON structure):

```bash
curl -X POST "http://127.0.0.1:8000/ask/employees?index_dir=data/index_business&top_k=5"
```

## 6) Exposer le site avec Cloudflared (sans domaine)

Quand le backend tourne, expose le site:

```powershell
cloudflared tunnel --url http://127.0.0.1:8000
```

Cloudflared affichera une URL publique `https://xxxx.trycloudflare.com` que tu peux partager.
```

## 7) Deploiement Render

Le depot contient maintenant un fichier `render.yaml`.

Commande de demarrage:

```bash
python -m uvicorn src.api.app:app --host 0.0.0.0 --port $PORT
```

Attention:

- Render ne lance pas Ollama localement.
- Pour les endpoints qui generent une reponse avec le LLM, il faut fournir un endpoint LLM accessible depuis Render via `OLLAMA_URL`, ou adapter le backend vers un autre fournisseur.
- Le retrieval Haystack/Qdrant peut etre deploye, mais la generation depend toujours du service LLM configure.

Notes GPU:

- Ollama utilise automatiquement le GPU NVIDIA si disponible.
- Verifier l'activite GPU pendant une requete:
  - `nvidia-smi`
- Si le modele est trop grand pour la VRAM, Ollama bascule partiellement/totalement sur CPU.

## Types de fichiers supportes pour l'index

- PDF (`.pdf`)
- Excel (`.xlsx`, `.xlsm`)

Le script lit recursivement `data/raw/`, detecte la categorie a partir du dossier parent (`RH`, `IT`, `Finance`, `Qualite`, `Securite`, etc.), puis:

- extrait le texte
- decoupe en chunks (mots)
- calcule les embeddings (`sentence-transformers`)
- sauvegarde un index persistant local

## Fichiers principaux

- downloader: `src/tools/download_corpus.py`
- build index: `src/tools/build_index.py`
- search index: `src/tools/search_index.py`
- rag answer (ollama): `src/tools/answer_rag.py`
- api rest: `src/api/app.py`
- lanceurs Windows:
  - `scripts/download_corpus.bat`
  - `scripts/build_index.bat`
  - `scripts/search_index.bat`
  - `scripts/answer_rag.bat`
  - `scripts/run_api.bat`
