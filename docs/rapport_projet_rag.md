# Rapport Projet - RAG Local Entreprise

## 1. Contexte du projet

Ce projet vise a construire un systeme RAG (Retrieval-Augmented Generation) local capable de repondre a des questions en langage naturel a partir d'un corpus documentaire d'entreprise.

L'objectif est de permettre a un utilisateur de poser une question depuis une interface web ou une API REST, puis d'obtenir une reponse generee a partir de documents internes indexes localement.

Le systeme repose sur :

- un corpus de documents structures par domaines metiers (`RH`, `IT`, `Finance`, `Qualite`, `Securite`)
- une phase d'indexation locale avec embeddings
- un moteur de retrieval semantique
- un LLM local via `Ollama`
- une API REST locale exposee via `FastAPI`
- une interface web chatbot connectee a l'API

---

## 2. Objectifs fonctionnels

### Objectif principal

Permettre la consultation intelligente de documents internes par question/reponse.

### Objectifs secondaires

- organiser les documents selon leur domaine metier
- indexer localement les contenus pour un usage RAG
- exposer le systeme via API REST
- fournir une interface web simple d'utilisation
- rendre l'application partageable via `cloudflared`

---

## 3. Perimetre actuel

### Documents pris en charge

Le projet prend actuellement en charge principalement :

- PDF
- Excel (`.xlsx`, `.xlsm`)

### Domaines metiers couverts

- RH
- IT
- Finance
- Qualite
- Securite

### Fonctions actuellement disponibles

- classement de documents dans `data/raw/<categorie>`
- extraction du texte depuis PDF et Excel
- decoupage en chunks
- calcul des embeddings
- creation d'un index local
- recherche semantique locale
- reponse RAG avec Ollama
- API REST (`/ask`, `/ask/employees`, `/health`)
- interface web chatbot

---

## 4. Architecture actuelle

### 4.1 Stockage documentaire

Les documents sont stockes localement dans :

- `data/raw/RH`
- `data/raw/IT`
- `data/raw/Finance`
- `data/raw/Qualite`
- `data/raw/Securite`

### 4.2 Indexation

Le script principal d'indexation est :

- `src/tools/build_index.py`

Il effectue :

- lecture recursive des fichiers
- extraction du contenu texte
- chunking du contenu
- generation des embeddings avec `sentence-transformers`
- sauvegarde de l'index dans `data/index` ou `data/index_business`

### 4.3 Retrieval

Le moteur de retrieval utilise :

- similarite vectorielle sur embeddings normalises
- `top_k` configurable

Le script de recherche est :

- `src/tools/search_index.py`

### 4.4 Generation

La generation de reponse s'appuie sur :

- `Ollama`
- modele local, par exemple `llama3.1:8b`

Le script principal de generation est :

- `src/tools/answer_rag.py`

### 4.5 API

API FastAPI :

- `src/api/app.py`

Endpoints actuellement disponibles :

- `GET /health`
- `POST /ask`
- `POST /ask/employees`
- `GET /` pour l'interface web

### 4.6 Interface web

Interface chatbot servie localement depuis :

- `src/web/index.html`
- `src/web/styles.css`
- `src/web/app.js`

---

## 5. Etat actuel du projet

### Ce qui est deja fait

- mise en place d'un downloader de documents publics
- constitution d'un corpus local organise par metier
- prise en compte de documents importes manuellement
- creation d'un pipeline d'indexation local
- recherche semantique fonctionnelle
- integration avec Ollama
- creation d'une API REST fonctionnelle
- creation d'une interface web chatbot
- generation d'un jeu de 30 questions de test

### Ce qui fonctionne bien

- systeme 100% local pour l'indexation et la reponse
- bonne pertinence sur les questions simples et ciblees
- reponse claire en ligne de commande
- extraction structuree possible sur certains cas specifiques (`/ask/employees`)

### Points a surveiller

- certaines reponses API peuvent rester trop verbeuses selon le prompt
- pas encore de systeme de permissions / authentification
- pas encore de filtrage automatique par categorie metier
- pas de mesure automatique de la precision
- pas de reranking avance
- pas de gestion enterprise-grade (monitoring, audit, gouvernance)

---

## 6. Limites actuelles

### Limites techniques

- retrieval uniquement vectoriel
- `top_k` fixe ou manuel
- pas de reranker
- embeddings recharges de maniere simple
- journalisation encore basique

### Limites fonctionnelles

- pas d'authentification utilisateur
- pas de gestion de profils / roles
- pas de suivi des conversations
- pas de workflows metier automatises

### Limites de production

- deploiement mono-machine
- dependance a la machine locale pour API + Ollama + index
- pas de resilience forte ni de haute disponibilite

---

## 7. Evaluation du systeme

Un fichier de 30 questions de test a ete cree pour evaluer le comportement du systeme :

- `examples/questions_30.json`

### Methode recommandee d'evaluation

Pour chaque question, mesurer :

- exactitude de la reponse
- pertinence des sources utilisees
- niveau d'hallucination
- temps de reponse

### Indicateurs suggeres

- precision des reponses
- precision des sources
- taux d'hallucination
- hit@k sur les bons documents

---

## 8. Cas d'usage possibles

### RH

- lister les employes par equipe
- retrouver les managers
- analyser les heures travaillees
- consulter les absences et conges

### IT

- retrouver les procedures de deploiement
- consulter les runbooks
- verifier les processus de gestion des changements
- consulter les tickets incidents

### Finance

- consulter les budgets IT
- verifier les contrats SLA
- extraire des informations financieres recurrentes

### Qualite

- consulter les KPI mensuels
- analyser uptime, MTTR, succes des deploiements
- suivre la satisfaction client

### Securite

- consulter la politique de securite
- retrouver les procedures d'incident
- analyser les projets Zero Trust et Disaster Recovery

---

## 9. Roadmap recommandee

### Court terme

- nettoyer le corpus et supprimer les fichiers temporaires
- regenerer l'index apres chaque mise a jour documentaire
- ameliorer les prompts selon le type de question
- activer un filtrage automatique par categorie metier

### Moyen terme

- ajouter authentification API
- ajouter logs et supervision
- construire un vrai jeu d'evaluation avec reponses attendues
- ajouter endpoint specialises pour les cas metiers frequents

### Long terme

- retrieval hybride (vectoriel + lexical)
- reranking
- historique conversationnel
- gestion des droits documentaires
- industrialisation du deploiement

---

## 10. Prochaines etapes conseillees

### Priorite 1

- rebuild propre de l'index apres nettoyage du corpus
- ajout du filtrage automatique par categorie
- ajout d'un mode d'evaluation systematique

### Priorite 2

- securiser l'API avant exposition publique
- ajouter une cle API et du rate-limiting
- renforcer la qualite des reponses de l'API

### Priorite 3

- ajouter de nouveaux connecteurs documentaires
- preparer une version plus robuste pour usage public

---

## 11. Fichiers importants du projet

### Backend / outils

- `src/tools/download_corpus.py`
- `src/tools/build_index.py`
- `src/tools/search_index.py`
- `src/tools/answer_rag.py`

### API

- `src/api/app.py`

### Frontend

- `src/web/index.html`
- `src/web/styles.css`
- `src/web/app.js`

### Scripts de lancement

- `scripts/download_corpus.bat`
- `scripts/build_index.bat`
- `scripts/search_index.bat`
- `scripts/answer_rag.bat`
- `scripts/run_api.bat`

### Donnees

- `data/raw/`
- `data/index/`
- `data/index_business/`

### Evaluation

- `examples/questions_30.json`

---

## 12. Conclusion

Le projet est actuellement a un stade de prototype avance / POC fonctionnel.

Il dispose deja des briques essentielles d'un systeme RAG local :

- corpus organise
- indexation locale
- retrieval semantique
- generation locale avec Ollama
- API REST
- interface web

La suite logique consiste a transformer ce prototype en systeme plus fiable, plus precis et plus securise, en travaillant sur :

- la qualite du retrieval
- la qualite des prompts
- l'evaluation systematique
- la securisation de l'API
- la robustesse d'exploitation

---

## 13. Notes personnelles / a completer

A completer plus tard :

- objectifs metier exacts
- liste des utilisateurs cibles
- exigences de securite
- contraintes de performance
- benchmarks de qualite
- choix de deploiement final
