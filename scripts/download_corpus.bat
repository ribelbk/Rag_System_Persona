@echo off
setlocal

cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Virtualenv introuvable: .venv\Scripts\python.exe
  echo Cree d'abord un environnement: python -m venv .venv
  exit /b 1
)

".venv\Scripts\python.exe" -m src.tools.download_corpus %*
