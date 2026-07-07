# Amadeus one-shot setup for Windows PowerShell.
# Run from the project folder:  .\setup.ps1
# Safe to re-run; every step skips work it has already done.

$ErrorActionPreference = "Stop"

Write-Host "[1/6] Python virtual environment"
if (-not (Test-Path .venv)) { python -m venv .venv }
.\.venv\Scripts\Activate.ps1

Write-Host "[2/6] Installing Amadeus and dependencies"
python -m pip install -e ".[dev,voice]" -q

Write-Host "[3/6] Checking Ollama"
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "Ollama is not installed. Get it from https://ollama.com/download,"
    Write-Host "then run this script again."
    exit 1
}

Write-Host "[4/6] Pulling models (the LLM is ~19 GB; skipped if present)"
ollama pull qwen3:30b-a3b-instruct-2507-q4_K_M
ollama pull nomic-embed-text

Write-Host "[5/6] Voice models (TTS + STT, one-time download)"
python -m amadeus voice-setup

Write-Host "[6/6] Avatar runtime (optional but tiny)"
python -m amadeus avatar-setup

Write-Host ""
Write-Host "Done. Start her with:  python -m amadeus"
Write-Host "Add a Live2D model to $env:USERPROFILE\.amadeus\models\avatar\ for a face."
Write-Host "Edit $env:USERPROFILE\.amadeus\persona.md to change who she is."
