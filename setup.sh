#!/usr/bin/env bash
# Amadeus one-shot setup for Linux/macOS. Run: bash setup.sh
# Safe to re-run; every step skips work it has already done.
set -euo pipefail

echo "[1/6] Python virtual environment"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate

echo "[2/6] Installing Amadeus and dependencies"
python -m pip install -e ".[dev,voice]" -q

echo "[3/6] Checking Ollama"
if ! command -v ollama >/dev/null; then
    echo "Ollama is not installed. Get it from https://ollama.com/download,"
    echo "then run this script again."
    exit 1
fi

echo "[4/6] Pulling models (the LLM is ~19 GB; skipped if present)"
ollama pull qwen3:30b-a3b-instruct-2507-q4_K_M
ollama pull nomic-embed-text

echo "[5/6] Voice models (TTS + STT, one-time download)"
python -m amadeus voice-setup

echo "[6/6] Avatar runtime (optional but tiny)"
python -m amadeus avatar-setup

echo
echo "Done. Start her with:  python -m amadeus"
echo "Add a Live2D model to ~/.amadeus/models/avatar/ for a face."
echo "Edit ~/.amadeus/persona.md to change who she is."
