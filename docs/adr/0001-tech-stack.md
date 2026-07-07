# ADR 0001 — Technology stack

Date: 2026-07-04 · Status: accepted

## Context
Personal, non-commercial desktop AI companion. Owner has a Claude Pro
subscription, which does not include Anthropic API access (verified against
Anthropic support docs); API billing is separate. Requirement: no mandatory
recurring costs.

## Decisions
1. **LLM: local via Ollama** (`qwen3:8b` initially) behind an `LLMProvider`
   protocol. Anthropic API is a config-switch drop-in if a key is ever added.
2. **Embeddings: local** (`nomic-embed-text` via Ollama) behind the same
   pattern, with a deterministic hashing embedder so tests run offline.
3. **Storage: single SQLite file.** FTS5 for keywords; embeddings as BLOBs
   with brute-force numpy cosine (fast at personal scale; `sqlite-vec` is
   the upgrade path). Knowledge graph = typed link table, not a graph DB.
4. **UI: Python backend (FastAPI + WebSocket) + web frontend in pywebview.**
   Voice/ML stack is Python; mature Live2D/VRM runtimes are JavaScript;
   pywebview joins them without Electron weight or a Rust toolchain.
   Frontend is plain web tech, so a Tauri migration stays open.
5. **Voice: deferred** (text-first). Bus topics and cancellable streaming
   are designed in now so barge-in bolts on cleanly.
6. **IP policy: no bundled copyrighted assets** — no ripped art/Live2D
   models, no voice clones of the show's actresses, no show dialogue.
   Behavior is specified from the show; assets are user-supplied.

## Consequences
Zero mandatory running costs; everything works offline. Local 8B-class
models are weaker than frontier APIs at persona consistency — mitigated by
strong prompt scaffolding (Phase 2) and the provider seam.

## Amendment (2026-07-05)

The default chat model changed from `qwen3:30b-a3b` to
`qwen3:30b-a3b-instruct-2507-q4_K_M`. The original is a hybrid reasoning
model: it generates a hidden thinking monologue before each reply, which
added 30 to 90 seconds of latency, and on some Ollama versions the
suppression flags were ignored and the reasoning leaked into spoken
replies as plain text. The Instruct-2507 variant has no thinking mode in
its weights, so neither failure is possible. Non-thinking models are now
the recommendation for any voice-driven use of this project.
