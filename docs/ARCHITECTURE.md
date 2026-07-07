# Architecture

## Overview

Amadeus is six subsystems around an async event bus. Subsystems never import
each other; they publish and subscribe to topics (`amadeus.events.Topic`).
This is what lets voice (Phase 4) and the avatar (Phase 5) attach without
touching the core loop.

```
Desktop UI (pywebview window; web frontend) ── WebSocket ──┐
                                                           │
Voice pipeline (VAD → STT → LLM → TTS, cancellable) ───────┤
                                                           ├── Event Bus
Avatar engine (animation state machine) ───────────────────┤
                                                           │
Cognition (persona, prompt builder, emotional state) ──────┤
                                                           │
Memory system (stores, retrieval, consolidation) ──────────┘
                        │
              SQLite (single file: memories, FTS5, links)
```

## Memory system (implemented, Phase 1)

Identity in the source material is explicitly memory-based, so memory gets
the largest engineering investment. Every memory shares one schema
(`memory/models.py`) and is distinguished by type:

| Type          | Meaning                                        | Lifecycle |
|---------------|------------------------------------------------|-----------|
| working       | current-turn scratch state                     | discarded |
| short_term    | recent conversational context                  | consolidated or decayed |
| episodic      | "what happened" — summarized events            | permanent, prunable |
| semantic      | "what is true" — extracted facts, preferences  | permanent, updatable |
| relationship  | dynamics and facts about the user              | permanent |
| profile       | stable user attributes                         | permanent |
| journal       | Amadeus's first-person post-session reflections| permanent |

Each row carries: timestamp, importance (0–1), emotional valence (−1–1),
keywords, embedding, access statistics, session id, and free-form metadata.
Typed links between memories (`relates_to`, `caused_by`, `about_person`,
`contradicts`, `derived_from`) form the knowledge graph — a table, not a
graph database; at personal scale that is the correct amount of machinery.

### Hybrid retrieval

Candidates = union of vector search and FTS5 keyword search, scored:

```
score = w_sem · cosine + w_kw · norm_bm25 + w_rec · exp(−age/half-life) + w_imp · importance
```

Weights are configuration, results expose their per-channel breakdown for
tuning, and retrieved memories are "touched" (access count / last accessed)
so future consolidation can strengthen frequently-recalled memories —
loosely analogous to reconsolidation in human memory.

Vector search is brute-force cosine over numpy. Deliberate: at ≤10^5
memories it is single-digit milliseconds, has zero native dependencies, and
`sqlite-vec` remains a documented drop-in upgrade.

### Consolidation (implemented)

On `SESSION_ENDED`: an LLM pass summarizes the transcript into episodic
memories, extracts semantic facts (linked `derived_from` their episodes),
updates relationship/profile entries, writes a journal reflection, and
decays or merges stale short-term items. Contradiction between a new fact
and an old one produces a `contradicts` link and an importance re-weighting
rather than silent overwrite — Amadeus should be able to say "you told me
differently before."

## Emotional state (implemented)

A vector of floats (mood, energy, curiosity, confidence, stress, trust) with
homeostatic decay toward a per-trait baseline. Conversation events apply
small deltas; the current state is rendered into the system prompt as *style
guidance*, never as behavioral commands. Trust moves on a much slower
timescale than mood. Design goal: gradual warmth over weeks of interaction,
zero random mood swings.

## Cognition loop (implemented)

Prompt assembly per turn: persona definition + emotional-state style hints +
time awareness (current time, time since last session) + retrieved memories
(formatted with their ages: "three days ago you told me…") + short-term
context + user message. Responses stream token-by-token over the bus.

## LLM and embedding providers

`llm/base.py` defines the provider protocol (async token stream, cancellable
for barge-in). Implementations: Ollama (local, default) and Anthropic
(future, config switch). Embeddings mirror this: Ollama's
`nomic-embed-text` in production, a deterministic hashing embedder for
tests/offline so the suite never needs a model server.

## Voice (implemented)

Audio lives in the browser deliberately: `getUserMedia` provides hardware
echo cancellation, which is what makes barge-in viable over speakers.
Pipeline: browser mic worklet (16 kHz int16 frames over `/ws/voice`) →
Silero VAD (ONNX) → utterance detector state machine (pre-roll, blip
rejection) → faster-whisper STT → engine token stream → sentence chunker →
Kokoro TTS → int16 frames back to a scheduled playback queue.

Barge-in: while SPEAKING, ~400 ms of sustained user speech cancels the
generation task; cancellation propagates into the engine generator whose
`finally` archives the partial turn as interrupted. The interrupting speech
stays in the detector and becomes the next utterance. Voice dependencies
are an optional extra (`pip install -e ".[voice]"`); Kokoro model files are
fetched by `python -m amadeus voice-setup`.

## Avatar (v1 implemented)

`static/avatar.js`: `ParametricAvatar` renders one frame from a flat params
object; `AvatarDriver` runs the layered animation — breathing base loop,
randomized blink scheduler (with double-blinks), gaze micro-movement with
state-dependent bias, lip sync from an AnalyserNode tapping live TTS
playback (mouth matches what you hear), posture from conversation state,
and an expression channel mapped from the persistent emotional state
(trust softens the resting face and blush, energy sets eye openness and
blink rate, curiosity raises brows). Interruptions trigger a visible
reaction. The old presence line lives on inside the avatar scene.

A Live2D/VRM backend only needs to implement `render(params)`, mapping the
same params onto model parameters; user supplies model assets, none are
bundled.

## UI (chat v0 implemented)

FastAPI backend + WebSocket, web frontend (Vite + TypeScript) in a pywebview
native window. Chosen because the ML/voice stack is Python and the mature
avatar runtimes are JavaScript; pywebview joins the two without Electron's
footprint. Migration path to Tauri exists if ever needed since the frontend
is plain web tech.
