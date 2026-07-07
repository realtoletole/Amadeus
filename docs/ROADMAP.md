# Roadmap

## Phase 0 — Foundation ✅
- [x] Repo scaffolding, pyproject, test tooling
- [x] Configuration (`pydantic-settings`, env overrides)
- [x] Async event bus with well-known topics

## Phase 1 — Memory core ✅
- [x] Unified memory model with layers, metadata, emotional valence
- [x] SQLite store: CRUD, FTS5 kept in sync via triggers, embedding BLOBs
- [x] Knowledge-graph link table with typed relations + neighbor queries
- [x] Embedding providers (Ollama + offline hashing fallback)
- [x] Hybrid retrieval (semantic + keyword + recency + importance) with
      score breakdowns and access tracking
- [x] Test suite

## Phase 2 — Cognition loop ✅
- [x] Persona definition (character document, tone rules)
- [x] Prompt builder: persona + time awareness + retrieved memories +
      short-term context
- [x] Session manager (multi-session persistence, transcript archive)
- [x] CLI chat with streaming — first "she remembers yesterday" milestone
- [x] Short-term memory capture during conversation

## Phase 3 — Consolidation & inner life ✅
- [x] Post-session consolidation job (episodic summaries, semantic fact
      extraction, `derived_from` links) with startup recovery of
      unconsolidated sessions
- [x] Contradiction handling (`contradicts` links, importance halving)
- [x] Relationship/profile updater with near-duplicate strengthening
- [x] Journal & reflection writer (`python -m amadeus journal`)
- [x] Emotional state model (homeostatic decay, bounded session deltas,
      prose style hints; trust on a 60-day half-life)
- [x] Short-term cleanup after consolidation (broader LTM pruning: Phase 6)

## Phase 4 — Voice ✅ (logic verified; audio hardware verified on user machine)
- [x] Browser mic capture (echo-cancelled) -> 16 kHz PCM over `/ws/voice`
- [x] Silero VAD (ONNX) + utterance detector state machine (pre-roll,
      blip rejection, min length)
- [x] STT: faster-whisper (auto GPU with CPU fallback)
- [x] Sentence-chunked TTS: Kokoro (final pick; `voice-setup` downloads models)
- [x] Barge-in: sustained speech cancels generation; partial turn archived
      as interrupted; interrupting speech becomes the next utterance
- [ ] Latency polish after real-hardware feedback (streaming synth overlap)
- [x] Spoken replies everywhere: TTS preloads at startup and voices every
      reply, typed or spoken (v0.6.0, `AMADEUS_SPEAK_REPLIES=false` to disable)

## Phase 5 — Desktop UI & avatar (next)
- [x] FastAPI + WebSocket backend (pulled forward, v0.3)
- [x] Chat UI v0 with streaming + stop/interrupt (pulled forward, v0.3;
      vanilla JS, no build step — Vite arrives with the avatar if needed)
- [x] pywebview native window shell (v0.12.0): `python -m amadeus` opens a
      native window titled with the persona's name; graceful shutdown waits
      for memory consolidation; `AMADEUS_UI_MODE=browser` and automatic
      fallback keep the browser path available
- [x] Character name is data end-to-end: window title, wordmark, and tab
      title follow persona.md (v0.12.0)
- [x] Public customization guide: docs/CUSTOMIZATION.md (v0.12.0)
- [x] Avatar v1: parametric renderer — layered animation (breathing,
      randomized blinking, gaze drift, audio-energy lip sync, posture by
      conversation state), expressions driven by the persistent emotional
      state (trust literally shows on her face), interruption reaction;
      renderer behind a stable params interface for Live2D later
- [x] Live2D backend: `avatar-setup` fetches the runtime from official
      sources; drop a `*.model3.json` folder into `<data>/models/avatar/`
      and refresh — the driver maps lip sync, blinks, gaze, emotion, and
      physics onto standard Cubism params; parametric renderer remains the
      instant-on fallback. Framing knobs: `AMADEUS_AVATAR_SCALE`,
      `AMADEUS_AVATAR_OFFSET_Y`.
- [x] Fullscreen UI: character fills the viewport; conversation lives in a
      translucent blurred panel at the bottom (v0.9.0)
- [x] Crash-proof animation loop; reduced-motion detection logged
- [ ] Per-model framing/tuning pass (iterating with user screenshots)
- [x] Expression system (v0.10.0): the LLM places invisible [express:name]
      tags; a stream filter converts them to events (never reaching text,
      archive, or TTS); the frontend crossfades the model's exp3 files with
      weighted blending, auto-clearing after ~9 s
- [ ] VRM backend (`three-vrm`)

## v0.11.0
- [x] Persona is data: character defined in `<data>/persona.md` (seeded on
      first run with the Amadeus template; edit to create any companion)
- [x] Parametric avatar removed; stage is empty until a Live2D model loads,
      with actionable system notices instead of a substitute face

## v0.13.0
- [x] In-app settings panel: gear button opens a modal covering the LLM
      (dropdown of installed Ollama models), voice, hearing, and avatar
      framing; saves write to `.env` preserving unrelated lines and report
      which changes need a restart; avatar framing applies live
- [x] In-app character editor: persona.md editable from the panel and
      re-read at every reply, so edits apply from the next message

## v0.13.1
- [x] The avatar no longer honors the OS reduced-motion setting; Windows'
      "animation effects" toggle can stay off. The override is logged.

## Phase 6 — Long-run polish
- [ ] Time-awareness niceties ("it's been three days")
- [ ] Relationship arc tuning (slow trust growth)
- [ ] Memory pruning & archival policy
- [ ] Personality-consistency evaluation harness
- [ ] `sqlite-vec` migration if memory count warrants it

## Open decisions
- ~~Chat model final pick~~ settled: `qwen3:30b-a3b-instruct-2507-q4_K_M` (non-thinking; see ADR 0001 amendment)
