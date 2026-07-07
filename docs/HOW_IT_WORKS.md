# How it works

A reading guide for anyone evaluating the codebase. Each section names its
files so you can jump straight to the code.

## The turn loop

`cognition/engine.py`. One user message becomes: memory retrieval, prompt
assembly, a streamed LLM response, archival. The response is an async
generator, and if the consumer stops iterating (stop button, voice
barge-in, closed window) a `finally` block archives the partial reply
flagged `interrupted`. Every interruption feature in the project reuses
that one seam.

The prompt (`cognition/prompt.py`) contains the persona from
`persona.md`, the current time and the gap since the last session, an
emotional style paragraph, retrieved memories rendered with plain ages
("3 days ago, semantic: ..."), the expression instructions when an avatar
is installed, and a response-discipline block. When retrieval finds
nothing, the prompt says so outright: "you have NO stored memories about
this person, do not invent any." That closed-world framing came from a
live failure where a 30B local model performed familiarity it did not
have, and it turned out to matter more than any politeness rule.

## Memory

`memory/models.py`, `memory/store.py`, `memory/retrieval.py`. Every
memory shares one schema and differs by type (working, short-term,
episodic, semantic, relationship, profile, journal). Storage is a single
SQLite file: an FTS5 index kept in sync by triggers, embeddings stored as
BLOBs, and a typed link table (`derived_from`, `contradicts`,
`relates_to`) that acts as the knowledge graph. Retrieval takes the union
of vector and keyword candidates and scores each one:

```
score = w_sem * cosine + w_kw * norm_bm25 + w_rec * exp(-age/half_life) + w_imp * importance
```

Results carry their per-channel breakdown, visible in the terminal
client's `/recall` command, and retrieved memories get their access
counters bumped so consolidation can later strengthen what actually gets
used. Vector search is brute-force numpy cosine on purpose: below about
100k memories it costs single-digit milliseconds and zero native
dependencies.

## Consolidation

`cognition/consolidation.py`. On session end, one structured-output LLM
call turns the transcript into episodic summaries, third-person facts,
relationship notes, profile attributes, a first-person journal entry, and
bounded emotional deltas. New facts get checked against existing memory.
A cosine of 0.92 or higher strengthens the original instead of storing a
duplicate. Similar-but-different pairs go to a second batched LLM call,
and a confirmed conflict produces a `contradicts` link plus a halving of
the old fact's importance. Both facts survive, which is what lets her say
"you told me differently before."

The whole job is fail-safe. Sessions carry a `consolidated` flag, so a
crash or an unreachable Ollama leaves the session retryable at the next
launch. Startup recovery is capped at the two most recent pending
sessions, a limit added after an unbounded version queued a whole
migration backlog of 30B jobs in front of the user's first message.

## Emotional state

`cognition/emotions.py`, persisted through `memory/state.py`. Six traits
in [0, 1], each with a baseline and a decay half-life. Mood and energy
reset in about a day. Trust takes about sixty days, and every meaningful
session adds a small guaranteed increment on top of whatever the LLM
reports, so warmth accumulates on a timescale of weeks regardless of
model mood. The state never appears as numbers in the prompt; it renders
as one quiet paragraph ("your energy is low tonight, keep replies a
little shorter") and as avatar parameters: resting mouth curve, blink
rate, blush.

## Voice

`voice/`. The browser captures 16 kHz mono audio in an AudioWorklet and
streams int16 frames over `/ws/voice`. Capture is browser-side for one
reason: `getUserMedia` gives hardware echo cancellation, and without it
her own voice through your speakers would trigger the mic and barge-in
would be unusable. Server-side, `voice/vad.py` is a pure state machine
over 512-sample frames with an injected speech-probability function, so
the segmentation logic (pre-roll so your first syllable isn't clipped,
blip rejection, utterance end after 700 ms of silence) is unit tested
without any audio hardware. `voice/controller.py` runs the turn: Whisper
transcription off the event loop, the engine stream, a sentence chunker
feeding Kokoro so speech starts on the first sentence while the LLM is
still writing the second, and barge-in, where roughly 400 ms of sustained
user speech cancels the generation task. The cancellation lands on the
engine's interrupted seam, and the speech that did the interrupting stays
in the detector and becomes the next utterance.

TTS also preloads at server startup and runs for typed messages, so she
speaks every reply whether or not the microphone is on.

## Avatar

`server/static/avatar.js`, discovery in `avatar.py`. The renderer sits
behind a `render(params)` contract and loads any user-supplied Cubism 4
model, mapping the driver's signals onto standard parameter ids, with a
per-model JSON override for nonstandard ones. No character is bundled.
The persona's name flows from `persona.md` to the window title and the
UI wordmark, because the character is data everywhere, not just in the
prompt.

The driver runs layered animation: a breathing sine, a blink scheduler
with randomized gaps and occasional double-blinks, gaze drift that biases
up-left while she's thinking, posture from conversation state, and mouth
from the RMS energy of an AnalyserNode tapping live TTS playback, so her
lips match exactly what you hear with no extra data over the wire.

Two integration details cost real debugging time and are worth knowing.
First, the Live2D framework's update cycle restores a saved parameter
snapshot every frame, erasing values set before `update()`; parameters
are therefore injected through the motionManager hook, inside the
load/save window, which also lets the model's physics react to the driven
head motion. Second, expressions bypass the library's expression manager
entirely: the app reads the model's `.exp3.json` files itself and
crossfades their parameter lists with weighted blending, triggered by
`[express:name]` tags that the LLM places and a stream filter
(`cognition/stream_filter.py`) strips out of the text, the archive, and
the TTS feed, even when a tag arrives split across streamed chunks.

## Testing

89 tests, none requiring hardware. LLM providers, STT, TTS, and VAD are
replaced by deterministic fakes, and embeddings use a hashing embedder
whose cosine similarity tracks token overlap, which is enough to exercise
retrieval end to end. The suite covers the WebSocket protocol including a
stop mid-stream over a real test socket, the consolidation pipeline
including contradiction handling, the tag filters with tags split across
chunks, and the utterance detector frame by frame.
