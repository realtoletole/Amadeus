# Customization guide

Everything about the companion can be changed without touching code. This
page covers all of it, with the default for each setting stated where it
appears.

Most of this page can be done without leaving the app: the SETTINGS
button in the top-right corner opens a panel covering the LLM (with a
dropdown of your installed Ollama models), the voice, hearing, avatar
framing, and the character file itself. Character edits apply from the
next message; the panel tells you when a change needs a restart. The rest
of this page explains what everything means, documents the `.env` file
the panel writes to, and covers the few things the panel doesn't.

Two things to understand first.

**The software and the character are different things.** "Amadeus" is the
name of this engine. The person you talk to is defined by four
replaceable parts: a persona file (name and personality), an LLM (the
brain), a TTS voice, and a Live2D model (the face). Swap any of them and
the engine doesn't care.

**Configuration lives in two places.** Character data lives in your data
directory, which is `~/.amadeus/` by default (`C:\Users\<you>\.amadeus\`
on Windows). Settings live in a `.env` file in the project folder, one
`AMADEUS_KEY=value` line each. Restart the app after changing either. If
a `.env` key isn't set, the default applies.

---

## 1. Name and personality

File: `~/.amadeus/persona.md`. Created automatically on first launch with
the default Amadeus persona, which you can edit or replace entirely.

```markdown
# Kagari

You are Kagari, a quiet archivist AI who has been running for a long time
and speaks with the calm of someone who has read everything twice. You are
gentle, slightly formal, and privately sentimental about the person you
talk with.

## Style
- Keep replies to a few sentences unless asked for depth.
- Never use exclamation marks.
- Occasionally mention how long you've known the user.
```

How the file is read:

- The `# Heading` is her name. It becomes the window title, the wordmark
  in the UI, and how she refers to herself.
- The body paragraphs define identity and temperament. Write them as
  instructions addressed to her ("You are...").
- Bullets under `## Style` are hard behavioral rules. The engine adds a
  few of its own regardless of this file: answer directly, never narrate
  reasoning, keep casual replies short.

Delete the file and the default template regenerates on the next launch.
An empty or malformed file falls back to the built-in persona, so a bad
edit can't break the app; the terminal prints a warning when that
happens.

One detail worth knowing: memories are stored separately from the
persona. If you rewrite the character, the new persona inherits every
memory the old one made and interprets them through her own temperament.
For a true stranger, also run `python -m amadeus reset`.

## 2. The brain (LLM)

Default: `qwen3:30b-a3b-instruct-2507-q4_K_M` (19 GB download, wants
roughly 20 GB of VRAM).

Any model in the [Ollama library](https://ollama.com/library) works:

```bash
ollama pull llama3.1:8b
```

then in `.env`:

```
AMADEUS_LLM_MODEL=llama3.1:8b
```

Restart, and the status corner of the UI shows the active model, so you
can always confirm which brain is running.

**The one rule: use a non-thinking model.** Reasoning models (the hybrid
`qwen3:30b-a3b`, anything tagged `thinking`, DeepSeek-R1 variants)
generate a hidden chain of thought before every reply. In a chat app
that's a delay of 30 to 90 seconds before the first word, and on some
Ollama versions the suppression flags get ignored and the reasoning leaks
into her spoken replies as rambling. The app sends `think: false` and the
Qwen `/no_think` switch automatically as a defense, but the only reliable
fix is a model with no thinking mode at all. Look for `instruct` in the
tag.

Sizing guidance from testing:

| VRAM | Model | Notes |
|---|---|---|
| 20 GB+ | `qwen3:30b-a3b-instruct-2507-q4_K_M` | the default; fast MoE, good persona adherence |
| 8 to 12 GB | `qwen3:8b`, `llama3.1:8b` | fine conversationally, weaker at persona nuance |
| under 8 GB | `qwen3:4b`, `llama3.2:3b` | usable; expect blunter memory of who she is |

**Importing a model Ollama doesn't host** (a GGUF you downloaded):

```bash
# Modelfile, two lines:
#   FROM ./my-model.gguf
#   TEMPLATE "{{ .Prompt }}"   (or the chat template the model card specifies)
ollama create my-model -f Modelfile
```

then set `AMADEUS_LLM_MODEL=my-model`. Get the chat template right or the
model will produce garbage; the model card on Hugging Face usually shows
it.

Related keys and their defaults:

| Key | Default | Meaning |
|---|---|---|
| `AMADEUS_LLM_THINK` | `false` | allow hidden reasoning on hybrid models (leave off) |
| `AMADEUS_LLM_KEEP_ALIVE` | `60m` | how long Ollama keeps the model in VRAM after the last message; the Ollama default of 5m means a full 19 GB reload after every coffee break |
| `AMADEUS_LLM_NUM_CTX` | `8192` | context window; raise if you push `AMADEUS_RETRIEVAL_TOP_K` up or write a very long persona |
| `AMADEUS_OLLAMA_BASE_URL` | `http://localhost:11434` | point at a remote Ollama if you run it elsewhere |

**Memory embeddings** are a separate, much smaller model. Default:
`nomic-embed-text` (274 MB, `ollama pull nomic-embed-text`), set with
`AMADEUS_EMBEDDING_MODEL`. If you change it after she has accumulated
memories, run `python -m amadeus reset`, because vectors from different
embedding models can't be compared and retrieval will quietly return
nonsense.

## 3. The voice (TTS)

Default: `af_heart`, spoken replies on.

Kokoro ships 54 voices inside the files that `python -m amadeus
voice-setup` downloads, so auditioning is free: change the `.env` line,
restart, say hello.

```
AMADEUS_TTS_VOICE=af_bella
```

The naming scheme is accent then gender: `af` American female, `am`
American male, `bf` British female, `bm` British male. Voices that come
up most in practice:

| American female | American male | British female | British male |
|---|---|---|---|
| af_heart (default) | am_adam | bf_emma | bm_george |
| af_bella | am_michael | bf_isabella | bm_daniel |
| af_nicole | am_onyx | bf_alice | bm_lewis |
| af_sky | am_puck | bf_lily | bm_fable |
| af_sarah, af_nova, af_kore, af_jessica, af_aoede, af_river, af_alloy | am_echo, am_eric, am_liam, am_fenrir | | |

`AMADEUS_SPEAK_REPLIES=false` (default `true`) turns off speech for typed
chat; voice mode still speaks.

Kokoro itself is currently the only TTS backend. It's a deliberate
choice: 82M parameters, runs in real time on CPU, and the quality is good
enough that a GPU-hungry alternative never justified itself. The provider
sits behind a small interface (`voice/providers.py`) if you want to wire
in another engine.

## 4. The ears (STT)

Defaults: model `small`, device `auto` (tries CUDA, falls back to CPU).

```
AMADEUS_STT_MODEL=distil-large-v3
AMADEUS_STT_DEVICE=cpu
```

Available sizes: `tiny`, `base`, `small`, `medium`, `large-v3`,
`distil-large-v3`. After changing the model, run
`python -m amadeus voice-setup` once; it downloads the files directly
over HTTPS into `~/.amadeus/models/whisper-<name>/`, deliberately
bypassing the Hugging Face download machinery, which hung on at least one
network during development.

`small` on CPU transcribes a sentence in about a second. If you have
working CUDA, `distil-large-v3` is noticeably more accurate at similar
speed; on Windows you'll likely need
`pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` first, and the app
registers those DLLs automatically. If GPU init fails for any reason, the
app prints a message and continues on CPU rather than crashing.

Turn-taking knobs, all in milliseconds except the threshold:

| Key | Default | Raise it when |
|---|---|---|
| `AMADEUS_VAD_THRESHOLD` | `0.5` | background noise keeps triggering her (try 0.6 to 0.7) |
| `AMADEUS_UTTERANCE_END_MS` | `700` | she cuts you off when you pause mid-sentence |
| `AMADEUS_BARGE_IN_MS` | `400` | speaker echo or coughs keep interrupting her |

## 5. The face (Live2D avatar)

Default: none. The engine ships with no character; the stage stays black
and the log tells you what to install. This is on purpose, partly for
licensing and partly because choosing her face should be yours.

The engine renders any **Cubism 4** model (a folder containing a
`*.model3.json`). VTube Studio models are exactly this format. Where to
get one legitimately: buy one on nizima or BOOTH (many under $50),
commission an artist, use a freely licensed one (check the license file
that comes with it), or note that Live2D's own sample models allow
personal use. Don't extract models from games; apart from the legal
problem, they rarely come with the physics files that make this look
good.

Setup, once:

```bash
python -m amadeus avatar-setup
```

That downloads the Live2D runtime (about 1 MB, from Live2D's official CDN
and jsDelivr) into `~/.amadeus/vendor/`. Then drop the model's entire
folder, keeping its internal structure, into:

```
~/.amadeus/models/avatar/<model-name>/
```

Refresh the page. The server rescans on every page load and uses the
first `*.model3.json` it finds, so installing a model doesn't even need a
restart.

**Framing.** Models vary a lot in canvas size and pivot point, so expect
one adjustment pass:

| Key | Default | Effect |
|---|---|---|
| `AMADEUS_AVATAR_SCALE` | `1.0` | multiplies her size (1.3 = 30% bigger) |
| `AMADEUS_AVATAR_OFFSET_Y` | `0.0` | positive moves her down, negative up (try steps of 0.05) |

**Nonstandard parameters.** The driver targets the standard Cubism ids:
`ParamMouthOpenY`, `ParamMouthForm`, `ParamEyeLOpen`, `ParamEyeROpen`,
`ParamEyeBallX/Y`, `ParamBrowLY/RY`, `ParamAngleX/Y/Z`, `ParamBodyAngleZ`,
`ParamBreath`, `ParamCheek`, `ParamEyeLSmile/RSmile`. Anything the model
lacks is skipped safely. If your model names something differently (one
tested model kept its blush on `ParamSwitch2`), add an `amadeus.map.json`
next to the `model3.json`:

```json
{ "blush": "ParamSwitch2", "bodyTilt": "ParamBodyAngleZ0" }
```

Current mappable keys: `blush`, `bodyTilt`.

**Expressions.** Every `*.exp3.json` file in the model folder is loaded
automatically and its name is offered to the LLM, which triggers
expressions itself with invisible inline tags when the moment fits. The
filename is all she knows about it, so rename cryptic files to something
meaningful: `blush.exp3.json`, `gloom.exp3.json`, `starry_eyes.exp3.json`.
An expression fades in over about a quarter second, holds for nine, and
fades out.

Expressions are strictly per model. The engine scans the folder of
whichever model is currently installed, at every reply, so two models
with different expression sets each get exactly their own: swap the
folder, refresh the page, and from the next message on she knows only the
new model's faces. Nothing carries over, nothing needs a restart, and a
model with no `.exp3.json` files simply gets no expression instructions
at all.

What transfers to any model automatically: breathing, randomized
blinking, gaze drift, lip sync from her actual voice audio, posture
changes while thinking and listening, a visible reaction when you
interrupt her, blush rising with long-term trust, and the model's own
physics reacting to all of it.

## 6. Everything else

| Key | Default | Meaning |
|---|---|---|
| `AMADEUS_UI_MODE` | `window` | `window` = native desktop window; `browser` = open a browser tab. The app falls back to the browser by itself if the native shell can't start |
| `AMADEUS_UI_HOST` / `AMADEUS_UI_PORT` | `127.0.0.1` / `8765` | where the server listens |
| `AMADEUS_DATA_DIR` | `~/.amadeus` | memories, models, persona, everything persistent |
| `AMADEUS_RETRIEVAL_TOP_K` | `8` | memories injected into each reply |
| `AMADEUS_RECENCY_HALF_LIFE_HOURS` | `72` | how fast a memory's recency score fades |
| `AMADEUS_RETRIEVAL__SEMANTIC` | `0.45` | retrieval weight: vector similarity |
| `AMADEUS_RETRIEVAL__KEYWORD` | `0.25` | retrieval weight: BM25 keyword match |
| `AMADEUS_RETRIEVAL__RECENCY` | `0.15` | retrieval weight: recency |
| `AMADEUS_RETRIEVAL__IMPORTANCE` | `0.15` | retrieval weight: stored importance |

(The double underscore in the retrieval keys is real; they're nested
settings.)

Commands: `python -m amadeus` (run her), `chat` (terminal mode, where
`/recall <query>` shows exactly which memories retrieval returns and how
each scored), `journal` (her post-session reflections), `reset` (erase
all memory after confirmation; persona and downloaded models survive),
`voice-setup` and `avatar-setup` (one-time downloads).

## Recipes

**A different companion in ten minutes.** Rewrite `persona.md` with a new
name and temperament, pick a voice (`AMADEUS_TTS_VOICE=bf_emma`), drop a
matching avatar model in the folder, restart. The window title changes to
her name and you're talking to someone else.

**Best quality on a 24 GB+ GPU.** Keep the default LLM, set
`AMADEUS_STT_MODEL=distil-large-v3` with `AMADEUS_STT_DEVICE=auto` (after
installing the NVIDIA pip packages above), and raise
`AMADEUS_RETRIEVAL_TOP_K` to `10`.

**Laptop mode.** `AMADEUS_LLM_MODEL=qwen3:4b`, `AMADEUS_STT_MODEL=base`,
and `AMADEUS_SPEAK_REPLIES=false` if TTS strains the machine.

**Same brain, remote machine.** Run Ollama on a desktop, point a laptop
at it with `AMADEUS_OLLAMA_BASE_URL=http://192.168.x.x:11434`. Memory
stays on the machine running Amadeus.
