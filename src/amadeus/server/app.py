"""Web UI server.

FastAPI app serving the static frontend and a WebSocket chat endpoint.

WebSocket protocol (JSON messages):
  client -> server:  {"type": "user_message", "text": "..."}
                     {"type": "stop"}
  server -> client:  {"type": "session", "model": "..."}
                     {"type": "token", "text": "..."}
                     {"type": "done"}
                     {"type": "interrupted"}
                     {"type": "error", "text": "..."}

"stop" cancels the in-flight generation; the engine archives the partial
reply flagged ``interrupted`` — the same path Phase 4 voice barge-in uses.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable

import httpx
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..cognition.engine import CognitionEngine
from ..cognition.stream_filter import ExpressionTagFilter
from ..voice.chunker import SentenceChunker
from ..voice.controller import VoiceTurnController
from ..voice.vad import FRAME_SIZE

_STATIC = Path(__file__).parent / "static"

# factory returning (stt, tts, detector_factory); may raise with a
# human-readable message if voice dependencies/models are missing
VoiceFactory = Callable[[], tuple[object, object, Callable[[], object]]]


def create_app(
    engine: CognitionEngine,
    *,
    model_label: str = "",
    voice_factory: VoiceFactory | None = None,
    tts_loader: Callable[[], object] | None = None,
    avatar_info: Callable[[], dict] | None = None,
    vendor_dir: Path | None = None,
    avatar_model_dir: Path | None = None,
    settings=None,
    env_file: Path | None = None,
) -> FastAPI:
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # retry sessions left unconsolidated by a crash or Ollama being down
        recovery = asyncio.create_task(engine.consolidate_pending())
        preload = None
        if tts_loader is not None:
            # load TTS at startup (off-loop) so the first reply can speak
            async def _preload() -> None:
                app.state.tts = await asyncio.to_thread(tts_loader)

            app.state.tts = None
            preload = asyncio.create_task(_preload())
        yield
        recovery.cancel()
        if preload:
            preload.cancel()

    app = FastAPI(title="Amadeus", docs_url=None, redoc_url=None, lifespan=lifespan)

    def session_payload() -> dict:
        from .. import __version__

        return {
            "type": "session",
            "model": model_label,
            "version": __version__,
            "name": engine.persona.name,
            "emotion": engine.emotional_state().model_dump(exclude={"updated_at"}),
        }

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(_STATIC / "index.html")

    @app.get("/avatar.js")
    async def avatar_js() -> FileResponse:
        return FileResponse(_STATIC / "avatar.js", media_type="text/javascript")

    @app.get("/api/avatar")
    async def api_avatar() -> dict:
        return avatar_info() if avatar_info else {"renderer": "parametric"}

    if settings is not None and env_file is not None:
        from .. import settings_store
        from ..persona.loader import persona_path
        from ..voice.providers import KOKORO_VOICES

        @app.get("/api/settings")
        async def get_settings() -> dict:
            installed: list[str] = []
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    response = await client.get(f"{settings.ollama_base_url}/api/tags")
                    installed = sorted(
                        m["name"] for m in response.json().get("models", [])
                    )
            except Exception:  # noqa: BLE001 - Ollama down; dropdown just stays empty
                pass
            return {
                "values": settings_store.current_values(settings),
                "options": {
                    "llm_model": installed,
                    "tts_voice": KOKORO_VOICES,
                    "stt_model": settings_store.STT_SIZES,
                    "stt_device": settings_store.STT_DEVICES,
                },
            }

        @app.post("/api/settings")
        async def post_settings(changes: dict) -> dict:
            try:
                restart = settings_store.update_env_file(env_file, changes)
            except ValueError as error:
                return {"ok": False, "error": str(error)}
            settings_store.apply_hot(settings, changes)
            return {"ok": True, "restart_required": restart}

        @app.get("/api/persona")
        async def get_persona() -> dict:
            from ..persona.loader import load_persona

            path = persona_path(settings)
            if not path.exists():
                load_persona(settings)   # seeds the template on fresh installs
            return {"text": path.read_text(encoding="utf-8")}

        @app.post("/api/persona")
        async def post_persona(body: dict) -> dict:
            persona_path(settings).write_text(str(body.get("text", "")), encoding="utf-8")
            # engine re-reads persona.md on every reply; applies next message
            return {"ok": True, "name": engine.persona.name}

    if vendor_dir is not None and vendor_dir.exists():
        app.mount("/vendor", StaticFiles(directory=vendor_dir), name="vendor")
    if avatar_model_dir is not None and avatar_model_dir.exists():
        app.mount(
            "/avatar-model", StaticFiles(directory=avatar_model_dir), name="avatar-model"
        )

    @app.websocket("/ws/chat")
    async def ws_chat(ws: WebSocket) -> None:
        await ws.accept()
        session_id = engine.start_session()
        await ws.send_json(session_payload())
        task: asyncio.Task | None = None

        async def speak(tts, sentence: str, *, first: bool) -> bool:
            audio = await asyncio.to_thread(tts.synthesize, sentence)
            if first:
                await ws.send_json({"type": "tts_begin", "sample_rate": tts.sample_rate})
            pcm16 = np.clip(audio * 32767.0, -32768, 32767).astype("<i2")
            await ws.send_bytes(pcm16.tobytes())
            return True

        async def generate(text: str) -> None:
            tts = getattr(app.state, "tts", None)
            chunker = SentenceChunker() if tts else None
            expr = ExpressionTagFilter()
            spoke = False

            async def put(visible: str) -> None:
                nonlocal spoke
                if not visible:
                    return
                await ws.send_json({"type": "token", "text": visible})
                if chunker:
                    for sentence in chunker.feed(visible):
                        spoke = await speak(tts, sentence, first=not spoke)

            try:
                async for token in engine.respond(text):
                    visible, names = expr.feed(token)
                    for name in names:
                        await ws.send_json({"type": "expression", "name": name})
                    await put(visible)
                await put(expr.flush())
                if chunker and (remainder := chunker.flush()):
                    spoke = await speak(tts, remainder, first=not spoke)
                await ws.send_json({"type": "done"})
            except httpx.ConnectError:
                await ws.send_json(
                    {
                        "type": "error",
                        "text": "Can't reach Ollama. Start it (`ollama serve`) and "
                        "make sure the model is pulled, then try again.",
                    }
                )

        try:
            while True:
                message = await ws.receive_json()
                kind = message.get("type")
                if kind == "user_message":
                    text = str(message.get("text", "")).strip()
                    if not text:
                        continue
                    if task and not task.done():
                        await ws.send_json(
                            {"type": "error", "text": "Still responding — stop first."}
                        )
                        continue
                    task = asyncio.create_task(generate(text))
                elif kind == "stop" and task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    await ws.send_json({"type": "interrupted"})
        except WebSocketDisconnect:
            if task and not task.done():
                task.cancel()
        finally:
            # deterministic cleanup of THIS connection's session; the socket
            # is already closed, so blocking here costs the user nothing
            await engine.close_session(session_id)

    @app.websocket("/ws/voice")
    async def ws_voice(ws: WebSocket) -> None:
        await ws.accept()
        if voice_factory is None:
            await ws.send_json({"type": "error", "text": "Voice is not configured."})
            await ws.close()
            return
        try:
            # first press loads Whisper/Kokoro (may download models);
            # keep it OFF the event loop so the rest of the app stays alive
            await ws.send_json({"type": "state", "value": "loading models"})
            stt, tts, make_detector = await asyncio.to_thread(voice_factory)
        except Exception as error:  # noqa: BLE001 - report setup problems to the UI
            await ws.send_json({"type": "error", "text": str(error)})
            await ws.close()
            return

        session_id = engine.start_session()

        async def emit(event: dict | bytes) -> None:
            if isinstance(event, bytes):
                await ws.send_bytes(event)
            else:
                await ws.send_json(event)

        controller = VoiceTurnController(
            engine=engine, stt=stt, tts=tts, detector=make_detector(), emit=emit
        )
        await ws.send_json(session_payload())
        await ws.send_json({"type": "state", "value": controller.state})

        pending = np.empty(0, dtype=np.float32)
        try:
            while True:
                message = await ws.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if (data := message.get("bytes")) is not None:
                    chunk = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
                    pending = np.concatenate([pending, chunk])
                    while len(pending) >= FRAME_SIZE:
                        await controller.feed_audio(pending[:FRAME_SIZE])
                        pending = pending[FRAME_SIZE:]
                elif (text := message.get("text")) is not None:
                    import json as _json

                    try:
                        payload = _json.loads(text)
                    except ValueError:
                        continue
                    if payload.get("type") == "user_message":
                        if not controller.submit_text(str(payload.get("text", ""))):
                            await ws.send_json(
                                {"type": "error", "text": "Still responding — stop first."}
                            )
                    elif payload.get("type") == "stop":
                        await controller.interrupt()
        except WebSocketDisconnect:
            pass
        finally:
            await controller.shutdown()
            await engine.close_session(session_id)

    return app
