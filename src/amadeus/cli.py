"""Amadeus entry points.

Run with:  python -m amadeus         (web UI, default)
           python -m amadeus chat    (terminal chat)
           python -m amadeus reset   (erase ALL memory and start fresh)
           python -m amadeus journal (read Amadeus's reflections)

Commands inside the chat:
  /recall <query>   show what memory retrieval returns (with score breakdown)
  /quit             end the session
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx

from .app import build_engine
from .config import load_settings


def _print_recall(engine, query: str) -> None:
    results = engine.retriever.retrieve(query, touch=False)
    if not results:
        print("  (no memories retrieved)")
        return
    for r in results:
        print(
            f"  {r.score:.3f} (sem {r.semantic:.2f} | kw {r.keyword:.2f} | "
            f"rec {r.recency:.2f} | imp {r.importance:.2f}) "
            f"[{r.memory.type.value}] {r.memory.content}"
        )


async def chat() -> None:
    settings = load_settings()
    engine = build_engine(settings)
    engine.start_session()
    print(
        f"Amadeus (model: {settings.llm_model} via {settings.llm_provider}, "
        f"db: {settings.db_path})\nType /quit to end the session.\n"
    )
    try:
        while True:
            try:
                user_text = (await asyncio.to_thread(input, "you > ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_text:
                continue
            if user_text.lower() in ("/quit", "/exit"):
                break
            if user_text.startswith("/recall "):
                _print_recall(engine, user_text.removeprefix("/recall "))
                continue

            print("amadeus > ", end="", flush=True)
            try:
                async for token in engine.respond(user_text):
                    print(token, end="", flush=True)
            except httpx.ConnectError:
                print(
                    f"\n[can't reach Ollama at {settings.ollama_base_url} — "
                    f"is it running? Try `ollama serve`, and make sure the model "
                    f"is pulled: `ollama pull {settings.llm_model}`]"
                )
                continue
            print("\n")
    finally:
        print("Consolidating memories...", flush=True)
        try:
            await engine.close_session()
            print("Session ended. Memories consolidated.")
        except Exception:  # noqa: BLE001
            print("Session ended (consolidation will be retried next launch).")


def run_ui() -> None:
    import threading
    import webbrowser

    import uvicorn

    from .app import build_voice_runtime
    from .server import create_app

    settings = load_settings()
    engine = build_engine(settings)
    from . import avatar as avatar_module

    voice_factory, tts_loader = build_voice_runtime(settings)
    engine.expression_source = lambda: avatar_module.expression_names(settings)
    app = create_app(
        engine,
        model_label=settings.llm_model,
        voice_factory=voice_factory,
        tts_loader=tts_loader if settings.speak_replies else None,
        avatar_info=lambda: avatar_module.avatar_info(settings),
        vendor_dir=avatar_module.vendor_dir(settings),
        avatar_model_dir=avatar_module.avatar_model_dir(settings),
        settings=settings,
        env_file=Path.cwd() / ".env",
    )
    import socket

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((settings.ui_host, settings.ui_port))
    except OSError:
        print(
            f"ERROR: port {settings.ui_port} is already in use — another Amadeus "
            "server is still running (check your other terminal windows, or run "
            "`taskkill /F /IM python.exe` to stop all Python processes)."
        )
        return
    finally:
        probe.close()

    url = f"http://{settings.ui_host}:{settings.ui_port}"

    def run_in_browser() -> None:
        print(f"Amadeus UI running at {url}  (Ctrl+C to quit)")
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
        uvicorn.run(app, host=settings.ui_host, port=settings.ui_port, log_level="warning")

    if settings.ui_mode != "window":
        run_in_browser()
        return

    try:
        import webview  # pywebview: native window shell
    except Exception:  # noqa: BLE001 - missing backend etc.
        print("[ui] pywebview unavailable — falling back to the browser "
              "(set AMADEUS_UI_MODE=browser to silence this)")
        run_in_browser()
        return

    # server on a background thread; the native window owns the main thread
    server = uvicorn.Server(
        uvicorn.Config(app, host=settings.ui_host, port=settings.ui_port,
                       log_level="warning")
    )
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    print(f"Amadeus running at {url} in a native window (close it to quit)")

    persona_name = engine.persona.name
    try:
        webview.create_window(
            persona_name, url,
            width=1180, height=820, min_size=(760, 560),
            background_color="#000000",
        )
        webview.start()
    except Exception as error:  # noqa: BLE001 - WebView2 runtime problems etc.
        print(f"[ui] native window failed ({error}) — falling back to the browser")
        webbrowser.open(url)
        try:
            while server_thread.is_alive():
                server_thread.join(timeout=1.0)
        except KeyboardInterrupt:
            pass

    # window closed: let in-flight session cleanup (memory consolidation) finish
    print("Window closed — saving memories...")
    server.should_exit = True
    server_thread.join(timeout=180)
    print("Done.")


def reset_memory(*, assume_yes: bool = False) -> bool:
    """Erase the database — all memories, sessions, and transcripts."""
    settings = load_settings()
    if not settings.db_path.exists():
        print(f"Nothing to erase — no database at {settings.db_path}.")
        return False
    if not assume_yes:
        answer = input(
            f"This permanently erases ALL of Amadeus's memory "
            f"({settings.db_path}). Type 'yes' to confirm: "
        ).strip().lower()
        if answer != "yes":
            print("Cancelled. Nothing was deleted.")
            return False
    settings.db_path.unlink()
    print("Memory erased. Amadeus will start completely fresh.")
    return True


def show_journal() -> None:
    from .memory import MemoryType
    from .memory.embeddings import HashingEmbedder
    from .memory.store import MemoryStore

    settings = load_settings()
    if not settings.db_path.exists():
        print("No journal yet — no database found.")
        return
    store = MemoryStore(settings.db_path, HashingEmbedder())
    entries = store.all(MemoryType.JOURNAL)
    if not entries:
        print("No journal entries yet. They're written when sessions are consolidated.")
        return
    for entry in entries:
        stamp = entry.created_at.astimezone().strftime("%Y-%m-%d %H:%M")
        print(f"--- {stamp} ---\n{entry.content}\n")


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    command = args[0] if args else "ui"
    if command == "ui":
        run_ui()
        return 0
    if command == "chat":
        asyncio.run(chat())
        return 0
    if command == "reset":
        reset_memory(assume_yes="--yes" in args)
        return 0
    if command == "journal":
        show_journal()
        return 0
    if command == "avatar-setup":
        from . import avatar as avatar_module

        settings = load_settings()
        print("Downloading Live2D runtime (~1 MB, from official sources)...")
        avatar_module.download_avatar_vendor(settings)
        model_dir = avatar_module.avatar_model_dir(settings)
        model_dir.mkdir(parents=True, exist_ok=True)
        model = avatar_module.find_model3(settings)
        print(f"Runtime ready. Put your Live2D model folder in:\n  {model_dir}")
        if model:
            print(f"Model detected: {model.name} — refresh the browser and she's wearing it.")
        else:
            print("(a folder containing a *.model3.json plus its textures)")
        return 0
    if command == "voice-setup":
        from .voice.providers import download_kokoro_models, download_whisper_model

        settings = load_settings()
        print("Downloading Kokoro TTS models (~330 MB)...")
        download_kokoro_models(settings.data_dir / "models" / "kokoro")
        print(f"Downloading Whisper '{settings.stt_model}' STT model (~460 MB)...")
        download_whisper_model(
            settings.data_dir / "models" / f"whisper-{settings.stt_model}",
            settings.stt_model,
        )
        print("Done. Voice is ready.")
        return 0
    print(f"unknown command: {command!r} (available: ui, chat, reset, journal, voice-setup, avatar-setup)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
