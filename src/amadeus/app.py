"""Application factory: wire configuration into a ready CognitionEngine."""

from __future__ import annotations

from .cognition.consolidation import Consolidator
from .cognition.engine import CognitionEngine
from .config import Settings
from .llm.ollama import OllamaProvider
from .memory.archive import ConversationArchive
from .memory.embeddings import create_embedder
from .memory.retrieval import MemoryRetriever
from .memory.state import StateStore
from .memory.store import MemoryStore
from .persona.loader import load_persona


def build_voice_runtime(settings: Settings):
    """Returns (voice_factory, tts_loader) sharing one model cache.

    tts_loader is safe to call eagerly at startup: it returns the TTS
    provider, or None (with a printed reason) if voice isn't set up —
    text chat then simply stays silent. voice_factory raises instead,
    because a mic session without models can't do anything useful."""
    cache: dict[str, object] = {}

    def load_tts():
        if "tts" in cache:
            return cache["tts"]
        try:
            from .voice.providers import KokoroTTS
        except ImportError:
            print('[voice] TTS unavailable (pip install -e ".[voice]") — replies stay silent')
            return None
        model_dir = settings.data_dir / "models" / "kokoro"
        if not (model_dir / "kokoro-v1.0.onnx").exists():
            print("[voice] TTS models missing (python -m amadeus voice-setup) — replies stay silent")
            return None
        cache["tts"] = KokoroTTS(model_dir, voice=settings.tts_voice)
        print("[voice] TTS ready — replies will be spoken", flush=True)
        return cache["tts"]

    def voice_factory():
        try:
            from .voice.providers import FasterWhisperSTT
            from .voice.vad import create_detector
        except ImportError as error:
            raise RuntimeError(
                'Voice dependencies missing. Install them with: pip install -e ".[voice]"'
            ) from error

        tts = load_tts()
        if tts is None:
            raise RuntimeError("Kokoro TTS models not found. Run: python -m amadeus voice-setup")
        if "stt" not in cache:
            whisper_dir = settings.data_dir / "models" / f"whisper-{settings.stt_model}"
            cache["stt"] = FasterWhisperSTT(
                settings.stt_model, settings.stt_device, model_dir=whisper_dir
            )
            print("[voice] ready — say something", flush=True)

        def make_detector():
            return create_detector(
                threshold=settings.vad_threshold, end_ms=settings.utterance_end_ms
            )

        return cache["stt"], tts, make_detector

    return voice_factory, load_tts


def build_engine(settings: Settings) -> CognitionEngine:
    settings.ensure_dirs()

    embedder = create_embedder(
        settings.embedding_provider,
        base_url=settings.ollama_base_url,
        model=settings.embedding_model,
        dim=settings.embedding_dim,
    )
    store = MemoryStore(settings.db_path, embedder)
    retriever = MemoryRetriever(
        store,
        settings.retrieval,
        recency_half_life_hours=settings.recency_half_life_hours,
    )
    archive = ConversationArchive(settings.db_path)

    if settings.llm_provider == "ollama":
        provider = OllamaProvider(
            settings.ollama_base_url,
            settings.llm_model,
            think=settings.llm_think,
            keep_alive=settings.llm_keep_alive,
            num_ctx=settings.llm_num_ctx,
        )
    else:  # pragma: no cover - future providers (anthropic) plug in here
        raise ValueError(f"unknown llm provider: {settings.llm_provider!r}")

    return CognitionEngine(
        provider=provider,
        store=store,
        retriever=retriever,
        archive=archive,
        persona=lambda: load_persona(settings),
        state_store=StateStore(settings.db_path),
        consolidator=Consolidator(provider=provider, store=store, archive=archive),
        retrieval_top_k=settings.retrieval_top_k,
    )
