"""Central configuration for Amadeus.

All tunables live here and can be overridden via environment variables
prefixed with ``AMADEUS_`` (e.g. ``AMADEUS_LLM_MODEL=qwen3:8b``) or a
``.env`` file in the working directory.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RetrievalWeights(BaseModel):
    """Weights for the hybrid memory-retrieval scorer. Should sum to ~1.0."""

    semantic: float = 0.45
    keyword: float = 0.25
    recency: float = 0.15
    importance: float = 0.15


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AMADEUS_", env_file=".env", env_nested_delimiter="__"
    )

    # Storage
    data_dir: Path = Field(default=Path.home() / ".amadeus")

    # LLM provider ("ollama" now; "anthropic" is a drop-in once an API key exists)
    llm_provider: str = "ollama"
    llm_model: str = "qwen3:30b-a3b-instruct-2507-q4_K_M"
    ollama_base_url: str = "http://localhost:11434"
    llm_think: bool = False           # Qwen3 hidden reasoning: huge latency, off by default
    llm_keep_alive: str = "60m"       # keep the model loaded between messages
    llm_num_ctx: int = 8192           # explicit context window (no silent truncation)

    # Embeddings
    embedding_provider: str = "ollama"  # "ollama" | "hashing" (offline/test fallback)
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 768

    # Voice
    speak_replies: bool = True        # synthesize TTS for every reply, incl. text chat
    stt_model: str = "small"          # faster-whisper model name
    stt_device: str = "auto"          # "auto" | "cuda" | "cpu"
    tts_voice: str = "af_heart"       # kokoro voice id
    vad_threshold: float = 0.5
    utterance_end_ms: int = 700       # silence that ends an utterance
    barge_in_ms: int = 400            # sustained speech that interrupts her

    # Avatar (Live2D framing knobs; only used when a model is installed)
    avatar_scale: float = 1.0
    avatar_offset_y: float = 0.0

    # UI server
    ui_mode: str = "window"           # "window" (native, pywebview) | "browser"
    ui_host: str = "127.0.0.1"
    ui_port: int = 8765

    # Memory / retrieval
    retrieval: RetrievalWeights = RetrievalWeights()
    recency_half_life_hours: float = 72.0
    retrieval_top_k: int = 8

    @property
    def db_path(self) -> Path:
        return self.data_dir / "amadeus.db"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
