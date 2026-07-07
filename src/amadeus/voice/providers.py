"""Speech-to-text and text-to-speech providers.

Thin adapters behind small protocols so the turn controller is testable
with fakes and engines are swappable. Both are blocking calls by design —
the controller runs them via ``asyncio.to_thread``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class STTProvider(Protocol):
    def transcribe(self, audio: np.ndarray) -> str:
        """16 kHz mono float32 -> text."""
        ...


class TTSProvider(Protocol):
    sample_rate: int

    def synthesize(self, text: str) -> np.ndarray:
        """Text -> mono float32 audio at ``sample_rate``."""
        ...


def _register_nvidia_dll_dirs() -> None:
    """On Windows, CUDA runtime DLLs (cuBLAS/cuDNN) can be installed as pip
    packages; ctranslate2 only finds them if their bin dirs are registered."""
    import importlib.util
    import os
    import sys

    if sys.platform != "win32":
        return
    for package in ("nvidia.cublas", "nvidia.cudnn"):
        spec = importlib.util.find_spec(package)
        if spec and spec.submodule_search_locations:
            for location in spec.submodule_search_locations:
                bin_dir = Path(location) / "bin"
                if bin_dir.exists():
                    os.add_dll_directory(str(bin_dir))


def resolve_whisper_source(model: str, model_dir: Path | None) -> str:
    """Prefer a locally downloaded model dir (from `voice-setup`) over the
    HuggingFace hub name, so loading never touches hub download machinery
    (which hangs silently on some setups)."""
    if model_dir is not None and (model_dir / "model.bin").exists():
        return str(model_dir)
    return model


class FasterWhisperSTT:
    def __init__(
        self, model: str = "small", device: str = "auto", model_dir: Path | None = None
    ) -> None:
        import os

        # hf_xet (HuggingFace's download accelerator) hangs silently on some
        # networks; the plain CDN path is reliable. Respect an explicit user
        # setting, otherwise disable it.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from faster_whisper import WhisperModel

        source = resolve_whisper_source(model, model_dir)
        local = source != model
        print(
            f"[voice] loading Whisper '{model}' "
            + ("from local files..." if local else "(first run downloads ~460 MB)..."),
            flush=True,
        )
        model = source

        compute = "float16" if device == "cuda" else "int8"
        try:
            self._model = WhisperModel(model, device=device, compute_type=compute)
        except Exception:
            # CUDA/cuDNN missing or misconfigured -> CPU fallback
            print("[voice] GPU unavailable for Whisper, falling back to CPU", flush=True)
            self._model = WhisperModel(model, device="cpu", compute_type="int8")
        print("[voice] Whisper loaded", flush=True)

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _info = self._model.transcribe(audio, language=None, beam_size=1)
        return " ".join(segment.text.strip() for segment in segments).strip()


class KokoroTTS:
    sample_rate = 24_000

    def __init__(self, model_dir: Path, voice: str = "af_heart") -> None:
        from kokoro_onnx import Kokoro

        print("[voice] loading Kokoro TTS...", flush=True)
        self._kokoro = Kokoro(
            str(model_dir / "kokoro-v1.0.onnx"), str(model_dir / "voices-v1.0.bin")
        )
        self._voice = voice

    def synthesize(self, text: str) -> np.ndarray:
        samples, rate = self._kokoro.create(text, voice=self._voice, speed=1.0)
        assert rate == self.sample_rate
        return np.asarray(samples, dtype=np.float32)


# Whisper (CTranslate2) model files, fetched directly over plain HTTPS so
# `voice-setup` never depends on huggingface_hub's downloader.
WHISPER_REQUIRED = ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"]
WHISPER_OPTIONAL = ["preprocessor_config.json", "vocabulary.json"]


def whisper_repo_url(model: str, filename: str) -> str:
    return f"https://huggingface.co/Systran/faster-whisper-{model}/resolve/main/{filename}"


# English voices shipped in voices-v1.0.bin (a/b = US/UK accent, f/m = gender)
KOKORO_VOICES = [
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
]

KOKORO_FILES = {
    "kokoro-v1.0.onnx":
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    "voices-v1.0.bin":
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
}


def _fetch(client, url: str, target: Path, *, optional: bool = False) -> None:
    if target.exists():
        print(f"  {target.name}: already present")
        return
    print(f"  downloading {target.name} ...")
    with client.stream("GET", url) as response:
        if optional and response.status_code == 404:
            print(f"  {target.name}: not in repo (ok, optional)")
            return
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        written, next_mark = 0, 10
        partial = target.with_suffix(target.suffix + ".part")
        with open(partial, "wb") as fh:
            for chunk in response.iter_bytes(1 << 20):
                fh.write(chunk)
                written += len(chunk)
                if total and written * 100 // total >= next_mark:
                    print(f"    {written * 100 // total}%", flush=True)
                    next_mark += 10
        partial.rename(target)
    print(f"  {target.name}: done")


def download_kokoro_models(model_dir: Path) -> None:
    """Fetch Kokoro model files (~330 MB total) into ``model_dir``."""
    import httpx

    model_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=None) as client:
        for name, url in KOKORO_FILES.items():
            _fetch(client, url, model_dir / name)


def download_whisper_model(model_dir: Path, model: str) -> None:
    """Fetch faster-whisper model files (~460 MB for 'small') over plain
    HTTPS into ``model_dir`` — no huggingface_hub involved."""
    import httpx

    model_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=None) as client:
        for name in WHISPER_REQUIRED:
            _fetch(client, whisper_repo_url(model, name), model_dir / name)
        for name in WHISPER_OPTIONAL:
            _fetch(client, whisper_repo_url(model, name), model_dir / name, optional=True)
