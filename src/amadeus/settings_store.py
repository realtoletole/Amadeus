"""In-app settings persistence.

The UI settings panel reads current values from :class:`Settings` and
writes changes back to the ``.env`` file, preserving comments and any
lines it does not manage. Only whitelisted keys are editable from the UI,
each with a caster that validates the incoming value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .config import Settings


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


# ui key -> (env var, caster, settings attribute, needs restart)
EDITABLE: dict[str, tuple[str, Callable, str, bool]] = {
    "llm_model": ("AMADEUS_LLM_MODEL", str, "llm_model", True),
    "tts_voice": ("AMADEUS_TTS_VOICE", str, "tts_voice", True),
    "speak_replies": ("AMADEUS_SPEAK_REPLIES", _bool, "speak_replies", True),
    "stt_model": ("AMADEUS_STT_MODEL", str, "stt_model", True),
    "stt_device": ("AMADEUS_STT_DEVICE", str, "stt_device", True),
    "vad_threshold": ("AMADEUS_VAD_THRESHOLD", float, "vad_threshold", True),
    "utterance_end_ms": ("AMADEUS_UTTERANCE_END_MS", int, "utterance_end_ms", True),
    "barge_in_ms": ("AMADEUS_BARGE_IN_MS", int, "barge_in_ms", True),
    "avatar_scale": ("AMADEUS_AVATAR_SCALE", float, "avatar_scale", False),
    "avatar_offset_y": ("AMADEUS_AVATAR_OFFSET_Y", float, "avatar_offset_y", False),
}

STT_SIZES = ["tiny", "base", "small", "medium", "large-v3", "distil-large-v3"]
STT_DEVICES = ["auto", "cpu", "cuda"]


def current_values(settings: Settings) -> dict:
    return {key: getattr(settings, attr) for key, (_, _, attr, _) in EDITABLE.items()}


def update_env_file(env_file: Path, changes: dict[str, object]) -> list[str]:
    """Validate ``changes`` against the whitelist, write them to the env
    file (preserving unrelated lines), and return the keys that need a
    restart to take effect. Raises ValueError on unknown keys or bad
    values."""
    validated: dict[str, str] = {}
    restart: list[str] = []
    for key, raw in changes.items():
        if key not in EDITABLE:
            raise ValueError(f"unknown setting: {key!r}")
        env_name, caster, _attr, needs_restart = EDITABLE[key]
        try:
            value = caster(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(f"bad value for {key!r}: {raw!r}") from error
        validated[env_name] = str(value).lower() if isinstance(value, bool) else str(value)
        if needs_restart:
            restart.append(key)

    lines = env_file.read_text(encoding="utf-8").splitlines() if env_file.exists() else []
    remaining = dict(validated)
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        name = stripped.split("=", 1)[0].strip() if "=" in stripped else None
        if name in remaining:
            out.append(f"{name}={remaining.pop(name)}")
        else:
            out.append(line)
    for name, value in remaining.items():
        out.append(f"{name}={value}")
    env_file.write_text("\n".join(out) + "\n", encoding="utf-8")
    return restart


def apply_hot(settings: Settings, changes: dict[str, object]) -> None:
    """Apply restart-free changes to the live Settings object so endpoints
    that read it (e.g. /api/avatar framing) reflect them immediately."""
    for key, raw in changes.items():
        env_name, caster, attr, needs_restart = EDITABLE[key]
        if not needs_restart:
            setattr(settings, attr, caster(raw))
