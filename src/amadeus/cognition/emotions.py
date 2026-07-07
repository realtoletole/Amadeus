"""Emotional state model.

Six traits in [0, 1] with homeostatic decay toward per-trait baselines.
Session events nudge them by small, bounded deltas; time pulls them home.
Trust decays on a ~60-day half-life so it genuinely accumulates over weeks,
while mood and energy reset within a day — gradual warmth, no mood swings.

The state is rendered into the system prompt as *style hints* (subtle prose
about how she's feeling), never as numbers or behavioral commands.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from pydantic import BaseModel, Field

# trait -> (baseline, decay half-life in hours)
TRAITS: dict[str, tuple[float, float]] = {
    "mood": (0.55, 24.0),
    "energy": (0.60, 24.0),
    "curiosity": (0.65, 72.0),
    "confidence": (0.60, 72.0),
    "stress": (0.30, 24.0),
    "trust": (0.15, 60 * 24.0),
}

MAX_DELTA = 0.1  # per-session cap on any single trait change


class EmotionalState(BaseModel):
    mood: float = Field(default=TRAITS["mood"][0], ge=0.0, le=1.0)
    energy: float = Field(default=TRAITS["energy"][0], ge=0.0, le=1.0)
    curiosity: float = Field(default=TRAITS["curiosity"][0], ge=0.0, le=1.0)
    confidence: float = Field(default=TRAITS["confidence"][0], ge=0.0, le=1.0)
    stress: float = Field(default=TRAITS["stress"][0], ge=0.0, le=1.0)
    trust: float = Field(default=TRAITS["trust"][0], ge=0.0, le=1.0)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def decayed(self, now: datetime | None = None) -> "EmotionalState":
        """Return the state after homeostatic decay toward baselines."""
        now = now or datetime.now(timezone.utc)
        hours = max((now - self.updated_at).total_seconds() / 3600.0, 0.0)
        values = {}
        for trait, (baseline, half_life) in TRAITS.items():
            current = getattr(self, trait)
            factor = math.pow(0.5, hours / half_life)
            values[trait] = baseline + (current - baseline) * factor
        return EmotionalState(**values, updated_at=now)

    def apply_deltas(self, deltas: dict[str, float]) -> "EmotionalState":
        """Apply bounded deltas (each clamped to ±MAX_DELTA), clamp to [0,1]."""
        values = {}
        for trait in TRAITS:
            delta = max(-MAX_DELTA, min(MAX_DELTA, float(deltas.get(trait, 0.0))))
            values[trait] = min(1.0, max(0.0, getattr(self, trait) + delta))
        return EmotionalState(**values, updated_at=datetime.now(timezone.utc))


def render_style_hints(state: EmotionalState) -> str:
    """Turn the state into one quiet paragraph of prose guidance."""
    hints: list[str] = []
    if state.trust < 0.3:
        hints.append("you're still getting to know this person — warm, but a little reserved")
    elif state.trust > 0.6:
        hints.append("you've built real familiarity with them — you can be more open and personal")
    if state.mood > 0.7:
        hints.append("you're in good spirits")
    elif state.mood < 0.35:
        hints.append("you're feeling a bit subdued; it's fine if that shows slightly")
    if state.energy < 0.35:
        hints.append("your energy is low tonight — keep replies a little shorter than usual")
    if state.curiosity > 0.75:
        hints.append("something has your curiosity up; follow interesting threads")
    if state.stress > 0.65:
        hints.append("you're carrying some tension; stay composed")
    if state.confidence < 0.35:
        hints.append("hedge a little more than usual when unsure")
    if not hints:
        return ""
    return (
        "Current internal state (let this color your tone subtly; never mention it "
        "directly): " + "; ".join(hints) + "."
    )
