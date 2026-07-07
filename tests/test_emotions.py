from datetime import datetime, timedelta, timezone

from amadeus.cognition.emotions import MAX_DELTA, TRAITS, EmotionalState, render_style_hints
from amadeus.memory import StateStore


def test_decay_pulls_toward_baseline_at_trait_speeds():
    then = datetime.now(timezone.utc) - timedelta(hours=48)
    state = EmotionalState(mood=1.0, trust=0.8, updated_at=then)
    decayed = state.decayed()
    mood_baseline = TRAITS["mood"][0]
    # mood (24h half-life): 48h = two half-lives -> 3/4 of the way home
    assert abs(decayed.mood - (mood_baseline + (1.0 - mood_baseline) * 0.25)) < 1e-6
    # trust (60-day half-life): 48h barely moves it
    assert decayed.trust > 0.78


def test_apply_deltas_clamps_per_delta_and_range():
    state = EmotionalState(mood=0.95, stress=0.05)
    updated = state.apply_deltas({"mood": 0.5, "stress": -0.5, "trust": 0.05})
    assert updated.mood == 1.0                        # +MAX_DELTA then clamped
    assert abs(updated.stress - 0.0) < 1e-9           # -MAX_DELTA then clamped
    assert abs(updated.trust - (TRAITS["trust"][0] + 0.05)) < 1e-9
    assert MAX_DELTA == 0.1


def test_style_hints_reflect_state():
    fresh = EmotionalState()  # default trust is low
    assert "getting to know" in render_style_hints(fresh)

    familiar_tired = EmotionalState(trust=0.8, energy=0.2)
    hints = render_style_hints(familiar_tired)
    assert "familiarity" in hints
    assert "shorter" in hints

    neutral = EmotionalState(trust=0.5, mood=0.5, energy=0.5, curiosity=0.5,
                             confidence=0.5, stress=0.5)
    assert render_style_hints(neutral) == ""


def test_state_store_roundtrip(tmp_path):
    store = StateStore(tmp_path / "s.db")
    assert store.get("emotional_state") is None
    state = EmotionalState(trust=0.4)
    store.set("emotional_state", state.model_dump_json())
    loaded = EmotionalState.model_validate_json(store.get("emotional_state"))
    assert abs(loaded.trust - 0.4) < 1e-9
