import pathlib

from amadeus import settings_store
from amadeus.config import Settings


def make_settings(tmp_path) -> Settings:
    return Settings(data_dir=pathlib.Path(tmp_path) / "data")


# -- env file store ------------------------------------------------------------

def test_update_env_preserves_unrelated_lines(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# my notes\nAMADEUS_STT_DEVICE=cpu\nSOME_OTHER_TOOL=1\n")
    restart = settings_store.update_env_file(env, {"tts_voice": "af_bella"})
    lines = env.read_text().splitlines()
    assert "# my notes" in lines
    assert "SOME_OTHER_TOOL=1" in lines
    assert "AMADEUS_TTS_VOICE=af_bella" in lines
    assert "AMADEUS_STT_DEVICE=cpu" in lines
    assert restart == ["tts_voice"]


def test_update_env_replaces_existing_key_in_place(tmp_path):
    env = tmp_path / ".env"
    env.write_text("AMADEUS_LLM_MODEL=old-model\n")
    settings_store.update_env_file(env, {"llm_model": "new-model"})
    text = env.read_text()
    assert "new-model" in text and "old-model" not in text
    assert text.count("AMADEUS_LLM_MODEL") == 1


def test_unknown_key_rejected(tmp_path):
    env = tmp_path / ".env"
    try:
        settings_store.update_env_file(env, {"data_dir": "/tmp/evil"})
        raise AssertionError("should have raised")
    except ValueError as error:
        assert "unknown setting" in str(error)
    assert not env.exists()


def test_bad_value_rejected_and_bool_cast(tmp_path):
    env = tmp_path / ".env"
    try:
        settings_store.update_env_file(env, {"vad_threshold": "loud"})
        raise AssertionError("should have raised")
    except ValueError as error:
        assert "bad value" in str(error)
    settings_store.update_env_file(env, {"speak_replies": "false"})
    assert "AMADEUS_SPEAK_REPLIES=false" in env.read_text()


def test_hot_keys_apply_to_live_settings(tmp_path):
    settings = make_settings(tmp_path)
    settings_store.apply_hot(settings, {"avatar_scale": "1.4", "llm_model": "x"})
    assert settings.avatar_scale == 1.4
    assert settings.llm_model != "x"   # restart-required key untouched


# -- API endpoints -------------------------------------------------------------

def make_client(tmp_path):
    from fastapi.testclient import TestClient

    from amadeus.cognition import CognitionEngine
    from amadeus.config import RetrievalWeights
    from amadeus.memory import ConversationArchive, MemoryRetriever, MemoryStore
    from amadeus.memory.embeddings import HashingEmbedder
    from amadeus.persona.loader import load_persona
    from amadeus.server import create_app

    settings = make_settings(tmp_path)
    settings.ensure_dirs()
    store = MemoryStore(tmp_path / "a.db", HashingEmbedder(dim=64))
    engine = CognitionEngine(
        provider=None,
        store=store,
        retriever=MemoryRetriever(store, RetrievalWeights()),
        archive=ConversationArchive(tmp_path / "a.db"),
        persona=lambda: load_persona(settings),
    )
    env_file = tmp_path / ".env"
    app = create_app(engine, settings=settings, env_file=env_file)
    return TestClient(app), settings, env_file, engine


def test_get_settings_returns_values_and_options(tmp_path):
    client, settings, _, _ = make_client(tmp_path)
    data = client.get("/api/settings").json()
    assert data["values"]["tts_voice"] == settings.tts_voice
    assert "af_heart" in data["options"]["tts_voice"]
    assert "small" in data["options"]["stt_model"]
    assert data["options"]["llm_model"] == []   # no Ollama in tests


def test_post_settings_writes_env_and_reports_restarts(tmp_path):
    client, settings, env_file, _ = make_client(tmp_path)
    result = client.post(
        "/api/settings",
        json={"tts_voice": "bf_emma", "avatar_scale": "1.3"},
    ).json()
    assert result["ok"] is True
    assert result["restart_required"] == ["tts_voice"]
    assert "AMADEUS_TTS_VOICE=bf_emma" in env_file.read_text()
    assert settings.avatar_scale == 1.3   # hot-applied


def test_post_settings_rejects_unknown_key(tmp_path):
    client, _, env_file, _ = make_client(tmp_path)
    result = client.post("/api/settings", json={"evil": "1"}).json()
    assert result["ok"] is False and "unknown" in result["error"]
    assert not env_file.exists()


def test_persona_roundtrip_and_hot_reload(tmp_path):
    client, _, _, engine = make_client(tmp_path)
    original = client.get("/api/persona").json()["text"]
    assert original.startswith("# Amadeus")
    result = client.post(
        "/api/persona",
        json={"text": "# Kagari\n\nYou are Kagari, a quiet archivist.\n"},
    ).json()
    assert result["ok"] is True and result["name"] == "Kagari"
    assert engine.persona.name == "Kagari"   # next reply already uses her
    assert client.get("/api/persona").json()["text"].startswith("# Kagari")
