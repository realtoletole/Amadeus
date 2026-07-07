import asyncio

from fastapi.testclient import TestClient

from amadeus.cognition import CognitionEngine
from amadeus.config import RetrievalWeights
from amadeus.memory import ConversationArchive, MemoryRetriever, MemoryStore
from amadeus.memory.embeddings import HashingEmbedder
from amadeus.persona import DEFAULT_PERSONA
from amadeus.server import create_app


class SlowFakeProvider:
    def __init__(self, chunks: list[str], delay: float = 0.0) -> None:
        self.chunks = chunks
        self.delay = delay

    async def stream_chat(self, messages, *, temperature: float = 0.7):
        for chunk in self.chunks:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield chunk


def make_client(tmp_path, provider):
    store = MemoryStore(tmp_path / "s.db", HashingEmbedder(dim=64))
    engine = CognitionEngine(
        provider=provider,
        store=store,
        retriever=MemoryRetriever(store, RetrievalWeights()),
        archive=ConversationArchive(tmp_path / "s.db"),
        persona=DEFAULT_PERSONA,
    )
    return TestClient(create_app(engine, model_label="test-model")), engine


def test_index_serves_frontend(tmp_path):
    client, _ = make_client(tmp_path, SlowFakeProvider(["hi"]))
    response = client.get("/")
    assert response.status_code == 200
    assert "AMADEUS" in response.text
    assert "ws/chat" in response.text


def test_ws_streams_tokens_and_archives(tmp_path):
    client, engine = make_client(tmp_path, SlowFakeProvider(["Hello", " world"]))
    with client.websocket_connect("/ws/chat") as ws:
        session = ws.receive_json()
        assert session["type"] == "session" and session["model"] == "test-model"
        assert {"mood", "trust", "energy"} <= set(session["emotion"])
        assert session["name"] == "Amadeus"   # persona name reaches the UI
        ws.send_json({"type": "user_message", "text": "hi"})
        tokens: list[str] = []
        while True:
            msg = ws.receive_json()
            if msg["type"] == "done":
                break
            assert msg["type"] == "token"
            tokens.append(msg["text"])
        assert "".join(tokens) == "Hello world"
        turns = engine.archive.recent_turns(engine.session_id)
        assert [(t.role, t.content) for t in turns] == [("user", "hi"), ("assistant", "Hello world")]


def test_ws_stop_interrupts_and_flags_turn(tmp_path):
    provider = SlowFakeProvider([f"chunk{i} " for i in range(200)], delay=0.02)
    client, engine = make_client(tmp_path, provider)
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # session
        ws.send_json({"type": "user_message", "text": "talk forever"})
        first = ws.receive_json()
        assert first["type"] == "token"
        ws.send_json({"type": "stop"})
        while True:
            msg = ws.receive_json()
            if msg["type"] == "interrupted":
                break
            assert msg["type"] == "token"
        session_id = engine.session_id
    turns_client_saw = None  # session ends on disconnect; read archive directly
    archive_turns = engine.archive.recent_turns(session_id)
    last = archive_turns[-1]
    assert last.role == "assistant"
    assert last.interrupted is True
    assert last.content.startswith("chunk0")
    del turns_client_saw


def test_empty_message_is_ignored(tmp_path):
    client, engine = make_client(tmp_path, SlowFakeProvider(["hi"]))
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()
        ws.send_json({"type": "user_message", "text": "   "})
        ws.send_json({"type": "user_message", "text": "real"})
        msg = ws.receive_json()
        assert msg == {"type": "token", "text": "hi"}


class SpeakingFakeTTS:
    sample_rate = 24_000

    def synthesize(self, text):
        import numpy as np

        return np.full(1200, 0.1, dtype=np.float32)


def test_text_chat_replies_are_spoken_when_tts_loaded(tmp_path):
    client, engine = make_client(
        tmp_path, SlowFakeProvider(["A sentence long enough to be spoken aloud. "])
    )
    client.app.router.lifespan_context  # noqa: B018 - app built below instead
    from amadeus.server import create_app

    app = create_app(engine, model_label="m", tts_loader=lambda: SpeakingFakeTTS())
    with TestClient(app) as running:   # context manager runs lifespan (preloads TTS)
        with running.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # session
            ws.send_json({"type": "user_message", "text": "hi"})
            seen, got_audio = set(), False
            while "done" not in seen:
                message = ws.receive()
                if message.get("bytes") is not None:
                    got_audio = True
                    continue
                import json as _json

                seen.add(_json.loads(message["text"])["type"])
            assert "tts_begin" in seen and got_audio


def test_text_chat_stays_silent_without_tts(tmp_path):
    client, engine = make_client(tmp_path, SlowFakeProvider(["Quiet reply, no audio frames."]))
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()
        ws.send_json({"type": "user_message", "text": "hi"})
        while True:
            message = ws.receive()
            assert message.get("bytes") is None
            import json as _json

            payload = _json.loads(message["text"])
            assert payload["type"] != "tts_begin"
            if payload["type"] == "done":
                break


def test_expression_tags_become_events_not_text(tmp_path):
    client, engine = make_client(
        tmp_path,
        SlowFakeProvider(["Interesting! [expr", "ess:starry_eyes] Very much so, truly."]),
    )
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # session
        ws.send_json({"type": "user_message", "text": "hi"})
        text, expressions = "", []
        while True:
            msg = ws.receive_json()
            if msg["type"] == "token":
                text += msg["text"]
            elif msg["type"] == "expression":
                expressions.append(msg["name"])
            elif msg["type"] == "done":
                break
        assert expressions == ["starry_eyes"]
        assert "express" not in text and "[" not in text
