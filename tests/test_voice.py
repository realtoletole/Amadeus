import asyncio

import numpy as np
from fastapi.testclient import TestClient

from amadeus.cognition import CognitionEngine
from amadeus.config import RetrievalWeights
from amadeus.memory import ConversationArchive, MemoryRetriever, MemoryStore
from amadeus.memory.embeddings import HashingEmbedder
from amadeus.persona import DEFAULT_PERSONA
from amadeus.server import create_app
from amadeus.voice import SentenceChunker, UtteranceDetector, VoiceTurnController
from amadeus.voice.vad import FRAME_SIZE

SPEECH = np.full(FRAME_SIZE, 0.9, dtype=np.float32)
SILENCE = np.zeros(FRAME_SIZE, dtype=np.float32)


def fake_vad(frame: np.ndarray) -> float:
    return 1.0 if float(np.abs(frame).max()) > 0.5 else 0.0


# ---------------- sentence chunker ----------------

def test_chunker_emits_complete_sentences():
    chunker = SentenceChunker()
    out = []
    for token in ["Hello there, nice to ", "meet you. How are ", "you doing today? Sti"]:
        out.extend(chunker.feed(token))
    assert out == ["Hello there, nice to meet you.", "How are you doing today?"]
    assert chunker.flush() == "Sti"


def test_chunker_merges_short_fragments():
    chunker = SentenceChunker()
    out = chunker.feed("Ok. That is a much longer sentence to speak aloud. ")
    assert out == ["Ok. That is a much longer sentence to speak aloud."]


# ---------------- utterance detector ----------------

def make_detector(**kw):
    return UtteranceDetector(fake_vad, **kw)


def test_detector_segments_utterance_with_preroll():
    detector = make_detector()
    result = None
    for _ in range(12):
        assert detector.feed(SPEECH) is None
    for _ in range(30):
        result = detector.feed(SILENCE)
        if result is not None:
            break
    assert result is not None
    assert len(result) % FRAME_SIZE == 0
    assert len(result) >= 12 * FRAME_SIZE  # speech + preroll retained


def test_detector_ignores_short_blips():
    detector = make_detector()
    for _ in range(3):                       # below start threshold
        assert detector.feed(SPEECH) is None
    for _ in range(40):
        assert detector.feed(SILENCE) is None


def test_detector_drops_too_short_utterances():
    detector = make_detector(min_utterance_ms=600)
    for _ in range(7):                       # activates, but too short overall
        detector.feed(SPEECH)
    for _ in range(40):
        assert detector.feed(SILENCE) is None


def test_current_speech_ms_tracks_run():
    detector = make_detector()
    for _ in range(5):
        detector.feed(SPEECH)
    assert 150 < detector.current_speech_ms < 200
    detector.feed(SILENCE)
    assert detector.current_speech_ms == 0


# ---------------- turn controller ----------------

class FakeBrain:
    def __init__(self, chunks, delay=0.0):
        self.chunks = chunks
        self.delay = delay

    async def stream_chat(self, messages, *, temperature=0.7):
        for chunk in self.chunks:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield chunk


class FakeSTT:
    def transcribe(self, audio):
        return "hello amadeus"


class FakeTTS:
    sample_rate = 24_000

    def synthesize(self, text):
        return np.full(2400, 0.1, dtype=np.float32)


def make_controller(tmp_path, brain):
    store = MemoryStore(tmp_path / "v.db", HashingEmbedder(dim=64))
    engine = CognitionEngine(
        provider=brain,
        store=store,
        retriever=MemoryRetriever(store, RetrievalWeights()),
        archive=ConversationArchive(tmp_path / "v.db"),
        persona=DEFAULT_PERSONA,
    )
    events: list = []

    async def emit(event):
        events.append(event)

    controller = VoiceTurnController(
        engine=engine,
        stt=FakeSTT(),
        tts=FakeTTS(),
        detector=make_detector(),
        emit=emit,
    )
    return controller, events, engine


async def feed_utterance(controller):
    for _ in range(12):
        await controller.feed_audio(SPEECH)
    for _ in range(30):
        await controller.feed_audio(SILENCE)


async def test_full_voice_turn(tmp_path):
    brain = FakeBrain(["Nice to meet you properly. ", "Say more whenever you like."])
    controller, events, engine = make_controller(tmp_path, brain)
    await feed_utterance(controller)
    assert controller._task is not None
    await controller._task

    kinds = [e["type"] if isinstance(e, dict) else "audio" for e in events]
    assert "user_transcript" in kinds
    assert "tts_begin" in kinds
    assert "audio" in kinds
    assert kinds[-2:] == ["done", "state"]  # ends back in listening
    states = [e["value"] for e in events if isinstance(e, dict) and e["type"] == "state"]
    assert states == ["thinking", "speaking", "listening"]
    transcript = [e for e in events if isinstance(e, dict) and e["type"] == "user_transcript"]
    assert transcript[0]["text"] == "hello amadeus"
    turns = engine.archive.recent_turns(engine.session_id)
    assert turns[0].content == "hello amadeus"
    assert turns[-1].interrupted is False


async def test_barge_in_interrupts_and_archives(tmp_path):
    brain = FakeBrain(
        ["This is the first sentence spoken aloud. "] + ["and it keeps going on "] * 100,
        delay=0.02,
    )
    controller, events, engine = make_controller(tmp_path, brain)
    await feed_utterance(controller)

    for _ in range(200):  # wait until she starts speaking
        if controller.state == "speaking":
            break
        await asyncio.sleep(0.02)
    assert controller.state == "speaking"

    for _ in range(14):  # ~450 ms of sustained speech -> barge-in
        await controller.feed_audio(SPEECH)
        if controller.state == "listening":
            break
    assert controller.state == "listening"
    assert any(isinstance(e, dict) and e["type"] == "interrupted" for e in events)
    turns = engine.archive.recent_turns(engine.session_id)
    assert turns[-1].role == "assistant"
    assert turns[-1].interrupted is True


async def test_empty_transcription_returns_to_listening(tmp_path):
    class SilentSTT:
        def transcribe(self, audio):
            return "   "

    brain = FakeBrain(["should never be used"])
    controller, events, engine = make_controller(tmp_path, brain)
    controller.stt = SilentSTT()
    await feed_utterance(controller)
    await controller._task
    assert controller.state == "listening"
    kinds = [e["type"] for e in events if isinstance(e, dict)]
    assert "user_transcript" not in kinds
    assert "token" not in kinds


# ---------------- server endpoint ----------------

def test_ws_voice_full_turn_over_socket(tmp_path):
    store = MemoryStore(tmp_path / "w.db", HashingEmbedder(dim=64))
    engine = CognitionEngine(
        provider=FakeBrain(["Hello from the other side of the wire."]),
        store=store,
        retriever=MemoryRetriever(store, RetrievalWeights()),
        archive=ConversationArchive(tmp_path / "w.db"),
        persona=DEFAULT_PERSONA,
    )

    def voice_factory():
        return FakeSTT(), FakeTTS(), lambda: make_detector()

    client = TestClient(create_app(engine, model_label="m", voice_factory=voice_factory))
    speech = (SPEECH * 32767).astype("<i2").tobytes()
    silence = (SILENCE * 32767).astype("<i2").tobytes()

    with client.websocket_connect("/ws/voice") as ws:
        assert ws.receive_json() == {"type": "state", "value": "loading models"}
        assert ws.receive_json()["type"] == "session"
        assert ws.receive_json() == {"type": "state", "value": "listening"}
        for _ in range(12):
            ws.send_bytes(speech)
        for _ in range(30):
            ws.send_bytes(silence)
        seen = set()
        got_audio = False
        while "done" not in seen:
            message = ws.receive()
            if message.get("bytes") is not None:
                got_audio = True
                continue
            import json as _json

            payload = _json.loads(message["text"])
            seen.add(payload["type"])
        assert {"user_transcript", "token", "tts_begin", "done"} <= seen
        assert got_audio


def test_ws_voice_reports_missing_setup(tmp_path):
    store = MemoryStore(tmp_path / "x.db", HashingEmbedder(dim=64))
    engine = CognitionEngine(
        provider=FakeBrain(["hi"]),
        store=store,
        retriever=MemoryRetriever(store, RetrievalWeights()),
        archive=ConversationArchive(tmp_path / "x.db"),
        persona=DEFAULT_PERSONA,
    )

    def broken_factory():
        raise RuntimeError("Kokoro TTS models not found. Run: python -m amadeus voice-setup")

    client = TestClient(create_app(engine, voice_factory=broken_factory))
    with client.websocket_connect("/ws/voice") as ws:
        assert ws.receive_json()["type"] == "state"   # loading models
        message = ws.receive_json()
        assert message["type"] == "error"
        assert "voice-setup" in message["text"]


async def test_submit_text_speaks_without_stt(tmp_path):
    brain = FakeBrain(["Typed words, spoken reply, works fine."])
    controller, events, engine = make_controller(tmp_path, brain)
    assert controller.submit_text("hello via keyboard") is True
    await controller._task

    kinds = [e["type"] if isinstance(e, dict) else "audio" for e in events]
    assert "user_transcript" not in kinds     # client already shows typed text
    assert "tts_begin" in kinds and "audio" in kinds and "done" in kinds
    turns = engine.archive.recent_turns(engine.session_id)
    assert turns[0].content == "hello via keyboard"


async def test_submit_text_rejected_while_busy(tmp_path):
    brain = FakeBrain(["Long first sentence that keeps her busy talking. "] * 50, delay=0.02)
    controller, events, engine = make_controller(tmp_path, brain)
    assert controller.submit_text("first message here") is True
    await asyncio.sleep(0.05)
    assert controller.submit_text("second while busy") is False
    controller._task.cancel()
    try:
        await controller._task
    except asyncio.CancelledError:
        pass


def test_ws_voice_accepts_typed_messages(tmp_path):
    store = MemoryStore(tmp_path / "t.db", HashingEmbedder(dim=64))
    engine = CognitionEngine(
        provider=FakeBrain(["A reply that gets spoken out loud properly."]),
        store=store,
        retriever=MemoryRetriever(store, RetrievalWeights()),
        archive=ConversationArchive(tmp_path / "t.db"),
        persona=DEFAULT_PERSONA,
    )

    def voice_factory():
        return FakeSTT(), FakeTTS(), lambda: make_detector()

    client = TestClient(create_app(engine, voice_factory=voice_factory))
    with client.websocket_connect("/ws/voice") as ws:
        ws.receive_json()  # state: loading models
        ws.receive_json()  # session
        ws.receive_json()  # state: listening
        ws.send_json({"type": "user_message", "text": "typed hello"})
        seen, got_audio = set(), False
        while "done" not in seen:
            message = ws.receive()
            if message.get("bytes") is not None:
                got_audio = True
                continue
            import json as _json

            seen.add(_json.loads(message["text"])["type"])
        assert got_audio and "tts_begin" in seen


def test_resolve_whisper_source_prefers_local_files(tmp_path):
    from amadeus.voice.providers import resolve_whisper_source

    assert resolve_whisper_source("small", None) == "small"
    assert resolve_whisper_source("small", tmp_path) == "small"   # dir empty
    (tmp_path / "model.bin").write_bytes(b"x")
    assert resolve_whisper_source("small", tmp_path) == str(tmp_path)
