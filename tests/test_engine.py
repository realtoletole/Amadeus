from amadeus.cognition import CognitionEngine
from amadeus.config import RetrievalWeights
from amadeus.memory import ConversationArchive, Memory, MemoryRetriever, MemoryStore, MemoryType
from amadeus.memory.embeddings import HashingEmbedder
from amadeus.persona import DEFAULT_PERSONA


class FakeProvider:
    """Yields canned chunks; records the messages it was given."""

    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.received = None

    async def stream_chat(self, messages, *, temperature: float = 0.7):
        self.received = messages
        for chunk in self.chunks:
            yield chunk


def make_engine(tmp_path, chunks):
    store = MemoryStore(tmp_path / "e.db", HashingEmbedder(dim=64))
    engine = CognitionEngine(
        provider=FakeProvider(chunks),
        store=store,
        retriever=MemoryRetriever(store, RetrievalWeights()),
        archive=ConversationArchive(tmp_path / "e.db"),
        persona=DEFAULT_PERSONA,
    )
    return engine, store


async def collect(stream):
    return "".join([token async for token in stream])


async def test_full_turn_streams_and_archives(tmp_path):
    engine, store = make_engine(tmp_path, ["<think>hmm</think>", "Hello", " there!"])
    reply = await collect(engine.respond("hi amadeus"))
    assert reply == "Hello there!"

    turns = engine.archive.recent_turns(engine.session_id)
    assert [(t.role, t.content) for t in turns] == [
        ("user", "hi amadeus"),
        ("assistant", "Hello there!"),
    ]
    assert turns[1].interrupted is False
    # short-term capture happened
    short_term = store.all(MemoryType.SHORT_TERM)
    assert len(short_term) == 1 and "hi amadeus" in short_term[0].content


async def test_memories_are_injected_into_system_prompt(tmp_path):
    engine, store = make_engine(tmp_path, ["ok"])
    store.add(Memory(type=MemoryType.SEMANTIC, content="User is writing a neuroscience thesis."))
    await collect(engine.respond("how should I structure my neuroscience thesis?"))
    system = engine.provider.received[0]
    assert system.role == "system"
    assert "neuroscience thesis" in system.content


async def test_history_carries_across_turns(tmp_path):
    engine, _ = make_engine(tmp_path, ["first reply"])
    await collect(engine.respond("turn one"))
    engine.provider.chunks = ["second reply"]
    await collect(engine.respond("turn two"))
    roles = [m.role for m in engine.provider.received]
    contents = [m.content for m in engine.provider.received]
    assert roles == ["system", "user", "assistant", "user"]
    assert contents[1] == "turn one" and contents[2] == "first reply"


async def test_interrupted_stream_is_archived_as_interrupted(tmp_path):
    engine, _ = make_engine(tmp_path, ["Hello", " this", " will", " be", " cut"])
    stream = engine.respond("say something long")
    received = []
    async for token in stream:
        received.append(token)
        if len(received) == 2:
            await stream.aclose()  # consumer bails out mid-stream (barge-in)
            break
    turns = engine.archive.recent_turns(engine.session_id)
    assert turns[-1].role == "assistant"
    assert turns[-1].content == "Hello this"
    assert turns[-1].interrupted is True


async def test_expression_names_follow_the_installed_model(tmp_path):
    """Swapping avatar models mid-run changes what she's told she can
    express, with no restart: the lookup happens at every prompt build."""
    engine, _ = make_engine(tmp_path, ["ok"])
    current = ["starry_eyes", "gloom"]
    engine.expression_source = lambda: list(current)

    await collect(engine.respond("hello there"))
    system = engine.provider.received[0].content
    assert "starry_eyes" in system and "gloom" in system

    current[:] = ["wink", "pout"]          # user swapped the model folder
    engine.provider.chunks = ["ok again"]
    await collect(engine.respond("hello again"))
    system = engine.provider.received[0].content
    assert "wink" in system and "pout" in system
    assert "starry_eyes" not in system     # old model's faces are gone


async def test_no_expression_source_means_no_expression_prompt(tmp_path):
    engine, _ = make_engine(tmp_path, ["ok"])
    await collect(engine.respond("hi"))
    assert "[express:" not in engine.provider.received[0].content
