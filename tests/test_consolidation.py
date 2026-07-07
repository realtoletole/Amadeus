import json

from amadeus.cognition import CognitionEngine, Consolidator
from amadeus.cognition.emotions import TRAITS
from amadeus.config import RetrievalWeights
from amadeus.memory import (
    ConversationArchive,
    LinkRelation,
    Memory,
    MemoryRetriever,
    MemoryStore,
    MemoryType,
    StateStore,
)
from amadeus.memory.embeddings import HashingEmbedder
from amadeus.persona import DEFAULT_PERSONA


class FakeBrain:
    """LLM double: streams canned chunks, returns queued completions."""

    def __init__(self, completions=None, chunks=("ok",)):
        self.completions = list(completions or [])
        self.chunks = list(chunks)

    async def stream_chat(self, messages, *, temperature=0.7):
        for chunk in self.chunks:
            yield chunk

    async def complete(self, messages, *, temperature=0.3, json_mode=False):
        return self.completions.pop(0)


EXTRACTION = {
    "episodic": [
        {"content": "The user introduced themselves and described their thesis work.",
         "importance": 0.6, "valence": 0.3}
    ],
    "facts": [
        {"content": "The user's name is Rin.", "importance": 0.8, "valence": 0.1},
        {"content": "The user is writing a thesis on neuroscience.", "importance": 0.7},
    ],
    "relationship": [{"content": "The user was open and friendly.", "importance": 0.4}],
    "profile": [{"content": "The user's name is Rin."}],
    "journal": "A first real conversation. They talk about their thesis the way "
               "some people talk about the weather — constantly, and with feeling.",
    "emotional_impact": {"mood": 0.05, "curiosity": 0.08, "trust": 0.03},
}


def make_setup(tmp_path, completions):
    store = MemoryStore(tmp_path / "c.db", HashingEmbedder(dim=64))
    archive = ConversationArchive(tmp_path / "c.db")
    brain = FakeBrain(completions=completions)
    consolidator = Consolidator(provider=brain, store=store, archive=archive)
    return store, archive, consolidator, brain


def start_session_with_turns(archive, turns):
    session = archive.start_session()
    for role, content in turns:
        archive.add_turn(session, role, content)
    archive.end_session(session)
    return session


async def test_consolidation_stores_all_memory_kinds(tmp_path):
    store, archive, consolidator, _ = make_setup(tmp_path, [json.dumps(EXTRACTION)])
    session = start_session_with_turns(
        archive, [("user", "hi, I'm Rin"), ("assistant", "hello"), ("user", "I study brains")]
    )
    store.add(Memory(type=MemoryType.SHORT_TERM, content="The user said: hi", session_id=session))

    result = await consolidator.consolidate(session)

    assert result is not None
    assert len(store.all(MemoryType.EPISODIC)) == 1
    assert len(store.all(MemoryType.SEMANTIC)) == 2
    assert len(store.all(MemoryType.RELATIONSHIP)) == 1
    assert len(store.all(MemoryType.PROFILE)) == 1
    journal = store.all(MemoryType.JOURNAL)
    assert len(journal) == 1 and "thesis" in journal[0].content
    # short-term captures cleaned up; session marked done
    assert store.all(MemoryType.SHORT_TERM) == []
    assert archive.unconsolidated_sessions() == []
    # facts link back to their episode
    episode = store.all(MemoryType.EPISODIC)[0]
    linked = {m.id for m, rel, _ in store.neighbors(episode.id) if rel is LinkRelation.DERIVED_FROM}
    semantic_ids = {m.id for m in store.all(MemoryType.SEMANTIC)}
    assert semantic_ids <= linked
    # trust bump for a meaningful session rides on the LLM's deltas
    assert result.emotional_impact["trust"] > 0.03


async def test_near_duplicate_fact_strengthens_original(tmp_path):
    store, archive, consolidator, _ = make_setup(tmp_path, [json.dumps({
        **EXTRACTION,
        "facts": [{"content": "The user is writing a thesis on neuroscience.", "importance": 0.6}],
        "relationship": [], "profile": [],
    })])
    existing = store.add(Memory(
        type=MemoryType.SEMANTIC,
        content="The user is writing a thesis on neuroscience.",
        importance=0.7,
    ))
    session = start_session_with_turns(archive, [("user", "thesis again"), ("user", "yep")])

    result = await consolidator.consolidate(session)

    assert result.deduplicated == 1
    assert len(store.all(MemoryType.SEMANTIC)) == 1
    assert store.get(existing.id).importance > 0.7


async def test_contradiction_links_and_demotes_old_fact(tmp_path):
    old_content = "The user's favorite drink is coffee."
    new_content = "The user's favorite drink is tea."
    conflict_reply = None  # filled after we know the old memory id

    store, archive, consolidator, brain = make_setup(tmp_path, [json.dumps({
        **EXTRACTION,
        "facts": [{"content": new_content, "importance": 0.7}],
        "relationship": [], "profile": [],
    })])
    old = store.add(Memory(type=MemoryType.SEMANTIC, content=old_content, importance=0.8))
    conflict_reply = json.dumps(
        {"contradictions": [{"new_index": 0, "existing_id": old.id}]}
    )
    brain.completions.append(conflict_reply)
    session = start_session_with_turns(archive, [("user", "actually I prefer tea"), ("user", "!")])

    result = await consolidator.consolidate(session)

    assert result.contradictions == 1
    assert store.get(old.id).importance == 0.4  # halved
    new = [m for m in store.all(MemoryType.SEMANTIC) if m.content == new_content][0]
    relations = {rel for _, rel, _ in store.neighbors(new.id)}
    assert LinkRelation.CONTRADICTS in relations


async def test_unparseable_output_leaves_session_retryable(tmp_path):
    store, archive, consolidator, _ = make_setup(tmp_path, ["definitely not json {"])
    session = start_session_with_turns(archive, [("user", "hello"), ("user", "anyone?")])
    result = await consolidator.consolidate(session)
    assert result is None
    assert archive.unconsolidated_sessions() == [session]


async def test_session_without_user_turns_is_skipped(tmp_path):
    store, archive, consolidator, _ = make_setup(tmp_path, [])
    session = start_session_with_turns(archive, [("assistant", "hello?")])
    assert await consolidator.consolidate(session) is None
    assert archive.unconsolidated_sessions() == []


async def test_engine_close_session_consolidates_and_moves_emotions(tmp_path):
    store = MemoryStore(tmp_path / "e.db", HashingEmbedder(dim=64))
    archive = ConversationArchive(tmp_path / "e.db")
    brain = FakeBrain(completions=[json.dumps(EXTRACTION)], chunks=["nice to meet you"])
    engine = CognitionEngine(
        provider=brain,
        store=store,
        retriever=MemoryRetriever(store, RetrievalWeights()),
        archive=archive,
        persona=DEFAULT_PERSONA,
        state_store=StateStore(tmp_path / "e.db"),
        consolidator=Consolidator(provider=brain, store=store, archive=archive),
    )
    async for _ in engine.respond("hi, I'm Rin"):
        pass
    async for _ in engine.respond("I study brains"):
        pass
    session_id = engine.session_id
    await engine.close_session()

    assert engine.session_id is None
    assert archive.unconsolidated_sessions() == []
    assert len(store.all(MemoryType.EPISODIC)) == 1
    # trust moved above its baseline and persisted
    assert engine.emotional_state().trust > TRAITS["trust"][0]
    del session_id


async def test_consolidate_pending_recovers_crashed_sessions(tmp_path):
    store, archive, consolidator, brain = make_setup(tmp_path, [json.dumps(EXTRACTION)])
    session = start_session_with_turns(archive, [("user", "hi"), ("user", "hello")])
    engine = CognitionEngine(
        provider=brain,
        store=store,
        retriever=MemoryRetriever(store, RetrievalWeights()),
        archive=archive,
        persona=DEFAULT_PERSONA,
        state_store=StateStore(tmp_path / "c.db"),
        consolidator=consolidator,
    )
    assert archive.unconsolidated_sessions() == [session]
    assert await engine.consolidate_pending() == 1
    assert archive.unconsolidated_sessions() == []


def test_archive_migration_adds_consolidated_column(tmp_path):
    import sqlite3

    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, started_at TEXT NOT NULL, ended_at TEXT)")
    conn.execute("INSERT INTO sessions VALUES ('abc', '2026-01-01T00:00:00+00:00', '2026-01-01T01:00:00+00:00')")
    conn.commit()
    conn.close()

    archive = ConversationArchive(db)  # must not raise; must migrate
    assert archive.unconsolidated_sessions() == ["abc"]


async def test_consolidate_pending_skips_backlog_beyond_limit(tmp_path):
    # one canned extraction: only the most recent session should be retried
    store, archive, consolidator, brain = make_setup(tmp_path, [json.dumps(EXTRACTION)])
    old_sessions = [
        start_session_with_turns(archive, [("user", f"old {i}"), ("user", "more")])
        for i in range(4)
    ]
    engine = CognitionEngine(
        provider=brain,
        store=store,
        retriever=MemoryRetriever(store, RetrievalWeights()),
        archive=archive,
        persona=DEFAULT_PERSONA,
        state_store=StateStore(tmp_path / "c.db"),
        consolidator=consolidator,
    )
    done = await engine.consolidate_pending(limit=1)
    assert done == 1                               # only the newest was processed
    assert archive.unconsolidated_sessions() == [] # backlog marked, not queued
    assert brain.completions == []                 # exactly one LLM job consumed
    del old_sessions
