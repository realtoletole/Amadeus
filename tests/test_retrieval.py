from datetime import datetime, timedelta, timezone

from amadeus.config import RetrievalWeights
from amadeus.memory import Memory, MemoryRetriever, MemoryStore, MemoryType
from amadeus.memory.embeddings import HashingEmbedder


def make(tmp_path, **weights):
    store = MemoryStore(tmp_path / "test.db", HashingEmbedder(dim=64))
    retriever = MemoryRetriever(
        store, RetrievalWeights(**weights) if weights else RetrievalWeights()
    )
    return store, retriever


def test_retrieves_relevant_over_irrelevant(tmp_path):
    store, retriever = make(tmp_path)
    relevant = store.add(
        Memory(type=MemoryType.SEMANTIC, content="User is writing a thesis on neuroscience.")
    )
    store.add(Memory(type=MemoryType.SEMANTIC, content="User dislikes cilantro."))
    results = retriever.retrieve("how is the neuroscience thesis going")
    assert results[0].memory.id == relevant.id


def test_working_memory_excluded_by_default(tmp_path):
    store, retriever = make(tmp_path)
    store.add(Memory(type=MemoryType.WORKING, content="scratch neuroscience note"))
    kept = store.add(Memory(type=MemoryType.SEMANTIC, content="neuroscience fact"))
    results = retriever.retrieve("neuroscience")
    assert [r.memory.id for r in results] == [kept.id]


def test_recency_breaks_ties(tmp_path):
    store, retriever = make(tmp_path, semantic=0.0, keyword=0.0, recency=1.0, importance=0.0)
    old = Memory(
        type=MemoryType.EPISODIC,
        content="talked about the weather",
        created_at=datetime.now(timezone.utc) - timedelta(days=30),
    )
    new = Memory(type=MemoryType.EPISODIC, content="talked about the weather again")
    store.add(old)
    store.add(new)
    results = retriever.retrieve("weather")
    assert results[0].memory.id == new.id


def test_importance_breaks_ties(tmp_path):
    store, retriever = make(tmp_path, semantic=0.0, keyword=0.0, recency=0.0, importance=1.0)
    store.add(Memory(type=MemoryType.EPISODIC, content="mentioned a deadline", importance=0.1))
    major = store.add(
        Memory(type=MemoryType.EPISODIC, content="mentioned a deadline for the defense",
               importance=0.9)
    )
    results = retriever.retrieve("deadline")
    assert results[0].memory.id == major.id
    assert results[0].score > 0


def test_retrieval_touches_results(tmp_path):
    store, retriever = make(tmp_path)
    memory = store.add(Memory(type=MemoryType.SEMANTIC, content="user likes astronomy"))
    retriever.retrieve("astronomy")
    assert store.get(memory.id).access_count == 1


def test_score_breakdown_is_exposed(tmp_path):
    store, retriever = make(tmp_path)
    store.add(Memory(type=MemoryType.SEMANTIC, content="user likes astronomy", importance=0.7))
    result = retriever.retrieve("astronomy")[0]
    assert result.importance == 0.7
    assert 0.0 <= result.recency <= 1.0
    assert result.semantic > 0 or result.keyword > 0
