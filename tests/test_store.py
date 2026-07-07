from amadeus.memory import LinkRelation, Memory, MemoryLink, MemoryStore, MemoryType
from amadeus.memory.embeddings import HashingEmbedder


def make_store(tmp_path):
    return MemoryStore(tmp_path / "test.db", HashingEmbedder(dim=64))


def test_add_and_get_roundtrip(tmp_path):
    store = make_store(tmp_path)
    memory = Memory(
        type=MemoryType.SEMANTIC,
        content="The user's favorite drink is Dr Pepper.",
        importance=0.8,
        keywords=["drink", "preference"],
        metadata={"source": "conversation"},
    )
    store.add(memory)
    loaded = store.get(memory.id)
    assert loaded is not None
    assert loaded.content == memory.content
    assert loaded.type is MemoryType.SEMANTIC
    assert loaded.importance == 0.8
    assert loaded.keywords == ["drink", "preference"]
    assert loaded.metadata == {"source": "conversation"}


def test_keyword_search_finds_and_ranks(tmp_path):
    store = make_store(tmp_path)
    store.add(Memory(type=MemoryType.EPISODIC, content="We discussed quantum entanglement."))
    store.add(Memory(type=MemoryType.EPISODIC, content="Talked about lunch plans."))
    hits = store.keyword_search("quantum")
    assert len(hits) == 1


def test_fts_stays_in_sync_on_delete(tmp_path):
    store = make_store(tmp_path)
    memory = store.add(Memory(type=MemoryType.EPISODIC, content="temporary thought"))
    assert store.keyword_search("temporary")
    store.delete(memory.id)
    assert store.keyword_search("temporary") == []


def test_vector_search_orders_by_similarity(tmp_path):
    store = make_store(tmp_path)
    close = store.add(Memory(type=MemoryType.SEMANTIC, content="time machine experiment physics"))
    far = store.add(Memory(type=MemoryType.SEMANTIC, content="banana bread recipe baking"))
    query = store.embedder.embed(["physics experiment with a time machine"])[0]
    results = store.vector_search(query)
    assert results[0][0] == close.id
    ids = [r[0] for r in results]
    assert ids.index(close.id) < ids.index(far.id)


def test_links_and_neighbors(tmp_path):
    store = make_store(tmp_path)
    episode = store.add(Memory(type=MemoryType.EPISODIC, content="User described their thesis."))
    fact = store.add(Memory(type=MemoryType.SEMANTIC, content="User studies neuroscience."))
    store.link(MemoryLink(src_id=fact.id, dst_id=episode.id, relation=LinkRelation.DERIVED_FROM))
    neighbors = store.neighbors(fact.id)
    assert len(neighbors) == 1
    neighbor, relation, weight = neighbors[0]
    assert neighbor.id == episode.id
    assert relation is LinkRelation.DERIVED_FROM
    # undirected view: episode also sees fact
    assert store.neighbors(episode.id)[0][0].id == fact.id


def test_touch_updates_access_stats(tmp_path):
    store = make_store(tmp_path)
    memory = store.add(Memory(type=MemoryType.SEMANTIC, content="fact"))
    store.touch([memory.id])
    loaded = store.get(memory.id)
    assert loaded.access_count == 1
    assert loaded.last_accessed is not None
