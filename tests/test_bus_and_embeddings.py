import numpy as np
import pytest

from amadeus.events import EventBus, Topic
from amadeus.memory.embeddings import HashingEmbedder


async def test_publish_reaches_all_subscribers():
    bus = EventBus()
    received: list[str] = []

    async def handler_a(payload):
        received.append(f"a:{payload}")

    async def handler_b(payload):
        received.append(f"b:{payload}")

    bus.subscribe(Topic.USER_MESSAGE, handler_a)
    bus.subscribe(Topic.USER_MESSAGE, handler_b)
    await bus.publish(Topic.USER_MESSAGE, "hello")
    assert sorted(received) == ["a:hello", "b:hello"]


async def test_failing_handler_does_not_break_others():
    bus = EventBus()
    received = []

    async def bad(payload):
        raise RuntimeError("boom")

    async def good(payload):
        received.append(payload)

    bus.subscribe(Topic.SESSION_ENDED, bad)
    bus.subscribe(Topic.SESSION_ENDED, good)
    await bus.publish(Topic.SESSION_ENDED, 42)
    assert received == [42]


async def test_publish_with_no_subscribers_is_noop():
    await EventBus().publish("nobody.listens", None)


def test_hashing_embedder_is_deterministic_and_normalized():
    embedder = HashingEmbedder(dim=64)
    a1 = embedder.embed(["hello world"])[0]
    a2 = embedder.embed(["hello world"])[0]
    assert np.allclose(a1, a2)
    assert pytest.approx(1.0, abs=1e-5) == float(np.linalg.norm(a1))


def test_hashing_embedder_similarity_tracks_overlap():
    embedder = HashingEmbedder(dim=128)
    base, similar, different = embedder.embed(
        ["time travel machine", "machine for time travel", "banana bread recipe"]
    )
    assert float(base @ similar) > float(base @ different)
