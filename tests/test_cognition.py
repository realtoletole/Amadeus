from datetime import datetime, timedelta, timezone

from amadeus.cognition.keywords import extract_keywords
from amadeus.cognition.prompt import build_system_prompt
from amadeus.cognition.stream_filter import ThinkTagFilter
from amadeus.memory import ConversationArchive, Memory, MemoryType, ScoredMemory
from amadeus.persona import DEFAULT_PERSONA
from amadeus.utils.timefmt import humanize_age


# -- archive -------------------------------------------------------------

def test_archive_session_and_turns_roundtrip(tmp_path):
    archive = ConversationArchive(tmp_path / "a.db")
    session = archive.start_session()
    archive.add_turn(session, "user", "hello")
    archive.add_turn(session, "assistant", "hi there")
    turns = archive.recent_turns(session)
    assert [(t.role, t.content) for t in turns] == [("user", "hello"), ("assistant", "hi there")]


def test_archive_recent_turns_respects_limit_and_order(tmp_path):
    archive = ConversationArchive(tmp_path / "a.db")
    session = archive.start_session()
    for i in range(5):
        archive.add_turn(session, "user", f"msg {i}")
    turns = archive.recent_turns(session, limit=2)
    assert [t.content for t in turns] == ["msg 3", "msg 4"]


def test_last_session_ended_excludes_current(tmp_path):
    archive = ConversationArchive(tmp_path / "a.db")
    first = archive.start_session()
    archive.end_session(first)
    second = archive.start_session()
    assert archive.last_session_ended_at(exclude=second) is not None
    # a brand-new database has no previous session
    fresh = ConversationArchive(tmp_path / "b.db")
    assert fresh.last_session_ended_at() is None


# -- timefmt & keywords ----------------------------------------------------

def test_humanize_age_buckets():
    now = datetime.now(timezone.utc)
    assert humanize_age(now - timedelta(seconds=10), now) == "just now"
    assert humanize_age(now - timedelta(minutes=5), now) == "5 minutes ago"
    assert humanize_age(now - timedelta(hours=1), now) == "1 hour ago"
    assert humanize_age(now - timedelta(days=3), now) == "3 days ago"
    assert humanize_age(now - timedelta(days=14), now) == "2 weeks ago"
    assert humanize_age(now - timedelta(days=90), now) == "3 months ago"


def test_extract_keywords_filters_stopwords():
    keywords = extract_keywords("I am working on my neuroscience thesis about memory")
    assert "neuroscience" in keywords and "thesis" in keywords
    assert "about" not in keywords and "the" not in keywords


# -- prompt builder --------------------------------------------------------

def _scored(content: str, days_old: int = 3) -> ScoredMemory:
    return ScoredMemory(
        memory=Memory(
            type=MemoryType.SEMANTIC,
            content=content,
            created_at=datetime.now(timezone.utc) - timedelta(days=days_old),
        ),
        score=0.9,
    )


def test_prompt_includes_persona_memories_and_gap():
    prompt = build_system_prompt(
        DEFAULT_PERSONA,
        memories=[_scored("User studies neuroscience.")],
        last_session_ended=datetime.now(timezone.utc) - timedelta(days=2),
    )
    assert "Amadeus" in prompt
    assert "User studies neuroscience." in prompt
    assert "3 days ago" in prompt          # memory age
    assert "2 days ago" in prompt          # session gap
    assert "Current date and time" in prompt


def test_prompt_first_conversation_has_no_false_history():
    prompt = build_system_prompt(DEFAULT_PERSONA, memories=[], last_session_ended=None)
    assert "first conversation" in prompt
    assert "Memories relevant" not in prompt


# -- think-tag filter --------------------------------------------------------

def _run_filter(chunks: list[str]) -> str:
    filt = ThinkTagFilter()
    out = "".join(filt.feed(c) for c in chunks)
    return out + filt.flush()


def test_think_filter_strips_block():
    assert _run_filter(["<think>internal monologue</think>\n\nHello!"]) == "Hello!"


def test_think_filter_handles_tags_split_across_chunks():
    chunks = ["<th", "ink>secret ", "reasoning</thi", "nk>\n", "\nVisible answer"]
    assert _run_filter(chunks) == "Visible answer"


def test_think_filter_passes_plain_text_untouched():
    assert _run_filter(["Hello ", "world"]) == "Hello world"


def test_think_filter_flushes_false_partial_tag():
    # "<thin" that never becomes a tag must not be swallowed
    assert _run_filter(["a < b and <thin"]) == "a < b and <thin"


def test_prompt_with_no_memories_forbids_invention():
    prompt = build_system_prompt(
        DEFAULT_PERSONA,
        memories=[],
        last_session_ended=datetime.now(timezone.utc) - timedelta(days=1),
    )
    assert "NO stored memories" in prompt
    assert "Do not invent" in prompt


def test_prompt_with_memories_states_closed_world():
    prompt = build_system_prompt(
        DEFAULT_PERSONA,
        memories=[_scored("User studies neuroscience.")],
        last_session_ended=None,
    )
    assert "ONLY knowledge" in prompt


# -- expression tag filter -----------------------------------------------------

def _run_expr(chunks):
    from amadeus.cognition.stream_filter import ExpressionTagFilter

    filt = ExpressionTagFilter()
    text, names = "", []
    for chunk in chunks:
        visible, found = filt.feed(chunk)
        text += visible
        names += found
    return text + filt.flush(), names


def test_expression_tag_extracted_and_stripped():
    text, names = _run_expr(["That's fascinating! [express:starry_eyes] Tell me more."])
    assert names == ["starry_eyes"]
    assert text == "That's fascinating!  Tell me more."


def test_expression_tag_split_across_chunks():
    text, names = _run_expr(["Oh no. [exp", "ress:gl", "oom] That's rough."])
    assert names == ["gloom"]
    assert "[" not in text and "gloom" not in text


def test_plain_brackets_pass_through():
    text, names = _run_expr(["math: f[x] = 2x [not a tag", "]"])
    assert names == []
    assert text == "math: f[x] = 2x [not a tag]"


def test_prompt_lists_expressions_when_available():
    prompt = build_system_prompt(
        DEFAULT_PERSONA, memories=[], last_session_ended=None,
        expressions=["blush", "gloom"],
    )
    assert "[express:blush]" in prompt
    assert "gloom" in prompt
    bare = build_system_prompt(DEFAULT_PERSONA, memories=[], last_session_ended=None)
    assert "express" not in bare


def test_ollama_payload_carries_latency_settings():
    from amadeus.llm.base import ChatMessage
    from amadeus.llm.ollama import OllamaProvider

    provider = OllamaProvider(
        "http://x", "qwen3:30b-a3b", think=False, keep_alive="60m", num_ctx=8192
    )
    payload = provider._payload(
        [ChatMessage(role="user", content="hi")], stream=True, temperature=0.7
    )
    assert payload["think"] is False
    assert payload["keep_alive"] == "60m"
    assert payload["options"]["num_ctx"] == 8192
    json_payload = provider._payload(
        [ChatMessage(role="user", content="hi")], stream=False,
        temperature=0.3, json_mode=True,
    )
    assert json_payload["format"] == "json" and json_payload["stream"] is False


def test_no_think_soft_switch_added_for_qwen():
    from amadeus.llm.base import ChatMessage
    from amadeus.llm.ollama import OllamaProvider

    provider = OllamaProvider("http://x", "qwen3:30b-a3b", think=False)
    payload = provider._payload(
        [ChatMessage(role="system", content="sys"), ChatMessage(role="user", content="hi")],
        stream=True, temperature=0.7,
    )
    assert payload["messages"][-1]["content"].endswith("/no_think")
    assert payload["messages"][0]["content"] == "sys"   # system untouched

    thinking = OllamaProvider("http://x", "qwen3:30b-a3b", think=True)
    payload = thinking._payload(
        [ChatMessage(role="user", content="hi")], stream=True, temperature=0.7
    )
    assert "/no_think" not in payload["messages"][-1]["content"]

    non_thinking_variant = OllamaProvider(
        "http://x", "qwen3:30b-a3b-instruct-2507", think=False
    )
    payload = non_thinking_variant._payload(
        [ChatMessage(role="user", content="hi")], stream=True, temperature=0.7
    )
    assert "/no_think" not in payload["messages"][-1]["content"]

    other = OllamaProvider("http://x", "llama3.3", think=False)
    payload = other._payload(
        [ChatMessage(role="user", content="hi")], stream=True, temperature=0.7
    )
    assert "/no_think" not in payload["messages"][-1]["content"]


def test_prompt_includes_response_discipline():
    prompt = build_system_prompt(DEFAULT_PERSONA, memories=[], last_session_ended=None)
    assert "Never narrate" in prompt
