"""System-prompt assembly.

Per turn: persona + current time + time since last session + retrieved
memories rendered with human-readable ages. (Phase 3 adds an emotional-
state style section here.)
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..memory.models import ScoredMemory
from ..persona.profile import Persona
from ..utils.timefmt import humanize_age


def build_system_prompt(
    persona: Persona,
    *,
    memories: list[ScoredMemory],
    last_session_ended: datetime | None,
    style_hints: str = "",
    expressions: list[str] | None = None,
    now: datetime | None = None,
) -> str:
    now = now or datetime.now(timezone.utc)
    parts: list[str] = [persona.render()]

    time_lines = [f"Current date and time: {now.astimezone().strftime('%A, %B %d, %Y, %H:%M')}"]
    if last_session_ended is not None:
        time_lines.append(
            f"Your last conversation with them ended {humanize_age(last_session_ended, now)}."
        )
    else:
        time_lines.append(
            "This is your very first conversation with this person — you have no "
            "memories of them yet."
        )
    parts.append("\n".join(time_lines))

    if style_hints:
        parts.append(style_hints)

    if expressions:
        parts.append(
            "You have a visible animated avatar. You can show a facial expression "
            "by placing a tag like [express:blush] in your reply at the moment the "
            "feeling occurs. Available expressions: "
            + ", ".join(sorted(expressions))
            + ". Use at most one per reply, only when it genuinely fits the emotion "
            "of the moment, and often none at all. The tag is invisible to the user; "
            "never mention it or describe your expression in words."
        )

    if memories:
        lines = ["Memories relevant to what they just said (from your memory database):"]
        for scored in memories:
            memory = scored.memory
            age = humanize_age(memory.created_at, now)
            lines.append(f"- ({age}, {memory.type.value}) {memory.content}")
        lines.append(
            "These entries and the visible conversation are your ONLY knowledge of "
            "this person. Use them only where they genuinely fit; never recite them "
            "as a list or mention the database itself. Do not claim to remember "
            "anything that is not listed here — if asked about something you have "
            "no memory of, say honestly that you don't remember it."
        )
        parts.append("\n".join(lines))
    else:
        parts.append(
            "You currently have NO stored memories about this person beyond the "
            "visible conversation. You know nothing about them — not their name, "
            "work, interests, or past conversations. Do not invent or imply any "
            "prior knowledge or shared history. If asked what you remember, say "
            "plainly that you don't have memories of them yet."
        )

    parts.append(
        "Response discipline: give the reply itself, directly. Never narrate "
        "your reasoning, planning, or thought process — no 'let me think', no "
        "step-by-step deliberation, no restating the question. In casual "
        "conversation keep replies to a few sentences; go longer only when "
        "the person clearly wants depth."
    )

    return "\n\n".join(parts)
