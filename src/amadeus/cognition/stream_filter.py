"""Strips ``<think>...</think>`` reasoning blocks from a token stream.

Reasoning models (e.g. Qwen 3) emit their internal monologue inside think
tags. Tags can be split across streamed chunks, so this filter is
stateful: feed it chunks, it emits only user-visible text.
"""

from __future__ import annotations

import re


class ThinkTagFilter:
    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self) -> None:
        self._buffer = ""
        self._inside = False
        self._emitted_visible = False  # used to trim whitespace after a think block

    def feed(self, chunk: str) -> str:
        self._buffer += chunk
        out: list[str] = []
        while True:
            if self._inside:
                idx = self._buffer.find(self.CLOSE)
                if idx == -1:
                    # keep only a potential partial closing tag
                    self._buffer = self._buffer[-(len(self.CLOSE) - 1):]
                    return self._finalize(out)
                self._buffer = self._buffer[idx + len(self.CLOSE):]
                self._inside = False
            else:
                idx = self._buffer.find(self.OPEN)
                if idx == -1:
                    keep = self._partial_prefix_len(self._buffer, self.OPEN)
                    cut = len(self._buffer) - keep
                    out.append(self._buffer[:cut])
                    self._buffer = self._buffer[cut:]
                    return self._finalize(out)
                out.append(self._buffer[:idx])
                self._buffer = self._buffer[idx + len(self.OPEN):]
                self._inside = True

    def flush(self) -> str:
        """Emit whatever remains (e.g. an unclosed partial tag that turned
        out to be ordinary text). Resets the filter."""
        remainder = "" if self._inside else self._buffer
        self._buffer, self._inside = "", False
        return self._trim(remainder)

    # -- helpers --------------------------------------------------------

    def _finalize(self, parts: list[str]) -> str:
        return self._trim("".join(parts))

    def _trim(self, text: str) -> str:
        """Strip leading whitespace until the first visible character has
        been emitted (models often emit '\\n\\n' right after </think>)."""
        if not self._emitted_visible:
            text = text.lstrip()
        if text:
            self._emitted_visible = True
        return text

    @staticmethod
    def _partial_prefix_len(text: str, tag: str) -> int:
        for k in range(min(len(tag) - 1, len(text)), 0, -1):
            if text.endswith(tag[:k]):
                return k
        return 0



class ExpressionTagFilter:
    """Extracts ``[express:name]`` tags from a token stream.

    feed() returns (visible_text, [expression_names]); tags never reach
    the display, the archive-bound reply string built by callers, or TTS.
    Handles tags split across streamed chunks.
    """

    _TAG = re.compile(r"\[express:([a-z0-9_]+)\]")
    _PREFIX = "[express:"

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, chunk: str) -> tuple[str, list[str]]:
        self._buffer += chunk
        names: list[str] = []
        while (match := self._TAG.search(self._buffer)) is not None:
            names.append(match.group(1))
            self._buffer = self._buffer[: match.start()] + self._buffer[match.end():]

        # hold back a possible partial tag at the end of the buffer
        cut = len(self._buffer)
        idx = self._buffer.rfind("[")
        if idx != -1:
            candidate = self._buffer[idx:]
            partial_prefix = self._PREFIX.startswith(candidate)
            open_tag = candidate.startswith(self._PREFIX) and "]" not in candidate
            if (partial_prefix or open_tag) and len(candidate) < 48:
                cut = idx
        out, self._buffer = self._buffer[:cut], self._buffer[cut:]
        return out, names

    def flush(self) -> str:
        out, self._buffer = self._buffer, ""
        return out
