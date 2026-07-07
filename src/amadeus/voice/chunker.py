"""Sentence chunking for streaming TTS.

Accumulates LLM tokens and emits speakable sentences as soon as they
complete, so synthesis starts on the first sentence while later ones are
still being generated. A minimum length avoids stuttering on fragments
like "Dr." or "Hm."
"""

from __future__ import annotations

import re

_BOUNDARY = re.compile(r"([.!?…]+[\"')\]]*)(\s+|$)")
MIN_SENTENCE_CHARS = 20


class SentenceChunker:
    def __init__(self, *, min_chars: int = MIN_SENTENCE_CHARS) -> None:
        self._buffer = ""
        self._min_chars = min_chars

    def feed(self, token: str) -> list[str]:
        self._buffer += token
        sentences: list[str] = []
        while True:
            match = _BOUNDARY.search(self._buffer)
            if match is None:
                break
            end = match.end(1)
            candidate = self._buffer[:end].strip()
            if len(candidate) < self._min_chars:
                # too short to speak alone (abbreviation, "Ok." etc.) —
                # keep accumulating unless nothing follows yet
                if match.end() >= len(self._buffer):
                    break
                # merge into the next sentence by skipping this boundary
                next_match = _BOUNDARY.search(self._buffer, match.end())
                if next_match is None:
                    break
                end = next_match.end(1)
                candidate = self._buffer[:end].strip()
            sentences.append(candidate)
            self._buffer = self._buffer[end:].lstrip()
        return sentences

    def flush(self) -> str | None:
        remainder = self._buffer.strip()
        self._buffer = ""
        return remainder or None
