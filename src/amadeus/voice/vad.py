"""Voice activity detection and utterance segmentation.

:class:`UtteranceDetector` is a pure state machine over 512-sample frames
(32 ms at 16 kHz): it takes any per-frame speech-probability function, so
tests inject a fake while production uses :class:`SileroVAD`. It buffers a
short pre-roll so the first syllable isn't clipped, ignores sub-minimum
blips, and emits the full utterance audio once trailing silence exceeds
``end_ms``. ``current_speech_ms`` exposes how long the person has been
speaking *right now* — that's what barge-in watches while Amadeus talks.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Protocol

import numpy as np

SAMPLE_RATE = 16_000
FRAME_SIZE = 512
FRAME_MS = FRAME_SIZE * 1000 / SAMPLE_RATE  # 32 ms


class VadFn(Protocol):
    def __call__(self, frame: np.ndarray) -> float: ...


class UtteranceDetector:
    def __init__(
        self,
        vad: VadFn,
        *,
        threshold: float = 0.5,
        start_ms: float = 190.0,
        end_ms: float = 700.0,
        min_utterance_ms: float = 300.0,
        pre_roll_ms: float = 220.0,
    ) -> None:
        self._vad = vad
        self._threshold = threshold
        self._start_frames = max(1, round(start_ms / FRAME_MS))
        self._end_frames = max(1, round(end_ms / FRAME_MS))
        self._min_frames = max(1, round(min_utterance_ms / FRAME_MS))
        self._pre_roll: deque[np.ndarray] = deque(maxlen=max(1, round(pre_roll_ms / FRAME_MS)))
        self._reset_utterance()

    def _reset_utterance(self) -> None:
        self._active = False
        self._speech_run = 0
        self._silence_run = 0
        self._frames: list[np.ndarray] = []

    @property
    def current_speech_ms(self) -> float:
        """How long speech has been running uninterrupted (for barge-in)."""
        return self._speech_run * FRAME_MS

    def feed(self, frame: np.ndarray) -> np.ndarray | None:
        """Feed one frame; returns the full utterance when one completes."""
        is_speech = self._vad(frame) >= self._threshold

        if not self._active:
            self._pre_roll.append(frame)
            self._speech_run = self._speech_run + 1 if is_speech else 0
            if self._speech_run >= self._start_frames:
                self._active = True
                self._frames = list(self._pre_roll)
                self._silence_run = 0
            return None

        self._frames.append(frame)
        if is_speech:
            self._speech_run += 1
            self._silence_run = 0
            return None

        self._speech_run = 0
        self._silence_run += 1
        if self._silence_run < self._end_frames:
            return None

        speech_frames = len(self._frames) - self._silence_run
        utterance = (
            np.concatenate(self._frames[: len(self._frames)])
            if speech_frames >= self._min_frames
            else None
        )
        self._pre_roll.clear()
        self._reset_utterance()
        return utterance


class SileroVAD:
    """Per-frame speech probability via Silero (ONNX runtime, no torch)."""

    def __init__(self) -> None:
        from pysilero_vad import SileroVoiceActivityDetector

        self._detector = SileroVoiceActivityDetector()

    def __call__(self, frame: np.ndarray) -> float:
        pcm16 = np.clip(frame * 32768.0, -32768, 32767).astype(np.int16)
        return float(self._detector(pcm16.tobytes()))


def create_detector(
    *, threshold: float, end_ms: float, vad: Callable[[np.ndarray], float] | None = None
) -> UtteranceDetector:
    return UtteranceDetector(vad or SileroVAD(), threshold=threshold, end_ms=end_ms)
