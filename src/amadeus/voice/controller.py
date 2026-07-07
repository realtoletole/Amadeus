"""Voice turn controller.

Orchestrates one voice conversation: mic frames in, events and TTS audio
out. States: LISTENING -> (utterance) -> THINKING -> SPEAKING -> LISTENING.

Barge-in: while SPEAKING, sustained user speech (``barge_in_ms``) cancels
the generation task. Cancellation propagates into the engine's response
generator, whose ``finally`` archives the partial turn as interrupted —
Amadeus knows she was cut off. The speech that interrupted her keeps
accumulating in the detector and becomes the next utterance.

Events emitted (dicts, plus raw ``bytes`` for TTS audio):
  {"type": "state", "value": "listening"|"thinking"|"speaking"}
  {"type": "user_transcript", "text": ...}
  {"type": "token", "text": ...}
  {"type": "tts_begin", "sample_rate": ...}   ... bytes frames ...
  {"type": "done"} | {"type": "interrupted"} | {"type": "error", "text": ...}
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

import numpy as np

from ..cognition.engine import CognitionEngine
from ..cognition.stream_filter import ExpressionTagFilter
from .chunker import SentenceChunker
from .providers import STTProvider, TTSProvider
from .vad import UtteranceDetector

logger = logging.getLogger(__name__)

Emit = Callable[[dict | bytes], Awaitable[None]]

LISTENING, THINKING, SPEAKING = "listening", "thinking", "speaking"


class VoiceTurnController:
    def __init__(
        self,
        *,
        engine: CognitionEngine,
        stt: STTProvider,
        tts: TTSProvider,
        detector: UtteranceDetector,
        emit: Emit,
        barge_in_ms: float = 400.0,
    ) -> None:
        self.engine = engine
        self.stt = stt
        self.tts = tts
        self.detector = detector
        self.emit = emit
        self.barge_in_ms = barge_in_ms
        self.state = LISTENING
        self._task: asyncio.Task | None = None

    async def _set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            await self.emit({"type": "state", "value": state})

    async def feed_audio(self, frame: np.ndarray) -> None:
        """Feed one 512-sample 16 kHz float32 mic frame."""
        utterance = self.detector.feed(frame)

        if self.state == SPEAKING and self.detector.current_speech_ms >= self.barge_in_ms:
            await self.interrupt()

        if utterance is not None and self.state == LISTENING:
            self._task = asyncio.create_task(self._handle_utterance(utterance))

    async def interrupt(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        await self.emit({"type": "interrupted"})
        await self._set_state(LISTENING)

    async def shutdown(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    # -- one full turn -----------------------------------------------------

    def submit_text(self, text: str) -> bool:
        """Typed input during a voice session: same spoken reply, no STT.
        Returns False if a turn is already in flight."""
        if (self._task and not self._task.done()) or not text.strip():
            return False

        async def run() -> None:
            await self._set_state(THINKING)
            try:
                await self._respond(text)
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001
                logger.exception("voice turn failed")
                await self.emit({"type": "error", "text": str(error)})
                await self._set_state(LISTENING)

        self._task = asyncio.create_task(run())
        return True

    async def _handle_utterance(self, audio: np.ndarray) -> None:
        await self._set_state(THINKING)
        try:
            text = await asyncio.to_thread(self.stt.transcribe, audio)
            if not text.strip():
                await self._set_state(LISTENING)
                return
            await self.emit({"type": "user_transcript", "text": text})
            await self._respond(text)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - keep the session alive
            logger.exception("voice turn failed")
            await self.emit({"type": "error", "text": str(error)})
            await self._set_state(LISTENING)

    async def _respond(self, text: str) -> None:
        chunker = SentenceChunker()
        expr = ExpressionTagFilter()
        spoke = False

        async def put(visible: str) -> None:
            nonlocal spoke
            if not visible:
                return
            await self.emit({"type": "token", "text": visible})
            for sentence in chunker.feed(visible):
                spoke = await self._speak(sentence, first=not spoke)

        async for token in self.engine.respond(text):
            visible, names = expr.feed(token)
            for name in names:
                await self.emit({"type": "expression", "name": name})
            await put(visible)
        await put(expr.flush())
        if remainder := chunker.flush():
            spoke = await self._speak(remainder, first=not spoke)

        await self.emit({"type": "done"})
        await self._set_state(LISTENING)

    async def _speak(self, sentence: str, *, first: bool) -> bool:
        audio = await asyncio.to_thread(self.tts.synthesize, sentence)
        if first:
            await self._set_state(SPEAKING)
            await self.emit({"type": "tts_begin", "sample_rate": self.tts.sample_rate})
        pcm16 = np.clip(audio * 32767.0, -32768, 32767).astype("<i2")
        await self.emit(pcm16.tobytes())
        return True
