"""Wakeword + interrupt-word watcher (polling).

Maintains a rolling audio buffer (last `window_s` seconds) and re-transcribes
it every `poll_s` seconds with the small Whisper model. Keyword match is plain
substring on the lowercased Whisper text — multilingual / Chinese-friendly
since Whisper handles both natively.

Two listening modes:
    "wake"       — listening for any of `wake_words`. on_wake() fires on hit.
    "interrupt"  — listening for any of `interrupt_words`. on_interrupt() fires.

Switch with `set_mode()`.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, List, Optional

import numpy as np

from audio_io import AudioBlock, SAMPLE_RATE
from transcribe import Transcriber


@dataclass
class KeywordWatcher:
    asr: Transcriber
    wake_words: List[str] = field(default_factory=list)
    end_words: List[str] = field(default_factory=list)
    window_s: float = 2.5
    poll_s: float = 0.7
    language: Optional[str] = "zh"
    on_wake: Optional[Callable[[str], Awaitable[None]]] = None
    on_end: Optional[Callable[[str], Awaitable[None]]] = None

    _buffer: List[np.ndarray] = field(default_factory=list, init=False)
    _buffered_samples: int = field(default=0, init=False)
    _max_samples: int = field(default=0, init=False)
    # mode: "wake" listen wake_words; "active" listen interrupt+end words
    _mode: str = field(default="wake", init=False)
    _task: Optional[asyncio.Task] = field(default=None, init=False)
    _last_event_ts: float = field(default=0.0, init=False)
    _cooldown_s: float = field(default=1.5, init=False)
    _last_text: str = field(default="", init=False)

    def __post_init__(self):
        self._max_samples = int(self.window_s * SAMPLE_RATE)

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str):
        # "off" = listen to nothing (used during TTS playback so the
        # assistant's own words don't trip wake / end keywords).
        assert mode in ("wake", "active", "off")
        if mode != self._mode:
            self._mode = mode
            self._buffer.clear()
            self._buffered_samples = 0
            self._last_text = ""

    async def on_block(self, block: AudioBlock):
        self._buffer.append(block.pcm)
        self._buffered_samples += block.pcm.shape[0]
        while self._buffered_samples > self._max_samples and self._buffer:
            head = self._buffer[0]
            if self._buffered_samples - head.shape[0] >= self._max_samples:
                self._buffered_samples -= head.shape[0]
                self._buffer.pop(0)
            else:
                drop = self._buffered_samples - self._max_samples
                self._buffer[0] = head[drop:]
                self._buffered_samples -= drop
                break

    def _snapshot(self) -> Optional[np.ndarray]:
        if self._buffered_samples < int(SAMPLE_RATE * 0.5):
            return None
        return np.concatenate(self._buffer)

    def snapshot_pcm_int16(self) -> bytes:
        """Return the current sliding-window audio as int16 PCM bytes.
        Used by main.py to prepend a pre-roll before the segmenter starts,
        so words spoken right before/with the wake-word aren't lost."""
        if not self._buffer:
            return b""
        audio = np.concatenate(self._buffer)
        i16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        return i16.tobytes()

    def _hit_keywords(self, text: str) -> Optional[tuple[str, str]]:
        """Return (category, keyword) on hit. categories: 'wake' / 'end'."""
        if self._mode == "off":
            return None
        lower = text.lower()
        if self._mode == "wake":
            for kw in self.wake_words:
                if kw and kw.lower() in lower:
                    return ("wake", kw)
            return None
        # active mode: only end words. Voice interrupt was removed because
        # whisper-small on the AEC-cleaned signal can't reliably catch single-
        # / two-character interrupt words; hotkey is the reliable backup.
        for kw in self.end_words:
            if kw and kw.lower() in lower:
                return ("end", kw)
        return None

    async def loop(self):
        while True:
            await asyncio.sleep(self.poll_s)
            if time.monotonic() - self._last_event_ts < self._cooldown_s:
                continue
            audio = self._snapshot()
            if audio is None:
                continue
            rms = float(np.sqrt(np.mean(audio ** 2)))
            if rms < 0.005:
                continue
            try:
                res = await self.asr.transcribe_array(audio, language=self.language, vad_filter=True)
            except Exception:
                continue
            text = res.text.strip()
            if not text or text == self._last_text:
                continue
            self._last_text = text
            hit = self._hit_keywords(text)
            if hit:
                category, kw = hit
                self._last_event_ts = time.monotonic()
                cb = {
                    "wake": self.on_wake,
                    "end": self.on_end,
                }.get(category)
                if cb:
                    try: await cb(text)
                    except Exception: pass

    def start(self):
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.loop())

    async def stop(self):
        if self._task and not self._task.done():
            self._task.cancel()
            try: await self._task
            except (asyncio.CancelledError, Exception): pass
            self._task = None
