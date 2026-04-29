"""faster-whisper wrapper.

Two model slots:
  - "monitor"  small  : always-loaded, low-cost, used by the wake-word /
                        interrupt-word monitor on a sliding window.
  - "main"     large-v3-turbo (or any larger): loaded on demand when the
                        wake word fires; transcribes the just-recorded segment
                        for high-quality output to send to claude.

Both run on the same GPU. Total resident VRAM is ~3 GB (small fp16 ~1 GB +
large-v3-turbo fp16 ~1.6 GB), comfortable inside the 5060's 8 GB.
"""
from __future__ import annotations

# IMPORTANT: ensure NVIDIA DLLs are on the search path before importing
# faster_whisper / ctranslate2. Works whether this file is loaded as a script
# (`python transcribe.py`) or as part of a package (`from voice_agent ...`).
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import cuda_setup  # noqa: F401  -- side-effect import

import asyncio
import io
import time
import wave
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from faster_whisper import WhisperModel

DEFAULT_SAMPLE_RATE = 16000


@dataclass
class TranscribeResult:
    text: str
    language: str
    duration_s: float
    elapsed_s: float


def pcm16_to_float32(pcm_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return arr


class Transcriber:
    """Lazy-loaded WhisperModel wrapper. Constructing this is cheap; first
    transcribe() triggers the actual model download/load."""

    def __init__(self, model_name: str = "small",
                 device: str = "cuda",
                 compute_type: str = "float16",
                 language: Optional[str] = None,
                 cpu_threads: int = 0):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.cpu_threads = cpu_threads
        self._model: Optional[WhisperModel] = None
        self._lock = asyncio.Lock()

    async def ensure_loaded(self):
        if self._model is not None:
            return
        loop = asyncio.get_event_loop()
        async with self._lock:
            if self._model is not None:
                return
            self._model = await loop.run_in_executor(
                None,
                lambda: WhisperModel(
                    self.model_name, device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=self.cpu_threads or 0,
                ),
            )

    async def transcribe_pcm16(self, pcm_bytes: bytes,
                                language: Optional[str] = None,
                                vad_filter: bool = False) -> TranscribeResult:
        await self.ensure_loaded()
        audio = pcm16_to_float32(pcm_bytes)
        return await self.transcribe_array(audio, language=language, vad_filter=vad_filter)

    async def transcribe_array(self, audio: np.ndarray,
                                language: Optional[str] = None,
                                vad_filter: bool = False) -> TranscribeResult:
        await self.ensure_loaded()
        loop = asyncio.get_event_loop()
        lang = language or self.language
        t0 = time.monotonic()

        def _run():
            segments_iter, info = self._model.transcribe(
                audio,
                language=lang,
                vad_filter=vad_filter,
                beam_size=1,         # greedy is fine for short utterances
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
            )
            text = "".join(s.text for s in segments_iter)
            return text, info

        text, info = await loop.run_in_executor(None, _run)
        elapsed = time.monotonic() - t0
        duration = float(audio.shape[0]) / DEFAULT_SAMPLE_RATE
        return TranscribeResult(
            text=text.strip(),
            language=getattr(info, "language", "") or "",
            duration_s=duration,
            elapsed_s=elapsed,
        )


# ---------- standalone smoke ----------


async def _demo():
    """Record 4 seconds of mic, transcribe with small."""
    import sounddevice as sd
    print("[mic] recording 4s @ 16 kHz mono...")
    audio = sd.rec(4 * DEFAULT_SAMPLE_RATE, samplerate=DEFAULT_SAMPLE_RATE,
                   channels=1, dtype="float32")
    sd.wait()
    audio = audio[:, 0] if audio.ndim == 2 else audio

    print("[whisper] loading small...")
    t = Transcriber(model_name="small", device="cuda", compute_type="float16",
                    language="zh")
    await t.ensure_loaded()
    res = await t.transcribe_array(audio)
    print(f"[result] text={res.text!r}")
    print(f"  duration={res.duration_s:.2f}s elapsed={res.elapsed_s:.2f}s "
          f"(rt-factor={res.elapsed_s/max(res.duration_s,0.001):.2f}x) lang={res.language}")


if __name__ == "__main__":
    asyncio.run(_demo())
