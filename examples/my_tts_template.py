"""Template for plugging your own TTS model into voicebridge.

How it works:
  1. Set `tts.backend = "python"` in voicebridge's config.json
  2. Point `tts.python.path` at this file (or a copy you customize)
  3. Implement `make_backend(**kwargs)` to return your `TTSBackend` instance

The backend just needs to:
  - declare `audio_format` (a string voicebridge passes to ffplay)
  - implement `async def stream(text: str)` yielding raw audio bytes

Common backends people plug here:
  - GPT-SoVITS / Bert-VITS2     — local clone of your own voice
  - XTTS v2 (coqui-tts)         — multilingual, decent zero-shot cloning
  - Bark / StyleTTS2            — high quality, slower
  - Piper                       — fast CPU-only, good for low-resource
  - your fine-tuned model       — anything pythonable

audio_format options:
  "mp3" / "wav" / "opus" / "aac" / "flac"   ← codecs ffplay auto-detects
  "pcm_s16le_<rate>_<channels>"             ← raw PCM s16 little-endian
  "pcm_f32le_<rate>_<channels>"             ← raw PCM float32 little-endian

  e.g. "pcm_s16le_24000_1" for 24 kHz 16-bit mono raw stream.

For *streaming* feel, your stream() should yield small chunks (50-200 ms each)
as soon as the model produces them, not the whole utterance at once.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import AsyncIterator

# Make voicebridge's src/ importable. This file is examples/my_tts_template.py;
# the package layout puts src/ next to examples/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tts import TTSBackend


class MyCustomTTS(TTSBackend):
    # Tell voicebridge what bytes your stream() will produce.
    audio_format = "pcm_s16le_24000_1"  # 24 kHz, 16-bit, mono raw PCM

    def __init__(self, model_path: str = "", device: str = "cuda", **kwargs):
        # Load your model once here. Store on self.
        self.model_path = model_path
        self.device = device
        # self.model = MyModel.load(model_path).to(device)

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        loop = asyncio.get_event_loop()
        # Push the model call to a worker thread so we don't block the event loop.
        # Real streaming models would yield incrementally from a generator running
        # in the worker; for non-streaming models, we just chunk the final tensor.
        pcm: bytes = await loop.run_in_executor(None, self._synthesize, text)

        # Chunk the bytes so playback can start ASAP and stop() can interrupt.
        # 100 ms chunks at 24 kHz s16le mono = 24000 * 0.1 * 2 = 4800 bytes
        chunk_bytes = 4800
        for i in range(0, len(pcm), chunk_bytes):
            yield pcm[i:i + chunk_bytes]
            await asyncio.sleep(0)  # cooperative yield

    def _synthesize(self, text: str) -> bytes:
        # ============================================================
        # REPLACE THIS WITH YOUR MODEL CALL.
        # Must return raw PCM bytes matching the `audio_format` declared above.
        # ============================================================
        # Example placeholder: 1 second of silence at 24 kHz s16le mono.
        return bytes(24000 * 2)


def make_backend(**kwargs) -> TTSBackend:
    """voicebridge calls this. kwargs come from config["tts"]["python"]["kwargs"]."""
    return MyCustomTTS(**kwargs)
