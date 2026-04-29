"""Pluggable streaming TTS.

Architecture:
    TTSBackend (abstract) — exposes `audio_format` + `async stream(text)` returning
                            an iterator of raw audio bytes.
    EdgeTTSBackend        — default, online, free, no API key. mp3 stream.
    OpenAITTSBackend      — POST to any /v1/audio/speech compatible endpoint
                            (OpenAI, ElevenLabs OpenAI mode, xtts-api-server,
                            self-hosted vLLM TTS gateways, etc.).
    PythonModuleBackend   — load `factory(kwargs) -> TTSBackend` from a user .py
                            file. Lets users plug arbitrary local models
                            (GPT-SoVITS, XTTS, your own fine-tune) without
                            modifying voicebridge itself.

    TTSPlayer             — backend-agnostic player. Pipes the byte stream into
                            ffplay (-nodisp), kills the subprocess on stop().
                            Tells ffplay the input format based on
                            backend.audio_format so PCM streams also work.

The backend's `stream()` is allowed to yield empty bytes / metadata chunks; the
player ignores anything that's not bytes-typed.

Audio format hints:
    "mp3"                       — auto-detected by ffplay
    "wav"                       — auto-detected by ffplay (file has RIFF header)
    "pcm_s16le_<rate>_<ch>"     — raw PCM, e.g. "pcm_s16le_24000_1"
    "pcm_f32le_<rate>_<ch>"     — float32 raw PCM
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import shutil
import subprocess
import threading
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, Optional

FFPLAY_BIN = shutil.which("ffplay") or "ffplay"


# ---------- backend interface ----------


class TTSBackend(ABC):
    audio_format: str = "mp3"  # one of: mp3 / wav / pcm_<sample_fmt>_<rate>_<channels>

    @abstractmethod
    async def stream(self, text: str) -> AsyncIterator[bytes]:
        """Yield raw audio bytes in `audio_format`. The player will pipe these into ffplay."""
        if False:  # pragma: no cover -- make this an async generator
            yield b""


# ---------- edge-tts (default) ----------


class EdgeTTSBackend(TTSBackend):
    audio_format = "mp3"

    def __init__(self, voice: str = "zh-CN-XiaoyiNeural",
                 rate: str = "+0%", volume: str = "+0%", pitch: str = "+0Hz"):
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.pitch = pitch

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        import edge_tts
        comm = edge_tts.Communicate(
            text, voice=self.voice, rate=self.rate, volume=self.volume, pitch=self.pitch
        )
        async for chunk in comm.stream():
            if chunk.get("type") == "audio":
                data = chunk.get("data")
                if data:
                    yield data


# ---------- OpenAI-compatible HTTP TTS ----------


class OpenAITTSBackend(TTSBackend):
    """POST text to an /v1/audio/speech-style endpoint and stream the body.

    Compatible with OpenAI, ElevenLabs (OpenAI mode), xtts-api-server, kokoro-fastapi,
    and most modern local TTS gateways. The endpoint must return audio/* over HTTP
    in chunked transfer or as a single body.

    Config:
        url       — full endpoint URL
        api_key   — optional bearer token
        model     — model name (server-specific)
        voice     — voice id (server-specific)
        format    — one of mp3 / wav / opus / aac / flac / pcm; passed through
                    in the request body and used to set ffplay format
        speed     — optional speed multiplier
        extra     — dict merged into the JSON request body (any server-specific knobs)
    """

    def __init__(self, url: str, api_key: str = "", model: str = "tts-1",
                 voice: str = "alloy", format: str = "mp3",
                 speed: float = 1.0, extra: Optional[Dict[str, Any]] = None):
        self.url = url
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.format = format
        self.speed = speed
        self.extra = dict(extra or {})
        # Map to ffplay-compatible audio_format hint
        self.audio_format = format if format in {"mp3", "wav", "opus", "aac", "flac"} else "mp3"

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        body = {
            "model": self.model,
            "voice": self.voice,
            "input": text,
            "response_format": self.format,
            "speed": self.speed,
        }
        body.update(self.extra)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Run blocking http in a thread to avoid blocking the event loop.
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        loop = asyncio.get_event_loop()

        def _producer():
            try:
                req = urllib.request.Request(
                    self.url, data=json.dumps(body).encode("utf-8"),
                    headers=headers, method="POST"
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    while True:
                        chunk = resp.read(4096)
                        if not chunk:
                            break
                        loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        await loop.run_in_executor(None, lambda: None)  # ensure executor warm
        loop.run_in_executor(None, _producer)
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item


# ---------- user-supplied python module ----------


class PythonModuleBackend(TTSBackend):
    """Wrap a backend produced by a user-supplied .py file.

    The user's file must expose a callable named `make_backend(**kwargs) -> TTSBackend`.
    Use this to plug GPT-SoVITS, XTTS, your own fine-tune, etc.
    """

    def __init__(self, path: str, factory: str = "make_backend",
                 kwargs: Optional[Dict[str, Any]] = None):
        spec = importlib.util.spec_from_file_location("user_tts_module", path)
        if not spec or not spec.loader:
            raise RuntimeError(f"cannot load python TTS module from: {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, factory):
            raise RuntimeError(f"{path} has no `{factory}` factory")
        self._inner: TTSBackend = getattr(mod, factory)(**(kwargs or {}))
        self.audio_format = self._inner.audio_format

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        async for chunk in self._inner.stream(text):
            yield chunk


# ---------- factory ----------


def load_backend(config: Dict[str, Any]) -> TTSBackend:
    """config is the contents of `config["tts"]` (the sub-dict)."""
    name = (config.get("backend") or "edge").lower()
    sub = config.get(name) or {}
    if name == "edge":
        return EdgeTTSBackend(**sub)
    if name in ("openai", "http"):
        return OpenAITTSBackend(**sub)
    if name == "python":
        return PythonModuleBackend(**sub)
    raise ValueError(f"unknown tts backend: {name!r}")


# ---------- player ----------


def _ffplay_args(audio_format: str) -> list[str]:
    """Build ffplay command line for the given audio format hint."""
    args = [FFPLAY_BIN, "-nodisp", "-autoexit", "-loglevel", "error"]
    if audio_format.startswith("pcm_"):
        # pcm_<sample_fmt>_<rate>_<channels>, e.g. pcm_s16le_24000_1
        try:
            _, sample_fmt, rate, channels = audio_format.split("_")
            args += ["-f", sample_fmt, "-ar", rate, "-ac", channels]
        except ValueError as e:
            raise ValueError(f"invalid pcm format hint: {audio_format!r}") from e
    args += ["-i", "pipe:0"]
    return args


@dataclass
class TTSPlayer:
    backend: TTSBackend
    _proc: Optional[subprocess.Popen] = field(default=None, init=False, repr=False)
    _task: Optional[asyncio.Task] = field(default=None, init=False, repr=False)
    # threading.Event so hotkey callbacks (running on a separate Win32 thread)
    # can stop playback immediately without going through asyncio scheduling.
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)

    def is_speaking(self) -> bool:
        return self._task is not None and not self._task.done()

    async def speak(self, text: str, on_done: Optional[Callable[[bool], None]] = None) -> bool:
        await self.stop()
        self._stop_event = threading.Event()
        self._task = asyncio.create_task(self._run(text))
        try:
            completed = await self._task
        except asyncio.CancelledError:
            completed = False
        finally:
            self._task = None
        if on_done:
            try: on_done(completed)
            except Exception: pass
        return completed

    async def stop(self):
        self._stop_event.set()
        if self._proc and self._proc.poll() is None:
            try: self._proc.kill()
            except ProcessLookupError: pass
            try: self._proc.wait(timeout=1)
            except Exception: pass
        self._proc = None
        if self._task and not self._task.done():
            self._task.cancel()
            try: await self._task
            except (asyncio.CancelledError, Exception): pass
        self._task = None

    async def _run(self, text: str) -> bool:
        if not text.strip():
            return True
        try:
            from paths import FFPLAY_LOG
            ffplay_log = open(FFPLAY_LOG, "ab")
            ffplay_log.write(f"\n=== {__import__('datetime').datetime.now().isoformat()} ===\n".encode())
            # CREATE_NO_WINDOW prevents the brief cmd / ffplay console window
            # that would otherwise pop up on every TTS utterance.
            CREATE_NO_WINDOW = 0x08000000
            self._proc = subprocess.Popen(
                _ffplay_args(self.backend.audio_format),
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=ffplay_log,
                creationflags=CREATE_NO_WINDOW,
            )
            print(f"[tts] ffplay pid={self._proc.pid} spawned (stderr -> .ffplay.log)")
        except Exception as e:
            print(f"[tts] ffplay spawn FAILED: {e}")
            return False
        chunk_count = 0
        bytes_written = 0
        try:
            async for chunk in self.backend.stream(text):
                if self._stop_event.is_set():
                    return False
                if not chunk or not isinstance(chunk, (bytes, bytearray)):
                    continue
                chunk_count += 1
                bytes_written += len(chunk)
                if chunk_count == 1:
                    print(f"[tts] first chunk arrived ({len(chunk)} bytes)")
                # If ffplay died early, print its stderr so we can see why.
                if self._proc.poll() is not None and self._proc.returncode != 0:
                    try:
                        err = self._proc.stderr.read(2000).decode("utf-8", "ignore") if self._proc.stderr else ""
                        if err.strip():
                            print(f"[tts/ffplay] died (rc={self._proc.returncode}): {err}")
                    except Exception:
                        pass
                    return False
                if self._proc.stdin and not self._proc.stdin.closed:
                    try:
                        self._proc.stdin.write(chunk)
                        self._proc.stdin.flush()
                    except (BrokenPipeError, OSError) as e:
                        try:
                            err = self._proc.stderr.read(2000).decode("utf-8", "ignore") if self._proc.stderr else ""
                            if err.strip():
                                print(f"[tts/ffplay] pipe broken: {err}")
                        except Exception:
                            pass
                        return False
            print(f"[tts] stream done: {chunk_count} chunks, {bytes_written} bytes total")
            try:
                if self._proc.stdin and not self._proc.stdin.closed:
                    self._proc.stdin.close()
            except OSError:
                pass
            while self._proc.poll() is None:
                if self._stop_event.is_set():
                    return False
                await asyncio.sleep(0.1)
            return True
        except asyncio.CancelledError:
            return False
        finally:
            if self._proc and self._proc.poll() is None:
                try: self._proc.kill()
                except ProcessLookupError: pass
            self._proc = None


# ---------- standalone smoke test ----------


async def _demo():
    import sys
    text = " ".join(sys.argv[1:]) or "你好,我是 Wren。这是流式语音合成的测试。"
    backend = EdgeTTSBackend()
    player = TTSPlayer(backend)
    completed = await player.speak(text)
    print(f"[tts] backend={type(backend).__name__} format={backend.audio_format} completed={completed}")


if __name__ == "__main__":
    asyncio.run(_demo())
