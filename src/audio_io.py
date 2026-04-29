"""Microphone capture + VAD + segmentation state machine.

Pipeline:
  sounddevice mic (callback thread)
    -> thread-safe queue of 512-sample blocks (32 ms @ 16 kHz, the size
       silero-vad expects)
    -> asyncio task that pulls blocks, feeds silero VADIterator, emits frames
       and speech-start/end events
    -> Segmenter consumes those events + frames and emits two kinds of events:
         segment_ready(audio_bytes, start_ts, end_ts)   on each `pause_segment_s`
         ready_to_submit()                              on each `pause_submit_s`
       Both timers reset whenever speech resumes.

The Segmenter is *passive* — it doesn't know about wake words or claude. It
just packages contiguous-with-short-gaps speech into segments and signals when
the user has been silent long enough to consider the utterance "done."

Activation/deactivation (wake word triggering segmentation, TTS pausing it,
etc.) is the caller's job — they `start_recording()` after a wake-word hit
and `stop_recording()` after submission.
"""
from __future__ import annotations

import asyncio
import queue as thread_queue
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, List, Optional

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
BLOCK_SAMPLES = 512  # silero-vad expected size at 16 kHz (32 ms)
BLOCK_DURATION_S = BLOCK_SAMPLES / SAMPLE_RATE

from collections import deque


# ---------- mic capture (thread → asyncio bridge) ----------


@dataclass
class AudioBlock:
    pcm: np.ndarray   # float32, shape (BLOCK_SAMPLES,), in [-1, 1]
    timestamp: float  # monotonic seconds when captured


class MicCapture:
    """Continuous mic capture via sounddevice. Push blocks into an asyncio.Queue."""

    def __init__(self, device: Optional[int] = None,
                 sample_rate: int = SAMPLE_RATE,
                 block_samples: int = BLOCK_SAMPLES):
        self.device = device
        self.sample_rate = sample_rate
        self.block_samples = block_samples
        self._tq: "thread_queue.Queue[AudioBlock]" = thread_queue.Queue(maxsize=1024)
        self._stream: Optional[sd.InputStream] = None
        self._pump_task: Optional[asyncio.Task] = None
        self.aq: "asyncio.Queue[AudioBlock]" = asyncio.Queue()

    def _callback(self, indata, frames, time_info, status):
        # Called from sounddevice's audio thread. Keep it cheap.
        if status:
            # XRuns etc. — log lazily, don't crash audio.
            pass
        pcm = indata[:, 0].astype(np.float32, copy=True) if indata.ndim == 2 else indata.astype(np.float32, copy=True)
        try:
            self._tq.put_nowait(AudioBlock(pcm=pcm, timestamp=time.monotonic()))
        except thread_queue.Full:
            # Drop block under backpressure rather than block the audio thread.
            try: self._tq.get_nowait()
            except thread_queue.Empty: pass
            try: self._tq.put_nowait(AudioBlock(pcm=pcm, timestamp=time.monotonic()))
            except thread_queue.Full: pass

    async def _pump(self):
        loop = asyncio.get_event_loop()
        while True:
            block = await loop.run_in_executor(None, self._tq.get)
            await self.aq.put(block)

    async def start(self):
        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            samplerate=self.sample_rate, channels=1, dtype="float32",
            blocksize=self.block_samples, device=self.device, callback=self._callback,
        )
        self._stream.start()
        self._pump_task = asyncio.create_task(self._pump())

    async def stop(self):
        if self._stream is not None:
            self._stream.stop(); self._stream.close(); self._stream = None
        if self._pump_task:
            self._pump_task.cancel()
            try: await self._pump_task
            except (asyncio.CancelledError, Exception): pass
            self._pump_task = None


# ---------- AEC (spectral subtraction over WASAPI loopback) ----------


class LoopbackCapture(MicCapture):
    """Same as MicCapture but for the speaker-loopback device (Stereo Mix etc.).

    On Windows the user typically picks a device whose name contains "扬声器",
    "Stereo Mix", or "Loopback". We resample the device's native rate down to
    16 kHz via sounddevice/portaudio's internal resampler.
    """


class FBNLMS:
    """Frequency-domain block-NLMS adaptive echo canceller.

    Adaptively learns the impulse response from `loopback` (what's playing in
    the speakers, captured via WASAPI loopback / Stereo Mix) to `mic`. Each
    block produces a clean residual `e ≈ speech_only`. Convergence happens
    over the first ~1-3 seconds of TTS playback.

    This is a textbook overlap-save block-NLMS:
        x_t = [x_prev || lb]                    (length 2N)
        Y   = H * FFT(x_t)
        y   = IFFT(Y)[N:]                        (current block echo estimate)
        e   = mic - y
        E   = FFT([0..0 || e])                   (zero-padded to 2N)
        H  += mu * conj(X) * E / (|X|^2 + eps)

    Stability: cap |H| update via simple regularization. Stalls gracefully
    when the speaker is silent (X≈0 → no update).
    """

    def __init__(self, block_size: int = BLOCK_SAMPLES,
                 fft_size: Optional[int] = None,
                 mu: float = 0.4, eps: float = 1e-3):
        self.N = block_size
        self.M = fft_size or 2 * block_size
        self.mu = mu
        self.eps = eps
        self.H = np.zeros(self.M // 2 + 1, dtype=np.complex64)
        self.x_prev = np.zeros(self.N, dtype=np.float32)

    def reset(self):
        self.H[:] = 0
        self.x_prev[:] = 0

    def process(self, mic: np.ndarray, lb: np.ndarray) -> np.ndarray:
        n = self.N
        if mic.shape[0] != n:
            mic = np.pad(mic, (0, n - mic.shape[0])) if mic.shape[0] < n else mic[:n]
        if lb.shape[0] != n:
            lb = np.pad(lb, (0, n - lb.shape[0])) if lb.shape[0] < n else lb[:n]

        x = np.concatenate([self.x_prev, lb])  # 2N
        self.x_prev = lb.astype(np.float32, copy=True)

        X = np.fft.rfft(x, n=self.M)
        Y = X * self.H
        y = np.fft.irfft(Y, n=self.M)[-n:].astype(np.float32)
        e = mic - y

        # Zero-pad e at front (overlap-save)
        e_padded = np.concatenate([np.zeros(self.N, dtype=np.float32), e])
        E = np.fft.rfft(e_padded, n=self.M)
        norm = (X.conj() * X).real + self.eps
        self.H = self.H + (self.mu * X.conj() * E / norm).astype(np.complex64)
        return e


def _spectral_subtract(mic: np.ndarray, lb: np.ndarray, alpha: float = 1.5,
                       floor: float = 0.05) -> np.ndarray:
    """Backup: simple spectral subtraction (unused now FBNLMS is in)."""
    n = mic.shape[0]
    if lb.shape[0] != n:
        if lb.shape[0] < n:
            lb = np.pad(lb, (0, n - lb.shape[0]))
        else:
            lb = lb[:n]
    M = np.fft.rfft(mic)
    L = np.fft.rfft(lb)
    mag_clean = np.maximum(np.abs(M) - alpha * np.abs(L), floor * np.abs(M))
    out = np.fft.irfft(mag_clean * np.exp(1j * np.angle(M)), n)
    return out.astype(np.float32, copy=False)


class EchoCanceller:
    """Mixes mic + loopback streams, emits cleaned mic blocks via spectral subtraction.

    Wires:
        mic     : MicCapture (already started before run())
        loopback: LoopbackCapture (already started before run())
        out_q   : asyncio.Queue receiving cleaned AudioBlock(s)
    """

    def __init__(self, mic: MicCapture, loopback: LoopbackCapture,
                 alpha: float = 1.5, delay_s: float = 0.020,
                 use_nlms: bool = True):
        self.mic = mic
        self.loopback = loopback
        self.alpha = alpha
        self.delay_s = delay_s
        self.aq: "asyncio.Queue[AudioBlock]" = asyncio.Queue()
        # Side-channel of raw (un-AEC'd) mic blocks. The wake/interrupt watcher
        # consumes these — empirically small-whisper does *better* at spotting
        # short keywords on raw mic than on NLMS-cleaned audio (NLMS leaves
        # musical noise that confuses the small model on single-character
        # keywords like "停").
        self.raw_aq: "asyncio.Queue[AudioBlock]" = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._lb_pump_task: Optional[asyncio.Task] = None
        self._lb_buffer: "deque[AudioBlock]" = deque(maxlen=64)  # ~2 s history
        self._nlms = FBNLMS() if use_nlms else None

    async def start(self):
        # Both streams must already be started by caller.
        self._lb_pump_task = asyncio.create_task(self._lb_pump())
        self._task = asyncio.create_task(self._mix())

    async def stop(self):
        for t in (self._task, self._lb_pump_task):
            if t and not t.done():
                t.cancel()
                try: await t
                except (asyncio.CancelledError, Exception): pass
        self._task = None; self._lb_pump_task = None

    async def _lb_pump(self):
        while True:
            blk = await self.loopback.aq.get()
            self._lb_buffer.append(blk)

    async def _mix(self):
        while True:
            mic_blk = await self.mic.aq.get()
            # Side-channel: forward the raw block for the watcher.
            try: self.raw_aq.put_nowait(mic_blk)
            except asyncio.QueueFull: pass
            cleaned_pcm = mic_blk.pcm
            if self._lb_buffer:
                target = mic_blk.timestamp - self.delay_s
                best = min(self._lb_buffer, key=lambda b: abs(b.timestamp - target))
                if abs(best.timestamp - target) < 0.060:
                    try:
                        if self._nlms is not None:
                            cleaned_pcm = self._nlms.process(mic_blk.pcm, best.pcm)
                        else:
                            cleaned_pcm = _spectral_subtract(mic_blk.pcm, best.pcm, alpha=self.alpha)
                    except Exception as e:
                        cleaned_pcm = mic_blk.pcm
            await self.aq.put(AudioBlock(pcm=cleaned_pcm, timestamp=mic_blk.timestamp))


# ---------- silero VAD ----------


class VADGate:
    """Wraps silero-vad. Per-block returns is_speech (after smoothing)."""

    def __init__(self, threshold: float = 0.5,
                 min_silence_ms: int = 100,
                 speech_pad_ms: int = 30):
        from silero_vad import load_silero_vad, VADIterator
        self._model = load_silero_vad(onnx=False)
        self._iter = VADIterator(
            self._model, sampling_rate=SAMPLE_RATE,
            threshold=threshold,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )
        self._in_speech = False
        # silero-vad takes torch.Tensor at the iterator level
        import torch  # noqa: F401  -- ensures torch is available
        self._torch = __import__("torch")

    def reset(self):
        self._iter.reset_states()
        self._in_speech = False

    def feed(self, block: np.ndarray) -> dict:
        """Feed one 512-sample block. Returns:
            {} if no boundary
            {"start": seconds_since_iter_start} on speech start
            {"end":   seconds_since_iter_start} on speech end
        Also updates internal in_speech flag.
        """
        t = self._torch.from_numpy(block)
        ev = self._iter(t, return_seconds=True) or {}
        if "start" in ev: self._in_speech = True
        if "end"   in ev: self._in_speech = False
        return ev

    @property
    def in_speech(self) -> bool:
        return self._in_speech


# ---------- segmenter ----------


SegmentCallback = Callable[[bytes, float, float], Awaitable[None]]
SubmitCallback = Callable[[], Awaitable[None]]


@dataclass
class SegmenterConfig:
    pause_segment_s: float = 2.0  # silence to flush a segment
    pause_submit_s: float = 8.0   # silence to trigger submission


class Segmenter:
    """Owns the audio buffer for the active utterance and emits segment/submit events.

    Lifecycle:
        start_recording()                  — begin buffering, reset timers
        stop_recording()                   — discard buffer, no more emits
        await on_block(block)              — feed each AudioBlock while recording

    The caller wires `on_segment_ready` / `on_ready_to_submit` to trigger
    transcription + bridge submission.
    """

    def __init__(self, vad: VADGate, config: Optional[SegmenterConfig] = None,
                 on_segment_ready: Optional[SegmentCallback] = None,
                 on_ready_to_submit: Optional[SubmitCallback] = None):
        self.vad = vad
        self.config = config or SegmenterConfig()
        self.on_segment_ready = on_segment_ready
        self.on_ready_to_submit = on_ready_to_submit
        self._buffer: List[np.ndarray] = []
        self._buffer_start_ts: Optional[float] = None
        self._last_speech_end_ts: Optional[float] = None
        self._submitted = False  # set after we fire on_ready_to_submit
        self._segment_flushed_for_this_gap = False  # gate against repeat flush within same silence
        self._recording = False
        self._segment_index = 0  # for callers that want IDs

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start_recording(self):
        self.vad.reset()
        self._buffer.clear()
        self._buffer_start_ts = None
        self._last_speech_end_ts = None
        self._submitted = False
        self._segment_flushed_for_this_gap = False
        self._recording = True
        self._segment_index = 0

    def stop_recording(self):
        self._recording = False
        self._buffer.clear()
        self._buffer_start_ts = None
        self._last_speech_end_ts = None

    async def on_block(self, block: AudioBlock):
        if not self._recording:
            return
        ev = self.vad.feed(block.pcm)
        if self._buffer_start_ts is None:
            self._buffer_start_ts = block.timestamp
        self._buffer.append(block.pcm)

        if ev:
            if "start" in ev:
                # User resumed speaking before submission window — cancel any
                # pending submission state so the new speech extends the utterance.
                self._submitted = False
                self._segment_flushed_for_this_gap = False
                self._last_speech_end_ts = None
            if "end" in ev:
                self._last_speech_end_ts = block.timestamp

        # Are we in a silence period long enough to flush a segment / submit?
        if not self.vad.in_speech and self._last_speech_end_ts is not None:
            silence = block.timestamp - self._last_speech_end_ts
            # Flush a segment once per silence period (gate prevents per-block re-flush).
            if (silence >= self.config.pause_segment_s
                    and not self._segment_flushed_for_this_gap
                    and self._buffer):
                self._segment_flushed_for_this_gap = True
                await self._flush_segment()
            if (silence >= self.config.pause_submit_s and not self._submitted
                    and self.on_ready_to_submit):
                self._submitted = True
                await self.on_ready_to_submit()

    async def _flush_segment(self):
        if not self._buffer:
            return
        start = self._buffer_start_ts or 0.0
        end = self._last_speech_end_ts or start
        pcm = np.concatenate(self._buffer)
        self._buffer.clear()
        self._buffer_start_ts = None
        self._segment_index += 1
        if self.on_segment_ready:
            # int16-encoded raw PCM — easier to hand to whisper / save to wav
            pcm_i16 = np.clip(pcm * 32767, -32768, 32767).astype(np.int16).tobytes()
            await self.on_segment_ready(pcm_i16, start, end)


# ---------- helper: a self-contained async pipeline ----------


@dataclass
class AudioPipeline:
    """Wires MicCapture (+ optional AEC) → VAD-fed Segmenter together.

    With AEC: mic + loopback streams run in parallel, EchoCanceller produces
    cleaned blocks to the segmenter / on_block callback.
    """
    mic: MicCapture = field(default_factory=MicCapture)
    vad: VADGate = field(default_factory=VADGate)
    config: SegmenterConfig = field(default_factory=SegmenterConfig)
    loopback: Optional[LoopbackCapture] = None
    aec_alpha: float = 1.5
    segmenter: Segmenter = field(init=False)
    aec: Optional[EchoCanceller] = field(default=None, init=False)
    _drain_task: Optional[asyncio.Task] = field(default=None, init=False)
    on_block: Optional[Callable[[AudioBlock], Awaitable[None]]] = None

    def __post_init__(self):
        self.segmenter = Segmenter(self.vad, self.config)

    async def start(self):
        await self.mic.start()
        if self.loopback is not None:
            # Loopback device sometimes fails right after a previous process
            # held it (PaErrorCode -9996). Retry a couple of times before
            # giving up on AEC.
            ok = False
            last_err = None
            for attempt in range(3):
                try:
                    await self.loopback.start()
                    self.aec = EchoCanceller(self.mic, self.loopback, alpha=self.aec_alpha)
                    await self.aec.start()
                    print(f"[audio] AEC enabled (loopback active{', attempt ' + str(attempt+1) if attempt else ''})")
                    ok = True
                    break
                except Exception as e:
                    last_err = e
                    if attempt < 2:
                        print(f"[audio] loopback open failed (attempt {attempt+1}/3): {e}; retrying...")
                        await asyncio.sleep(0.6)
            if not ok:
                print(f"[audio] loopback gave up, AEC disabled: {last_err}")
                self.loopback = None
                self.aec = None
        self._drain_task = asyncio.create_task(self._drain())

    async def stop(self):
        if self._drain_task:
            self._drain_task.cancel()
            try: await self._drain_task
            except (asyncio.CancelledError, Exception): pass
            self._drain_task = None
        if self.aec is not None:
            await self.aec.stop()
        if self.loopback is not None:
            await self.loopback.stop()
        await self.mic.stop()

    async def _drain(self):
        # Both segmenter and watcher consume AEC-cleaned audio. We tried
        # giving the watcher raw mic for better short-keyword recall, but it
        # hallucinated "停" out of the TTS's own playback — much worse FPR.
        # Use cleaned for both; voice interrupt has known FN, hotkey is the
        # reliable backup.
        seg_q = self.aec.aq if self.aec is not None else self.mic.aq
        # Drain raw_aq if present so it doesn't grow unboundedly (we forward
        # blocks into it from the AEC mixer, but no consumer needs them now).
        raw_q = self.aec.raw_aq if self.aec is not None else None
        while True:
            block = await seg_q.get()
            if raw_q is not None:
                try: raw_q.get_nowait()
                except asyncio.QueueEmpty: pass
            if self.on_block:
                try: await self.on_block(block)
                except Exception: pass
            await self.segmenter.on_block(block)


# ---------- standalone smoke test ----------


async def _demo():
    """Quick mic test: prints VAD events for ~10 seconds."""
    pipe = AudioPipeline()

    async def seg_cb(pcm_i16: bytes, t0: float, t1: float):
        secs = len(pcm_i16) / 2 / SAMPLE_RATE
        print(f"  [segment] {secs:.2f}s of audio buffered (t0={t0:.1f} t1={t1:.1f})")

    async def submit_cb():
        print("  [SUBMIT triggered]")

    pipe.segmenter.on_segment_ready = seg_cb
    pipe.segmenter.on_ready_to_submit = submit_cb

    await pipe.start()
    print("[mic] recording for 12s. say something, then be silent. Ctrl+C to stop early.")
    pipe.segmenter.start_recording()

    last_speech = False
    try:
        for _ in range(int(12 / BLOCK_DURATION_S)):
            await asyncio.sleep(BLOCK_DURATION_S)
            if pipe.vad.in_speech != last_speech:
                last_speech = pipe.vad.in_speech
                print(f"  [vad] {'speech' if last_speech else 'silence'}")
    finally:
        pipe.segmenter.stop_recording()
        await pipe.stop()


if __name__ == "__main__":
    asyncio.run(_demo())
