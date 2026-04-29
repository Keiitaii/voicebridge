"""voicebridge main loop.

Wires:
  AudioPipeline (mic + silero VAD + Segmenter)
  KeywordWatcher (small whisper sliding window for wake / end words)
  Transcriber (large-v3-turbo for the actual utterance)
  BridgeClient (push prompt + SSE assistant text)
  TTSPlayer (edge-tts / openai-compat / user python)

States:
  IDLE              — listening for wake word
  ACTIVE            — in dialog; recording user's utterance via Segmenter
  SUBMITTING        — silence threshold reached; transcribing + posting prompt
  SPEAKING          — TTS playing back claude's reply; watcher muted so the
                      assistant's own words don't trip wake/end keywords.
                      Interrupt is via the global hotkey only (Ctrl+Shift+Space
                      by default); voice-keyword interrupt was removed because
                      whisper-small couldn't reliably catch single-character
                      Chinese keywords through the AEC residue.
  After SPEAKING ends naturally → ACTIVE (multi-turn dialog continues).
  End-word during ACTIVE/SPEAKING → claude generates a short farewell, TTS
                                    plays it, then back to IDLE.
  Hotkey during SPEAKING → cancel TTS, prepend the prior prompt to the next
                            submission so context isn't lost (carry-over).

Run with: python main.py [path/to/config.json]
Default config path: <install-dir>\config.json
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

# DLLs first
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuda_setup  # noqa: F401

import analyze
from audio_io import AudioBlock, AudioPipeline, LoopbackCapture, MicCapture, SAMPLE_RATE, SegmenterConfig, VADGate

# P1.2 / P1.3 are heavy and optional. Lazy-import on first use; if the
# packages aren't installed (e.g. user opted out of tensorflow), the analysis
# block degrades gracefully.
_sound_tagger = None
_emotion_det = None
from bridge_client import BridgeClient
from transcribe import Transcriber, pcm16_to_float32
from tts import TTSPlayer, load_backend as load_tts_backend, EdgeTTSBackend
from wakeword import KeywordWatcher

from win_hotkey import GlobalHotkey

from paths import (
    CONFIG_PATH as DEFAULT_CONFIG_PATH,
    EXAMPLE_CONFIG_PATH,
    INSTALL_DIR,
    LAST_ASSISTANT,
    LAST_PROMPT,
)


# ---------- state machine ----------


class State:
    IDLE = "idle"
    ACTIVE = "active"
    SUBMITTING = "submitting"
    SPEAKING = "speaking"


@dataclass
class VoiceAgent:
    config: dict
    pipeline: AudioPipeline = field(init=False)
    monitor_asr: Transcriber = field(init=False)
    main_asr: Transcriber = field(init=False)
    bridge: BridgeClient = field(init=False)
    tts: TTSPlayer = field(init=False)
    watcher: KeywordWatcher = field(init=False)

    state: str = field(default=State.IDLE, init=False)
    _segments_pcm: List[bytes] = field(default_factory=list, init=False)
    _carry_over_prompt: str = field(default="", init=False)
    _last_submitted_prompt: str = field(default="", init=False)
    _last_assistant_text: str = field(default="", init=False)
    _sse_task: Optional[asyncio.Task] = field(default=None, init=False)
    _watcher_task_started: bool = field(default=False, init=False)
    _hotkey: Optional[GlobalHotkey] = field(default=None, init=False)
    _loop_ref: Optional[asyncio.AbstractEventLoop] = field(default=None, init=False)
    _tray: Optional[object] = field(default=None, init=False)
    _quit_requested: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    _farewell_pending: bool = field(default=False, init=False)
    # config-watch state for hot reload
    _config_path: Optional[Path] = field(default=None, init=False)
    _config_mtime: float = field(default=0.0, init=False)
    _config_watch_task: Optional[asyncio.Task] = field(default=None, init=False)
    # bridge health watch — quit voicebridge if user closes the isolated VSCode
    _bridge_watch_task: Optional[asyncio.Task] = field(default=None, init=False)
    # whether we launched the isolated VSCode ourselves (true ⇒ we should
    # close it on quit). If user already had a VSCode + bridge running when we
    # started, we leave it alone.
    _we_launched_vscode: bool = field(default=False, init=False)

    def __post_init__(self):
        ac = self.config.get("audio", {})
        loopback = None
        if ac.get("aec_enabled") and ac.get("loopback_device") is not None:
            loopback = LoopbackCapture(device=ac["loopback_device"])
        self.pipeline = AudioPipeline(
            mic=MicCapture(device=ac.get("input_device")),
            vad=VADGate(threshold=ac.get("vad_threshold", 0.5)),
            config=SegmenterConfig(
                pause_segment_s=ac.get("pause_segment_s", 2.0),
                pause_submit_s=ac.get("pause_submit_s", 4.0),
            ),
            loopback=loopback,
            aec_alpha=ac.get("aec_alpha", 1.5),
        )
        asr_cfg = self.config.get("asr", {})
        self.monitor_asr = Transcriber(
            model_name=asr_cfg.get("monitor_model", "small"),
            device=asr_cfg.get("device", "cuda"),
            compute_type=asr_cfg.get("compute_type", "float16"),
            language=asr_cfg.get("language", "zh"),
        )
        self.main_asr = Transcriber(
            model_name=asr_cfg.get("main_model", "large-v3-turbo"),
            device=asr_cfg.get("device", "cuda"),
            compute_type=asr_cfg.get("compute_type", "float16"),
            language=asr_cfg.get("language", "zh"),
        )
        self.bridge = BridgeClient(self.config.get("bridge", {}).get("url", "http://127.0.0.1:43117"))
        try:
            tts_backend = load_tts_backend(self.config.get("tts", {}))
        except Exception as e:
            print(f"[tts] backend load failed ({e}); falling back to default edge-tts")
            tts_backend = EdgeTTSBackend()
        self.tts = TTSPlayer(tts_backend)
        self.watcher = KeywordWatcher(
            asr=self.monitor_asr,
            wake_words=self.config.get("wake_words", []),
            end_words=self.config.get("end_words", []),
            language=asr_cfg.get("language", "zh"),
            on_wake=self._on_wake_hit,
            on_end=self._on_end_hit,
        )
        # Wire callbacks
        self.pipeline.on_block = self.watcher.on_block
        self.pipeline.segmenter.on_segment_ready = self._on_segment_ready
        self.pipeline.segmenter.on_ready_to_submit = self._on_submit_signal

    # ---------------- lifecycle ----------------

    async def start(self):
        print(f"[agent] booting — state={self.state}")
        # Eagerly preload monitor model so wake-word detection is instant after first audio.
        print("[agent] loading monitor whisper (small)...")
        await self.monitor_asr.ensure_loaded()
        print("[agent] monitor ready")
        # Lazy-load main on first submit (saves VRAM + startup time)

        # Bridge: reuse existing if up, else spawn isolated VSCode ourselves.
        try:
            status = await self.bridge.status()
            print(f"[bridge] already up: {status}")
        except Exception:
            print("[bridge] not running, launching isolated VSCode...")
            try:
                import launcher
                workspace = Path(self.config.get("workspace", str(INSTALL_DIR)))
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, launcher.launch_vscode, workspace)
                status = await loop.run_in_executor(
                    None, lambda: launcher.wait_for_bridge(timeout=180)
                )
                print(f"[bridge] up after launch: {status}")
                self._we_launched_vscode = True
            except Exception as e:
                print(f"[bridge] FAILED to start: {e}")
                return

        # Start audio pipeline + watcher + SSE
        await self.pipeline.start()
        self.watcher.start()
        self._sse_task = asyncio.create_task(
            self.bridge.stream_assistant(self._on_assistant_text, on_error=self._on_api_error)
        )

        # Register global hotkey for interrupt (Win32 API directly)
        self._loop_ref = asyncio.get_event_loop()
        hotkey = self.config.get("interrupt_hotkey", "ctrl+shift+space")
        if hotkey:
            try:
                self._hotkey = GlobalHotkey(hotkey, self._fire_hotkey_interrupt)
                if self._hotkey.start():
                    print(f"[agent] interrupt hotkey: {hotkey}")
                else:
                    print(f"[hotkey] registration failed for {hotkey}")
                    self._hotkey = None
            except Exception as e:
                print(f"[hotkey] failed: {e}")
                self._hotkey = None

        # Optional system tray icon
        try:
            from tray import Tray
            self._tray = Tray(
                on_quit=self._tray_quit,
                on_open_settings=self._tray_open_settings,
                tooltip="Voicebridge · listening",
            )
            self._tray.start()
        except Exception as e:
            print(f"[tray] not available: {e}")
            self._tray = None

        # Watch config.json for changes — hot-reloads wake/end words, hotkey,
        # tts backend without restarting the whole agent. Heavy fields (audio
        # device, ASR models) still need a restart and we just log a note.
        if self._config_path is not None:
            try: self._config_mtime = self._config_path.stat().st_mtime
            except Exception: self._config_mtime = 0.0
            self._config_watch_task = asyncio.create_task(self._watch_config())

        # Watch bridge health — if the user closes the isolated VSCode window,
        # the bridge HTTP server dies and we should exit too.
        self._bridge_watch_task = asyncio.create_task(self._watch_bridge())

        print(f"[agent] listening for wake words: {self.config.get('wake_words')}")

    def _tray_quit(self):
        """Called from tray's pystray thread."""
        if self._loop_ref is not None:
            self._loop_ref.call_soon_threadsafe(self._quit_requested.set)

    def _tray_open_settings(self):
        """Spawn the startup GUI as an independent subprocess so the user can
        edit config.json without touching this running agent. Changes apply
        on the *next* voicebridge launch (hot-reload is on the todo)."""
        import subprocess
        gui_script = str(Path(__file__).resolve().parent / "startup_gui.py")
        try:
            subprocess.Popen(
                [sys.executable, gui_script],
                cwd=str(Path(gui_script).parent),
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
        except Exception as e:
            print(f"[tray] settings spawn failed: {e}")

    async def _watch_bridge(self):
        """Poll bridge /status; if it stops responding for ~3 consecutive
        checks (~9s) assume the user closed the isolated VSCode and quit."""
        miss = 0
        while True:
            await asyncio.sleep(3.0)
            try:
                await self.bridge.status()
                miss = 0
            except Exception:
                miss += 1
                if miss >= 3:
                    print("[bridge] gone (isolated VSCode closed) — quitting voicebridge")
                    if self._loop_ref is not None:
                        self._loop_ref.call_soon_threadsafe(self._quit_requested.set)
                    return

    async def _watch_config(self):
        """Poll config.json mtime; on change, reload hot-applicable fields."""
        while True:
            await asyncio.sleep(1.5)
            if self._config_path is None:
                return
            try:
                mtime = self._config_path.stat().st_mtime
            except Exception:
                continue
            if mtime <= self._config_mtime:
                continue
            self._config_mtime = mtime
            try:
                new_cfg = json.loads(self._config_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[config] reload parse failed: {e}")
                continue
            await self._apply_config_hot(new_cfg)

    async def _apply_config_hot(self, new_cfg: dict):
        """Apply a new config to the running agent. Fields that can't be
        hot-applied get a console note; the user can restart to pick them up."""
        old = self.config
        self.config = new_cfg

        # Wake / end words → watcher
        new_wake = new_cfg.get("wake_words", [])
        new_end = new_cfg.get("end_words", [])
        if new_wake != self.watcher.wake_words or new_end != self.watcher.end_words:
            self.watcher.wake_words = list(new_wake)
            self.watcher.end_words = list(new_end)
            print(f"[config] reloaded keywords — wake={new_wake} end={new_end}")

        # Interrupt hotkey
        new_hk = new_cfg.get("interrupt_hotkey", "")
        old_hk = old.get("interrupt_hotkey", "")
        if new_hk != old_hk:
            if self._hotkey is not None:
                try: self._hotkey.stop()
                except Exception: pass
                self._hotkey = None
            if new_hk:
                try:
                    self._hotkey = GlobalHotkey(new_hk, self._fire_hotkey_interrupt)
                    if self._hotkey.start():
                        print(f"[config] hotkey re-bound to {new_hk}")
                except Exception as e:
                    print(f"[config] hotkey re-bind failed: {e}")

        # TTS backend
        if new_cfg.get("tts") != old.get("tts"):
            try:
                tts_backend = load_tts_backend(new_cfg.get("tts", {}))
                # swap atomically; in-flight playback (if any) keeps using old
                self.tts.backend = tts_backend
                print("[config] tts backend reloaded")
            except Exception as e:
                print(f"[config] tts reload failed: {e}")

        # Emotion model swap: invalidate the cached detector; next utterance
        # will lazy-load the new model. This works for tag/sound_tag too.
        if (old.get("analyze", {}).get("emotion_models")
                != new_cfg.get("analyze", {}).get("emotion_models")):
            global _emotion_det
            _emotion_det = None
            print("[config] emotion models changed — will reload on next utterance")
        if (old.get("analyze", {}).get("sound_tag")
                != new_cfg.get("analyze", {}).get("sound_tag")) or (
                old.get("analyze", {}).get("emotion")
                != new_cfg.get("analyze", {}).get("emotion")):
            print("[config] analyze toggles updated")

        # Heavy fields that genuinely need a real restart (audio devices /
        # ASR model swap touches GPU state).
        notes = []
        for path in ("audio.input_device", "audio.loopback_device",
                     "audio.aec_enabled", "asr.monitor_model",
                     "asr.main_model", "asr.compute_type"):
            section, _, key = path.partition(".")
            if (old.get(section, {}).get(key) != new_cfg.get(section, {}).get(key)):
                notes.append(path)
        if notes:
            print(f"[config] these fields changed but need a full restart: {notes}")

    async def stop(self):
        if self._config_watch_task and not self._config_watch_task.done():
            self._config_watch_task.cancel()
            try: await self._config_watch_task
            except (asyncio.CancelledError, Exception): pass
            self._config_watch_task = None
        if self._bridge_watch_task and not self._bridge_watch_task.done():
            self._bridge_watch_task.cancel()
            try: await self._bridge_watch_task
            except (asyncio.CancelledError, Exception): pass
            self._bridge_watch_task = None
        # Tell the isolated VSCode to close itself. The bridge only runs in
        # the dedicated voicebridge VSCode instance, so this never closes the
        # user's main VSCode.
        try:
            await self.bridge.shutdown()
            print("[bridge] shutdown sent")
        except Exception:
            pass
        if self._tray is not None:
            try: self._tray.stop()
            except Exception: pass
            self._tray = None
        if self._hotkey is not None:
            try: self._hotkey.stop()
            except Exception: pass
            self._hotkey = None
        if self._sse_task:
            self._sse_task.cancel()
            try: await self._sse_task
            except (asyncio.CancelledError, Exception): pass
        await self.tts.stop()
        await self.watcher.stop()
        await self.pipeline.stop()

    def _fire_hotkey_interrupt(self):
        """Called from the Win32 hotkey thread. Stop TTS immediately
        (thread-safe), then schedule the rest of the state flip onto the loop."""
        if self._loop_ref is None or self.state != State.SPEAKING:
            return
        # Stop TTS synchronously — _stop_event is a threading.Event so this
        # takes effect on the very next chunk read inside _run, no waiting on
        # the asyncio loop iteration.
        try:
            if self.tts._stop_event is not None:
                self.tts._stop_event.set()
            if self.tts._proc is not None and self.tts._proc.poll() is None:
                self.tts._proc.kill()
        except Exception:
            pass
        # Schedule the carry-over + state-transition logic onto the loop.
        try:
            asyncio.run_coroutine_threadsafe(
                self._on_interrupt_hit("[hotkey]"), self._loop_ref
            )
        except Exception:
            pass

    # ---------------- transitions ----------------

    def _set_state(self, new: str):
        if new != self.state:
            print(f"[state] {self.state} -> {new}")
            self.state = new
            if self._tray is not None:
                try: self._tray.update_state(new)
                except Exception: pass

    async def _on_wake_hit(self, text: str):
        if self.state != State.IDLE:
            return
        print(f"[wake] hit: {text!r}")
        self._segments_pcm.clear()
        # Pre-roll: pull the watcher's last ~2.5s sliding window so command
        # words spoken right before / together with the wake word aren't lost.
        pre = self.watcher.snapshot_pcm_int16()
        if pre:
            secs = len(pre) / 2 / SAMPLE_RATE
            print(f"[wake] pre-roll: {secs:.2f}s")
            self._segments_pcm.append(pre)
        self.pipeline.segmenter.start_recording()
        # Now in dialog: listen for interrupt + end words
        self.watcher.set_mode("active")
        self._set_state(State.ACTIVE)

    async def _on_end_hit(self, text: str):
        """User said an end-of-dialog word. Send claude one last farewell
        prompt, play its TTS, then return to IDLE."""
        if self.state == State.IDLE:
            return
        print(f"[end] hit: {text!r}")
        # Stop any in-progress TTS / recording so we cleanly start the farewell.
        await self.tts.stop()
        self.pipeline.segmenter.stop_recording()
        self._segments_pcm.clear()
        self._carry_over_prompt = ""
        self.watcher.set_mode("wake")
        self._farewell_pending = True

        farewell_prompt = (
            f"(用户用结束词 \"{text}\" 结束了本轮对话。请用一句话简短回应,不超过 25 字,"
            "可以告别 / 答应 / 嘱咐都行,根据上下文决定;下次对话开始前用户会重新说唤醒词。)"
        )
        self._set_state(State.SUBMITTING)
        try:
            await self.bridge.post_prompt(farewell_prompt)
            print("[end] farewell prompt sent, waiting for claude")
        except Exception as e:
            print(f"[end] farewell submit failed: {e}")
            self._farewell_pending = False
            self._set_state(State.IDLE)

    async def _on_segment_ready(self, pcm_i16: bytes, t0: float, t1: float):
        if self.state != State.ACTIVE:
            return
        secs = len(pcm_i16) / 2 / SAMPLE_RATE
        print(f"[segment] flushed {secs:.2f}s of audio (segments so far: {len(self._segments_pcm) + 1})")
        self._segments_pcm.append(pcm_i16)

    async def _on_submit_signal(self):
        if self.state != State.ACTIVE:
            return
        # Flush any pending in-buffer audio explicitly: segmenter has already
        # called _flush_segment if there was a 2s gap. After 8s gap there's
        # nothing else buffered.
        self.pipeline.segmenter.stop_recording()
        if not self._segments_pcm:
            print("[submit] no audio captured, staying in dialog")
            self._segments_pcm.clear()
            self.pipeline.segmenter.start_recording()
            self.watcher.set_mode("active")
            self._set_state(State.ACTIVE)
            return
        self._set_state(State.SUBMITTING)
        # Concat all segments → run main transcribe
        all_pcm = b"".join(self._segments_pcm)
        self._segments_pcm.clear()
        audio = pcm16_to_float32(all_pcm)
        secs = audio.shape[0] / SAMPLE_RATE
        print(f"[submit] transcribing {secs:.2f}s of audio with main model...")
        try:
            await self.main_asr.ensure_loaded()
            res = await self.main_asr.transcribe_array(audio, vad_filter=True)
        except Exception as e:
            print(f"[submit] transcribe failed: {e}")
            self._segments_pcm.clear()
            self.pipeline.segmenter.start_recording()
            self.watcher.set_mode("active")
            self._set_state(State.ACTIVE)
            return
        prompt = res.text.strip()
        print(f"[submit] transcribed in {res.elapsed_s:.2f}s: {prompt!r}")
        if not prompt:
            print("[submit] empty transcription, staying in dialog")
            self._segments_pcm.clear()
            self.pipeline.segmenter.start_recording()
            self.watcher.set_mode("active")
            self._set_state(State.ACTIVE)
            return

        # P1.1: paralinguistic analysis — speech rate, F0, jitter, shimmer.
        loop = asyncio.get_event_loop()
        try:
            analysis = await loop.run_in_executor(
                None, analyze.analyze_pcm_float, audio, SAMPLE_RATE
            )
            analyze.attach_speaking_rate(analysis, prompt)
            prompt_with_meta = analyze.format_for_prompt(analysis, prompt)
            print(f"[analyze] {analysis.duration_s:.1f}s · "
                  f"f0={analysis.f0_mean_hz:.0f}Hz±{analysis.f0_std_hz:.0f} · "
                  f"rms={analysis.rms:.3f}")
        except Exception as e:
            print(f"[analyze] failed: {e}")
            prompt_with_meta = prompt

        # P1.2: YAMNet sound tags (best-effort, optional). Toggleable via config.
        sound_line = ""
        if self.config.get("analyze", {}).get("sound_tag", True):
            try:
                global _sound_tagger
                if _sound_tagger is None:
                    import sound_tag
                    _sound_tagger = sound_tag.SoundTagger()
                tags = await loop.run_in_executor(None, _sound_tagger.predict, audio, 3)
                sound_line = __import__("sound_tag").format_tags(tags)
                if sound_line:
                    print(f"[sound_tag] {sound_line}")
            except Exception as e:
                print(f"[sound_tag] skipped: {e}")

        # P1.3: emotion classification (dual zh/en model, picked by detected language)
        emotion_line = ""
        if self.config.get("analyze", {}).get("emotion", False):
            try:
                global _emotion_det
                if _emotion_det is None:
                    import emotion
                    em_cfg = self.config.get("analyze", {}).get("emotion_models", {})
                    _emotion_det = emotion.EmotionDetector(
                        zh_model=em_cfg.get("zh", emotion.DEFAULT_ZH_MODEL),
                        en_model=em_cfg.get("en", emotion.DEFAULT_EN_MODEL),
                    )
                detected_lang = getattr(res, "language", "") or self.config.get("asr", {}).get("language", "zh")
                res_emo = await loop.run_in_executor(
                    None, _emotion_det.predict, audio, detected_lang
                )
                emotion_line = __import__("emotion").format_emotion(res_emo)
                if emotion_line:
                    print(f"[emotion] {emotion_line}")
            except Exception as e:
                print(f"[emotion] skipped: {e}")

        # Fold optional sub-signals into the same low-weight "副信息" block
        # produced by analyze.format_for_prompt. We splice into the trailing
        # bracket; if it's not there (very short utterance), append a fresh one.
        extras = []
        if sound_line: extras.append(f"环境声 {sound_line}")
        if emotion_line: extras.append(emotion_line)
        if extras:
            extras_str = " · ".join(extras)
            if prompt_with_meta.endswith("]"):
                # Insert before the closing bracket, comma-separated
                prompt_with_meta = prompt_with_meta[:-1] + " · " + extras_str + "]"
            else:
                prompt_with_meta = prompt_with_meta + f"\n\n[副语音:{extras_str}]"

        # If we were interrupted earlier, prepend the carry-over context.
        full_prompt = prompt_with_meta
        if self._carry_over_prompt:
            full_prompt = (
                "(承接上一条被语音打断的请求,未完成的上下文如下:\n"
                + self._carry_over_prompt
                + "\n)\n"
                + prompt_with_meta
            )
            self._carry_over_prompt = ""

        LAST_PROMPT.write_text(full_prompt, encoding="utf-8")
        self._last_submitted_prompt = full_prompt
        try:
            await self.bridge.post_prompt(full_prompt)
            print("[submit] prompt sent, waiting for claude reply...")
        except Exception as e:
            print(f"[submit] bridge POST failed: {e}")
            self._segments_pcm.clear()
            self.pipeline.segmenter.start_recording()
            self.watcher.set_mode("active")
            self._set_state(State.ACTIVE)
            return
        # State stays SUBMITTING until SSE assistant arrives → SPEAKING

    async def _on_assistant_text(self, text: str):
        # Multiple assistant chunks may stream in; we treat the latest as the
        # full reply for this round (claude transcript jsonl gives one record
        # per turn). If we're already speaking, ignore further events for the
        # same submission.
        if self.state not in (State.SUBMITTING, State.SPEAKING):
            return
        if self.state == State.SPEAKING:
            return
        self._last_assistant_text = text
        LAST_ASSISTANT.write_text(text, encoding="utf-8")
        print(f"[claude] reply ({len(text)} chars), starting TTS...")
        self._set_state(State.SPEAKING)
        # During TTS, the assistant's own words could otherwise trip wake/end
        # detection (e.g. claude saying "再见" in the reply). Mute the watcher.
        # Hotkey is unaffected and remains the reliable interrupt path.
        self.watcher.set_mode("off")
        try:
            completed = await self.tts.speak(text)
        finally:
            # Multi-turn dialog: a natural TTS completion does NOT return us
            # to IDLE. We stay in the dialog (recording the next utterance
            # immediately) until the user says an end-word. If TTS was
            # cancelled, an interrupt handler will move state itself.
            if self.state == State.SPEAKING:
                if completed:
                    if self._farewell_pending:
                        print("[tts] farewell finished — back to IDLE")
                        self._farewell_pending = False
                        self.watcher.set_mode("wake")
                        self._set_state(State.IDLE)
                    else:
                        print("[tts] finished — continuing dialog")
                        self._segments_pcm.clear()
                        self.pipeline.segmenter.start_recording()
                        self.watcher.set_mode("active")
                        self._set_state(State.ACTIVE)
                else:
                    print("[tts] cancelled — waiting for interrupt handler")

    async def _on_api_error(self, payload: dict):
        """Bridge tells us claude's API returned an error (e.g. 400 content
        filter). Substitute a short fallback so TTS doesn't read out the raw
        error string and the dialog continues."""
        if self.state not in (State.SUBMITTING, State.SPEAKING):
            return
        if self.state == State.SPEAKING:
            return
        status = payload.get("status", 0)
        detail = payload.get("detail", "")
        print(f"[claude/api] error {status}: {detail}")
        if "content filter" in detail.lower() or status == 400:
            fallback = "嗯,这条被系统拦了,换个说法再问我一下。"
        elif status in (429,):
            fallback = "请求太密了,缓一下再问吧。"
        elif 500 <= status < 600:
            fallback = "服务器临时抽风,稍后再试。"
        else:
            fallback = "刚才出了点问题,再说一遍?"
        # Route through the same path so TTS + state machine stay consistent.
        if self._farewell_pending:
            # Edge case: error happened on the farewell prompt — just go to IDLE
            print("[end] farewell errored — back to IDLE")
            self._farewell_pending = False
            self._set_state(State.IDLE)
            self.watcher.set_mode("wake")
            return
        await self._on_assistant_text(fallback)

    async def _on_interrupt_hit(self, text: str):
        if self.state != State.SPEAKING:
            return
        print(f"[interrupt] hit: {text!r}")
        # Compose carry-over: keep what user asked + a synopsis of partial reply.
        synopsis = self._last_assistant_text[:400]
        self._carry_over_prompt = (
            f"用户之前的请求: {self._last_submitted_prompt}\n"
            f"你刚才已经开始回复(已朗读片段):\n{synopsis}"
        )
        await self.tts.stop()
        # Go back to active recording for the new utterance
        self._segments_pcm.clear()
        self.pipeline.segmenter.start_recording()
        self.watcher.set_mode("active")
        self._set_state(State.ACTIVE)


# ---------- entrypoint ----------


def load_config(path: Optional[Path] = None) -> dict:
    p = path or DEFAULT_CONFIG_PATH
    if not p.exists():
        print(f"[config] {p} not found, copying from {EXAMPLE_CONFIG_PATH}")
        if not EXAMPLE_CONFIG_PATH.exists():
            raise FileNotFoundError(f"missing both {p} and {EXAMPLE_CONFIG_PATH}")
        p.write_text(EXAMPLE_CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return json.loads(p.read_text(encoding="utf-8"))


async def main():
    args = [a for a in sys.argv[1:]]
    no_gui = "--no-gui" in args
    args = [a for a in args if a != "--no-gui"]
    path = Path(args[0]) if args else None

    if no_gui:
        config = load_config(path)
    else:
        # Block on GUI before starting any audio / model loading.
        import startup_gui
        config = startup_gui.show_startup_gui()
        if config is None:
            print("[main] startup cancelled")
            return

    agent = VoiceAgent(config)
    agent._config_path = path or DEFAULT_CONFIG_PATH
    await agent.start()
    try:
        # Wait for either Ctrl+C / Cancel or the tray's "quit" item.
        await agent._quit_requested.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await agent.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[agent] bye")
