"""Startup GUI for voicebridge.

Shows on every launch. Lets the user configure all the things that change
between sessions / setups:

  - workspace path (folder picker)
  - wake words / end words (comma-separated)
  - interrupt hotkey (TTS playback can be cut by pressing this)
  - mic input device (dropdown of input devices)
  - AEC enable toggle + loopback device dropdown (Stereo Mix / 扬声器 etc.)
  - TTS backend choice (edge / openai / python) + main field per backend

Saves to config.json. Returns the merged config dict to the caller (which is
typically main.py's launcher).

Run standalone for testing:
    python startup_gui.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from tkinter import Tk, ttk, StringVar, BooleanVar, IntVar, filedialog, messagebox
import tkinter as tk
from typing import Any, Dict, List, Optional

import sounddevice as sd

from paths import CONFIG_PATH, EXAMPLE_CONFIG_PATH as EXAMPLE_PATH

EDGE_VOICES = [
    "zh-CN-XiaoyiNeural",     # young female, clear
    "zh-CN-XiaoxiaoNeural",   # default female
    "zh-CN-YunxiNeural",      # male
    "zh-CN-YunyangNeural",    # male, news anchor
    "zh-CN-XiaohanNeural",    # female, warm
    "zh-CN-liaoning-XiaobeiNeural",  # northeastern accent
    "en-US-AriaNeural",       # english female
    "en-US-GuyNeural",        # english male
]


def load_config() -> Dict[str, Any]:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    if EXAMPLE_PATH.exists():
        return json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))
    return {}


def save_config(cfg: Dict[str, Any]) -> None:
    CONFIG_PATH.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def list_input_devices() -> List[Dict[str, Any]]:
    out = []
    try:
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                out.append({
                    "idx": i, "name": d["name"],
                    "channels": d["max_input_channels"],
                    "sr": d["default_samplerate"],
                })
    except Exception as e:
        print(f"[gui] could not list devices: {e}")
    return out


def list_loopback_candidates(devs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Heuristic: anything whose name contains 'Stereo Mix' / '立体声混音' / '扬声器'
    / 'Loopback' / 'output' is a likely loopback source on Windows."""
    keywords = ("stereo mix", "立体声混音", "扬声器", "loopback", "output")
    return [d for d in devs if any(k in d["name"].lower() for k in keywords)]


# ---------- GUI ----------


class StartupGUI:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.result: Optional[Dict[str, Any]] = None

        self.root = Tk()
        self.root.title("voicebridge · 启动配置")
        self.root.geometry("680x780")

        devs = list_input_devices()
        self.devs = devs
        self.dev_labels = [f"[{d['idx']}] {d['name']}" for d in devs]
        self.loopback_devs = list_loopback_candidates(devs)
        self.loopback_labels = [f"[{d['idx']}] {d['name']}" for d in self.loopback_devs]

        self._build_ui()
        self._populate_from_config()

    def _build_ui(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=10, pady=10)

        # --- Tab 1: basics ---
        f1 = ttk.Frame(nb, padding=10)
        nb.add(f1, text="基础")
        self._build_tab_basic(f1)

        # --- Tab 2: audio ---
        f2 = ttk.Frame(nb, padding=10)
        nb.add(f2, text="音频")
        self._build_tab_audio(f2)

        # --- Tab 3: Analyze (P1) ---
        f3a = ttk.Frame(nb, padding=10)
        nb.add(f3a, text="分析")
        self._build_tab_analyze(f3a)

        # --- Tab 4: TTS ---
        f3 = ttk.Frame(nb, padding=10)
        nb.add(f3, text="TTS")
        self._build_tab_tts(f3)

        # bottom buttons
        btns = ttk.Frame(self.root, padding=10)
        btns.pack(fill="x")
        ttk.Button(btns, text="取消", command=self._on_cancel).pack(side="right", padx=4)
        ttk.Button(btns, text="保存并启动", command=self._on_ok).pack(side="right", padx=4)

    def _build_tab_basic(self, f):
        row = 0
        ttk.Label(f, text="工作目录 (workspace)").grid(row=row, column=0, sticky="w")
        self.var_workspace = StringVar()
        e = ttk.Entry(f, textvariable=self.var_workspace, width=55)
        e.grid(row=row, column=1, sticky="ew", padx=6)
        ttk.Button(f, text="浏览...", command=self._pick_workspace).grid(row=row, column=2)
        row += 1

        ttk.Label(f, text="唤醒词 (逗号分隔)").grid(row=row, column=0, sticky="w", pady=(10, 0))
        self.var_wake = StringVar()
        ttk.Entry(f, textvariable=self.var_wake, width=55).grid(row=row, column=1, columnspan=2, sticky="ew", padx=6, pady=(10, 0))
        row += 1

        ttk.Label(f, text="结束词 (逗号分隔)").grid(row=row, column=0, sticky="w", pady=(6, 0))
        self.var_end = StringVar()
        ttk.Entry(f, textvariable=self.var_end, width=55).grid(row=row, column=1, columnspan=2, sticky="ew", padx=6, pady=(6, 0))
        row += 1

        ttk.Label(f, text="停止朗读快捷键 (TTS 播放期间按一次即停)").grid(row=row, column=0, sticky="w", pady=(6, 0))
        self.var_hotkey = StringVar()
        ttk.Entry(f, textvariable=self.var_hotkey, width=55).grid(row=row, column=1, columnspan=2, sticky="ew", padx=6, pady=(6, 0))
        row += 1

        ttk.Label(f, text="VSCode bridge URL").grid(row=row, column=0, sticky="w", pady=(20, 0))
        self.var_bridge = StringVar()
        ttk.Entry(f, textvariable=self.var_bridge, width=55).grid(row=row, column=1, columnspan=2, sticky="ew", padx=6, pady=(20, 0))
        row += 1

        # Autostart toggle (creates / removes a Startup folder shortcut).
        # autostart.py lives at the package root (next to install.bat) so
        # users can double-click it without `cd src/`. Add the parent dir
        # to sys.path before importing.
        try:
            import sys as _sys, os as _os
            _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
            if _root not in _sys.path:
                _sys.path.insert(0, _root)
            import autostart as _autostart
            self._autostart_mod = _autostart
            self.var_autostart = BooleanVar(value=_autostart.status())
            ttk.Checkbutton(
                f, text="开机自启 (Startup 快捷方式)",
                variable=self.var_autostart,
            ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(12, 0))
            row += 1
        except Exception:
            self._autostart_mod = None
            self.var_autostart = None

        ttk.Label(f, text="ASR 主模型").grid(row=row, column=0, sticky="w", pady=(20, 0))
        self.var_main_model = StringVar()
        ttk.Combobox(f, textvariable=self.var_main_model, width=53,
                     values=["large-v3-turbo", "large-v3", "medium", "small"]
                     ).grid(row=row, column=1, columnspan=2, sticky="ew", padx=6, pady=(20, 0))
        row += 1

        ttk.Label(f, text="ASR 监听模型(轻量)").grid(row=row, column=0, sticky="w", pady=(6, 0))
        self.var_monitor_model = StringVar()
        ttk.Combobox(f, textvariable=self.var_monitor_model, width=53,
                     values=["small", "tiny", "base", "medium"]
                     ).grid(row=row, column=1, columnspan=2, sticky="ew", padx=6, pady=(6, 0))
        row += 1

        ttk.Label(f, text="语言").grid(row=row, column=0, sticky="w", pady=(6, 0))
        self.var_lang = StringVar()
        ttk.Combobox(f, textvariable=self.var_lang, width=53,
                     values=["zh", "en", "ja", "ko", "auto"]
                     ).grid(row=row, column=1, columnspan=2, sticky="ew", padx=6, pady=(6, 0))
        row += 1

        f.columnconfigure(1, weight=1)

    def _build_tab_analyze(self, f):
        ttk.Label(f, text="附加到提示词的语音分析维度:").grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.var_an_sound = BooleanVar()
        ttk.Checkbutton(
            f,
            text="环境声分类 (YAMNet) — 室内安静 / 键盘 / 音乐 / 狗叫等;~17MB 模型",
            variable=self.var_an_sound,
        ).grid(row=1, column=0, sticky="w")
        self.var_an_emotion = BooleanVar()
        ttk.Checkbutton(
            f,
            text="情绪识别 (wav2vec2 valence/arousal) — 首次启用需下载 ~1.2GB 模型",
            variable=self.var_an_emotion,
        ).grid(row=2, column=0, sticky="w", pady=(4, 0))
        ttk.Label(
            f,
            text="语速 / 音高 / jitter / shimmer 这些 librosa+parselmouth 特征始终启用,不需要单独开关。",
            foreground="gray",
        ).grid(row=3, column=0, sticky="w", pady=(12, 0))

    def _build_tab_audio(self, f):
        row = 0
        ttk.Label(f, text="麦克风设备").grid(row=row, column=0, sticky="w")
        self.var_mic = StringVar()
        ttk.Combobox(f, textvariable=self.var_mic, width=60,
                     values=self.dev_labels).grid(row=row, column=1, sticky="ew", padx=6)
        row += 1

        ttk.Label(f, text="VAD 灵敏度").grid(row=row, column=0, sticky="w", pady=(10, 0))
        self.var_vad = StringVar()
        ttk.Entry(f, textvariable=self.var_vad, width=10).grid(row=row, column=1, sticky="w", padx=6, pady=(10, 0))
        ttk.Label(f, text="0.0-1.0,默认 0.5").grid(row=row, column=2, sticky="w")
        row += 1

        ttk.Label(f, text="切片静默 (秒)").grid(row=row, column=0, sticky="w", pady=(6, 0))
        self.var_pause_seg = StringVar()
        ttk.Entry(f, textvariable=self.var_pause_seg, width=10).grid(row=row, column=1, sticky="w", padx=6, pady=(6, 0))
        ttk.Label(f, text="默认 2.0").grid(row=row, column=2, sticky="w")
        row += 1

        ttk.Label(f, text="提交静默 (秒)").grid(row=row, column=0, sticky="w", pady=(6, 0))
        self.var_pause_sub = StringVar()
        ttk.Entry(f, textvariable=self.var_pause_sub, width=10).grid(row=row, column=1, sticky="w", padx=6, pady=(6, 0))
        ttk.Label(f, text="默认 4.0").grid(row=row, column=2, sticky="w")
        row += 1

        # AEC
        ttk.Separator(f, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=14)
        row += 1
        self.var_aec = BooleanVar()
        ttk.Checkbutton(f, text="启用回声消除 (AEC) — 推荐开启,改善 TTS 期间录入质量",
                        variable=self.var_aec).grid(row=row, column=0, columnspan=3, sticky="w")
        row += 1

        ttk.Label(f, text="回环设备 (扬声器输出)").grid(row=row, column=0, sticky="w", pady=(6, 0))
        self.var_loopback = StringVar()
        ttk.Combobox(f, textvariable=self.var_loopback, width=60,
                     values=self.loopback_labels).grid(row=row, column=1, columnspan=2, sticky="ew", padx=6, pady=(6, 0))
        row += 1

        ttk.Label(f, text="说明:Windows 上常见的回环设备名包含 立体声混音 / Stereo Mix / 扬声器 / Loopback。" +
                  "\n如果列表为空,在系统声音设置启用'立体声混音'。",
                  foreground="gray", justify="left").grid(row=row, column=0, columnspan=3, sticky="w", pady=(8, 0))
        row += 1

        f.columnconfigure(1, weight=1)

    def _build_tab_tts(self, f):
        row = 0
        ttk.Label(f, text="TTS 后端").grid(row=row, column=0, sticky="w")
        self.var_tts_backend = StringVar()
        rb = ttk.Frame(f); rb.grid(row=row, column=1, sticky="w")
        ttk.Radiobutton(rb, text="edge-tts (在线,免费)", variable=self.var_tts_backend, value="edge",
                        command=self._on_tts_backend_change).pack(side="left", padx=4)
        ttk.Radiobutton(rb, text="OpenAI 兼容 HTTP", variable=self.var_tts_backend, value="openai",
                        command=self._on_tts_backend_change).pack(side="left", padx=4)
        ttk.Radiobutton(rb, text="自定义 Python", variable=self.var_tts_backend, value="python",
                        command=self._on_tts_backend_change).pack(side="left", padx=4)
        row += 1

        # edge-tts panel
        self.frame_edge = ttk.LabelFrame(f, text="edge-tts", padding=8)
        self.frame_edge.grid(row=row, column=0, columnspan=3, sticky="ew", pady=10)
        ttk.Label(self.frame_edge, text="声音").grid(row=0, column=0, sticky="w")
        self.var_edge_voice = StringVar()
        ttk.Combobox(self.frame_edge, textvariable=self.var_edge_voice, values=EDGE_VOICES, width=40
                     ).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Label(self.frame_edge, text="速度").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.var_edge_rate = StringVar()
        ttk.Entry(self.frame_edge, textvariable=self.var_edge_rate, width=10
                  ).grid(row=1, column=1, sticky="w", padx=6, pady=(4, 0))
        ttk.Label(self.frame_edge, text="例 +0% / -10% / +20%", foreground="gray").grid(row=1, column=2, sticky="w")
        ttk.Label(self.frame_edge, text="音量").grid(row=2, column=0, sticky="w", pady=(4, 0))
        self.var_edge_vol = StringVar()
        ttk.Entry(self.frame_edge, textvariable=self.var_edge_vol, width=10
                  ).grid(row=2, column=1, sticky="w", padx=6, pady=(4, 0))
        ttk.Label(self.frame_edge, text="音调").grid(row=3, column=0, sticky="w", pady=(4, 0))
        self.var_edge_pitch = StringVar()
        ttk.Entry(self.frame_edge, textvariable=self.var_edge_pitch, width=10
                  ).grid(row=3, column=1, sticky="w", padx=6, pady=(4, 0))
        ttk.Label(self.frame_edge, text="例 +0Hz / -50Hz", foreground="gray").grid(row=3, column=2, sticky="w")
        self.frame_edge.columnconfigure(1, weight=1)
        row += 1

        # openai panel
        self.frame_openai = ttk.LabelFrame(f, text="OpenAI 兼容 HTTP", padding=8)
        self.frame_openai.grid(row=row, column=0, columnspan=3, sticky="ew", pady=10)
        ttk.Label(self.frame_openai, text="URL").grid(row=0, column=0, sticky="w")
        self.var_oa_url = StringVar()
        ttk.Entry(self.frame_openai, textvariable=self.var_oa_url, width=50
                  ).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Label(self.frame_openai, text="API Key").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.var_oa_key = StringVar()
        ttk.Entry(self.frame_openai, textvariable=self.var_oa_key, width=50, show="*"
                  ).grid(row=1, column=1, sticky="ew", padx=6, pady=(4, 0))
        ttk.Label(self.frame_openai, text="model").grid(row=2, column=0, sticky="w", pady=(4, 0))
        self.var_oa_model = StringVar()
        ttk.Entry(self.frame_openai, textvariable=self.var_oa_model, width=50
                  ).grid(row=2, column=1, sticky="ew", padx=6, pady=(4, 0))
        ttk.Label(self.frame_openai, text="voice").grid(row=3, column=0, sticky="w", pady=(4, 0))
        self.var_oa_voice = StringVar()
        ttk.Entry(self.frame_openai, textvariable=self.var_oa_voice, width=50
                  ).grid(row=3, column=1, sticky="ew", padx=6, pady=(4, 0))
        ttk.Label(self.frame_openai, text="format").grid(row=4, column=0, sticky="w", pady=(4, 0))
        self.var_oa_fmt = StringVar()
        ttk.Combobox(self.frame_openai, textvariable=self.var_oa_fmt,
                     values=["mp3", "wav", "opus", "aac", "flac", "pcm"], width=10
                     ).grid(row=4, column=1, sticky="w", padx=6, pady=(4, 0))
        self.frame_openai.columnconfigure(1, weight=1)
        row += 1

        # python panel
        self.frame_python = ttk.LabelFrame(f, text="自定义 Python (用户挂自己的模型)", padding=8)
        self.frame_python.grid(row=row, column=0, columnspan=3, sticky="ew", pady=10)
        ttk.Label(self.frame_python, text="脚本路径").grid(row=0, column=0, sticky="w")
        self.var_py_path = StringVar()
        ttk.Entry(self.frame_python, textvariable=self.var_py_path, width=45
                  ).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(self.frame_python, text="浏览...", command=self._pick_py).grid(row=0, column=2)
        ttk.Label(self.frame_python, text="工厂函数").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.var_py_factory = StringVar()
        ttk.Entry(self.frame_python, textvariable=self.var_py_factory, width=45
                  ).grid(row=1, column=1, sticky="ew", padx=6, pady=(4, 0))
        ttk.Label(self.frame_python, text="kwargs (JSON)").grid(row=2, column=0, sticky="w", pady=(4, 0))
        self.var_py_kwargs = StringVar()
        ttk.Entry(self.frame_python, textvariable=self.var_py_kwargs, width=45
                  ).grid(row=2, column=1, sticky="ew", padx=6, pady=(4, 0))
        ttk.Label(self.frame_python,
                  text=f"模板见 {EXAMPLE_PATH.parent.parent}\\examples\\my_tts_template.py",
                  foreground="gray").grid(row=3, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self.frame_python.columnconfigure(1, weight=1)

        f.columnconfigure(0, weight=1)

    # ---------- helpers ----------

    def _on_tts_backend_change(self):
        kind = self.var_tts_backend.get()
        # Toggle frame "feel" by enabling/disabling children. We keep all visible
        # so the user can see the alternatives.
        for frame, name in ((self.frame_edge, "edge"),
                            (self.frame_openai, "openai"),
                            (self.frame_python, "python")):
            state = "normal" if kind == name else "disabled"
            for child in frame.winfo_children():
                try: child.configure(state=state)
                except Exception: pass

    def _populate_from_config(self):
        c = self.cfg
        self.var_workspace.set(c.get("workspace", str(Path.cwd())))
        self.var_wake.set(", ".join(c.get("wake_words", ["小助手"])))
        self.var_end.set(", ".join(c.get("end_words", ["再见"])))
        self.var_hotkey.set(c.get("interrupt_hotkey", "ctrl+shift+space"))
        self.var_bridge.set(c.get("bridge", {}).get("url", "http://127.0.0.1:43117"))

        a = c.get("audio", {})
        cur_dev = a.get("input_device", -1)
        for label, d in zip(self.dev_labels, self.devs):
            if d["idx"] == cur_dev:
                self.var_mic.set(label); break
        else:
            if self.dev_labels: self.var_mic.set(self.dev_labels[0])
        self.var_vad.set(str(a.get("vad_threshold", 0.5)))
        self.var_pause_seg.set(str(a.get("pause_segment_s", 2.0)))
        self.var_pause_sub.set(str(a.get("pause_submit_s", 4.0)))
        self.var_aec.set(bool(a.get("aec_enabled", True)))
        cur_lb = a.get("loopback_device", -1)
        for label, d in zip(self.loopback_labels, self.loopback_devs):
            if d["idx"] == cur_lb:
                self.var_loopback.set(label); break
        else:
            if self.loopback_labels: self.var_loopback.set(self.loopback_labels[0])

        asr = c.get("asr", {})
        self.var_main_model.set(asr.get("main_model", "large-v3-turbo"))
        self.var_monitor_model.set(asr.get("monitor_model", "small"))
        self.var_lang.set(asr.get("language", "zh"))

        an = c.get("analyze", {})
        self.var_an_sound.set(bool(an.get("sound_tag", True)))
        self.var_an_emotion.set(bool(an.get("emotion", False)))

        t = c.get("tts", {})
        self.var_tts_backend.set(t.get("backend", "edge"))
        e = t.get("edge", {})
        self.var_edge_voice.set(e.get("voice", "zh-CN-XiaoyiNeural"))
        self.var_edge_rate.set(e.get("rate", "+0%"))
        self.var_edge_vol.set(e.get("volume", "+0%"))
        self.var_edge_pitch.set(e.get("pitch", "+0Hz"))
        o = t.get("openai", {})
        self.var_oa_url.set(o.get("url", "http://localhost:8001/v1/audio/speech"))
        self.var_oa_key.set(o.get("api_key", ""))
        self.var_oa_model.set(o.get("model", "tts-1"))
        self.var_oa_voice.set(o.get("voice", "alloy"))
        self.var_oa_fmt.set(o.get("format", "mp3"))
        p = t.get("python", {})
        self.var_py_path.set(p.get("path", str(EXAMPLE_PATH.parent.parent / "examples" / "my_tts_template.py")))
        self.var_py_factory.set(p.get("factory", "make_backend"))
        self.var_py_kwargs.set(json.dumps(p.get("kwargs", {}), ensure_ascii=False))

        self._on_tts_backend_change()

    def _pick_workspace(self):
        d = filedialog.askdirectory(initialdir=self.var_workspace.get() or str(Path.cwd()))
        if d:
            self.var_workspace.set(d)

    def _pick_py(self):
        f = filedialog.askopenfilename(filetypes=[("Python", "*.py")])
        if f:
            self.var_py_path.set(f)

    def _label_to_idx(self, label: str, devs: List[Dict[str, Any]]) -> Optional[int]:
        for d in devs:
            if f"[{d['idx']}] {d['name']}" == label:
                return d["idx"]
        return None

    def _gather(self) -> Dict[str, Any]:
        wake = [w.strip() for w in self.var_wake.get().split(",") if w.strip()]
        end = [w.strip() for w in self.var_end.get().split(",") if w.strip()]
        try: kwargs = json.loads(self.var_py_kwargs.get() or "{}")
        except json.JSONDecodeError:
            messagebox.showerror("配置错误", "Python TTS 的 kwargs 不是合法 JSON")
            return {}
        cfg = {
            "workspace": self.var_workspace.get(),
            "wake_words": wake,
            "end_words": end,
            "interrupt_hotkey": self.var_hotkey.get(),
            "bridge": {"url": self.var_bridge.get()},
            "audio": {
                "input_device": self._label_to_idx(self.var_mic.get(), self.devs),
                "sample_rate": 16000,
                "vad_threshold": float(self.var_vad.get() or 0.5),
                "pause_segment_s": float(self.var_pause_seg.get() or 2.0),
                "pause_submit_s": float(self.var_pause_sub.get() or 4.0),
                "aec_enabled": bool(self.var_aec.get()),
                "loopback_device": self._label_to_idx(self.var_loopback.get(), self.loopback_devs),
            },
            "analyze": {
                "sound_tag": bool(self.var_an_sound.get()),
                "emotion": bool(self.var_an_emotion.get()),
            },
            "asr": {
                "monitor_model": self.var_monitor_model.get(),
                "main_model": self.var_main_model.get(),
                "device": "cuda",
                "compute_type": "float16",
                "language": self.var_lang.get() or "zh",
            },
            "tts": {
                "backend": self.var_tts_backend.get(),
                "edge": {
                    "voice": self.var_edge_voice.get(),
                    "rate": self.var_edge_rate.get(),
                    "volume": self.var_edge_vol.get(),
                    "pitch": self.var_edge_pitch.get(),
                },
                "openai": {
                    "url": self.var_oa_url.get(),
                    "api_key": self.var_oa_key.get(),
                    "model": self.var_oa_model.get(),
                    "voice": self.var_oa_voice.get(),
                    "format": self.var_oa_fmt.get(),
                    "speed": 1.0,
                },
                "python": {
                    "path": self.var_py_path.get(),
                    "factory": self.var_py_factory.get(),
                    "kwargs": kwargs,
                },
            },
        }
        return cfg

    def _on_ok(self):
        cfg = self._gather()
        if not cfg:
            return
        save_config(cfg)
        # Apply autostart toggle if changed
        if self._autostart_mod is not None and self.var_autostart is not None:
            want = bool(self.var_autostart.get())
            have = self._autostart_mod.status()
            if want and not have:
                try: self._autostart_mod.enable()
                except Exception as e: print(f"[autostart] enable failed: {e}")
            elif have and not want:
                try: self._autostart_mod.disable()
                except Exception as e: print(f"[autostart] disable failed: {e}")
        self.result = cfg
        self.root.destroy()

    def _on_cancel(self):
        self.result = None
        self.root.destroy()

    def run(self) -> Optional[Dict[str, Any]]:
        self.root.mainloop()
        return self.result


def show_startup_gui() -> Optional[Dict[str, Any]]:
    cfg = load_config()
    gui = StartupGUI(cfg)
    return gui.run()


if __name__ == "__main__":
    result = show_startup_gui()
    if result is None:
        print("[gui] cancelled")
        sys.exit(1)
    print("[gui] saved config:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
