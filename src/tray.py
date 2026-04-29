"""System tray icon for voicebridge.

Runs the pystray loop in a daemon thread so the asyncio main loop is
unaffected. Menu items call back into the main loop via a thread-safe
callback the caller wires up.
"""
from __future__ import annotations

import threading
from typing import Callable, Optional


STATE_COLORS = {
    "idle":       (60, 180, 60, 255),   # green — listening for wake word
    "active":     (255, 140, 0, 255),   # orange — recording user utterance
    "submitting": (180, 60, 220, 255),  # purple — transcribing + sending
    "speaking":   (0, 130, 230, 255),   # blue — TTS playing
}


class Tray:
    def __init__(self, on_quit: Callable[[], None],
                 on_open_settings: Optional[Callable[[], None]] = None,
                 on_show_logs: Optional[Callable[[], None]] = None,
                 tooltip: str = "Voicebridge"):
        self.on_quit = on_quit
        self.on_open_settings = on_open_settings  # double-click default
        self.on_show_logs = on_show_logs
        self.tooltip = tooltip
        self._icon = None
        self._thread: Optional[threading.Thread] = None
        self._started = threading.Event()
        self._current_state: str = "idle"

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._started.wait(timeout=2)

    def stop(self):
        if self._icon is not None:
            try: self._icon.stop()
            except Exception: pass
        self._icon = None

    def update_state(self, state: str):
        """Called by the main thread to recolor the tray icon."""
        if state == self._current_state:
            return
        self._current_state = state
        if self._icon is None:
            return
        try:
            self._icon.icon = self._make_icon(STATE_COLORS.get(state, STATE_COLORS["idle"]))
            self._icon.title = f"Voicebridge · {state}"
        except Exception:
            pass

    def _run(self):
        from pystray import Icon, Menu, MenuItem
        img = self._make_icon(STATE_COLORS["idle"])

        def settings_action(icon, item):
            if self.on_open_settings:
                try: self.on_open_settings()
                except Exception as e: print(f"[tray] settings error: {e}")

        def quit_action(icon, item):
            try: self.on_quit()
            finally: icon.stop()

        items = []
        # `default=True` makes this the action triggered by left-click /
        # double-click on the tray icon (right-click still shows the full menu).
        if self.on_open_settings is not None:
            items.append(MenuItem("设置", settings_action, default=True))
        if self.on_show_logs is not None:
            items.append(MenuItem("查看日志", lambda i, it: self.on_show_logs()))
        items.append(MenuItem("退出", quit_action))

        self._icon = Icon("voicebridge", img, self.tooltip, menu=Menu(*items))
        self._started.set()
        self._icon.run()

    def _make_icon(self, color=(255, 140, 0, 255)):
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse((6, 6, 58, 58), fill=color)
        # "W" centered (default font is small but readable at 16x16 tray size)
        try:
            font = ImageFont.truetype("arial.ttf", 36)
        except (OSError, IOError):
            font = ImageFont.load_default()
        bbox = d.textbbox((0, 0), "W", font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        d.text(((64 - w) / 2 - bbox[0], (64 - h) / 2 - bbox[1] - 2),
                "W", font=font, fill=(255, 255, 255, 255))
        return img
