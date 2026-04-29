"""Global hotkey via Win32 RegisterHotKey + thread message loop.

The `keyboard` Python library proved flaky on this Windows 11 build (first
press of a multi-key combo dropped, sometimes 3-4 presses needed). The Win32
API path is rock-solid: the OS kernel injects WM_HOTKEY directly into our
thread's message queue with no userland race.

Usage:
    h = GlobalHotkey("ctrl+shift+space", on_press_callback)
    h.start()
    # ...
    h.stop()

Modifier names: ctrl|control, alt, shift, win|super|meta (case-insensitive).
Key names: a-z, 0-9, f1-f24, plus a few specials (space, enter, tab, esc,
backspace, ins, del, home, end, pgup, pgdn, up, down, left, right, pause,
printscreen).
"""
from __future__ import annotations

import ctypes
import threading
from ctypes import wintypes
from typing import Callable, Optional

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

WM_HOTKEY = 0x0312
WM_QUIT = 0x0012

MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_NOREPEAT = 0x4000
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008

MOD_MAP = {
    "ctrl": MOD_CONTROL, "control": MOD_CONTROL,
    "alt": MOD_ALT, "option": MOD_ALT,
    "shift": MOD_SHIFT,
    "win": MOD_WIN, "super": MOD_WIN, "meta": MOD_WIN, "cmd": MOD_WIN,
}

VK_SPECIALS = {
    "space": 0x20, "enter": 0x0D, "return": 0x0D, "tab": 0x09, "esc": 0x1B,
    "escape": 0x1B, "backspace": 0x08, "ins": 0x2D, "insert": 0x2D,
    "del": 0x2E, "delete": 0x2E, "home": 0x24, "end": 0x23,
    "pgup": 0x21, "pageup": 0x21, "pgdn": 0x22, "pagedown": 0x22,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    "pause": 0x13, "printscreen": 0x2C, "prtsc": 0x2C, "scrolllock": 0x91,
    "numlock": 0x90, "capslock": 0x14,
}


def _parse(hotkey: str) -> tuple[int, int]:
    parts = [p.strip().lower() for p in hotkey.split("+") if p.strip()]
    if not parts:
        raise ValueError(f"empty hotkey: {hotkey!r}")
    mods = 0
    vk: Optional[int] = None
    for p in parts:
        if p in MOD_MAP:
            mods |= MOD_MAP[p]
        elif p in VK_SPECIALS:
            vk = VK_SPECIALS[p]
        elif len(p) == 1 and p.isalnum():
            vk = ord(p.upper())
        elif p.startswith("f") and p[1:].isdigit():
            n = int(p[1:])
            if 1 <= n <= 24:
                vk = 0x6F + n  # F1=0x70, F2=0x71, ..., F24=0x87
            else:
                raise ValueError(f"invalid F-key: {p!r}")
        else:
            raise ValueError(f"unknown hotkey token: {p!r}")
    if vk is None:
        raise ValueError(f"hotkey must include a non-modifier key: {hotkey!r}")
    return mods, vk


class GlobalHotkey:
    """Single-shot global hotkey. Call start() to register, stop() to unregister."""

    def __init__(self, hotkey: str, callback: Callable[[], None]):
        self.hotkey = hotkey
        self.callback = callback
        self._mods, self._vk = _parse(hotkey)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._tid: Optional[int] = None
        self._ready = threading.Event()
        self._ok = False

    def start(self) -> bool:
        if self._thread is not None:
            return self._ok
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        # Wait for registration to either succeed or fail
        self._ready.wait(timeout=2.0)
        return self._ok

    def stop(self):
        if self._thread is None:
            return
        self._stop.set()
        if self._tid is not None:
            user32.PostThreadMessageW(wintypes.DWORD(self._tid), WM_QUIT, 0, 0)
        self._thread.join(timeout=1)
        self._thread = None

    def _run(self):
        import time
        self._tid = kernel32.GetCurrentThreadId()
        hotkey_id = 1
        # Retry up to 3 times — winerr 1409 (HOTKEY_ALREADY_REGISTERED) often
        # happens right after a previous voicebridge process exited because
        # the OS hasn't fully released its hotkey yet.
        ok = False
        for attempt in range(3):
            ok = user32.RegisterHotKey(None, hotkey_id,
                                        self._mods | MOD_NOREPEAT, self._vk)
            if ok:
                break
            err = ctypes.get_last_error()
            if attempt < 2:
                print(f"[hotkey] RegisterHotKey failed (attempt {attempt+1}/3) winerr={err}; retrying...")
                time.sleep(0.7)
            else:
                print(f"[hotkey] RegisterHotKey failed: hotkey={self.hotkey!r} winerr={err}")
        if not ok:
            self._ok = False
            self._ready.set()
            return
        self._ok = True
        self._ready.set()
        try:
            msg = wintypes.MSG()
            while not self._stop.is_set():
                bret = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
                if bret == 0 or bret == -1:
                    break
                if msg.message == WM_HOTKEY:
                    try:
                        self.callback()
                    except Exception as e:
                        print(f"[hotkey] callback error: {e}")
        finally:
            user32.UnregisterHotKey(None, hotkey_id)


if __name__ == "__main__":
    import time
    print("Press Ctrl+Shift+Space (or change the binding below)...")
    counter = {"n": 0}
    def cb():
        counter["n"] += 1
        print(f"  hit #{counter['n']}")
    h = GlobalHotkey("ctrl+shift+space", cb)
    if not h.start():
        print("registration failed"); raise SystemExit(1)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        h.stop()
