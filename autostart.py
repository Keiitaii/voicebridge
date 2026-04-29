"""voicebridge shortcut installer (top-level entry point).

Double-click `install.bat` next to this file to install all three shortcuts
in one go (autostart + desktop + Start menu).

Or invoke from the command line:
    python autostart.py status              # what's installed
    python autostart.py enable              # autostart only (no GUI)
    python autostart.py desktop             # desktop only
    python autostart.py startmenu           # Start menu only
    python autostart.py install-all         # all three
    python autostart.py disable             # remove autostart only
    python autostart.py uninstall-all       # remove all three

This script is intentionally self-contained — it doesn't import anything
from src/, so it works even before voicebridge's Python deps are installed.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Tuple

INSTALL_DIR = Path(__file__).resolve().parent
ENTRY = INSTALL_DIR / "src" / "main.py"
APPDATA = Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
USERPROFILE = Path.home()

STARTUP_DIR = APPDATA / r"Microsoft\Windows\Start Menu\Programs\Startup"
DESKTOP_DIR = USERPROFILE / "Desktop"
STARTMENU_DIR = APPDATA / r"Microsoft\Windows\Start Menu\Programs"

SHORTCUT_NAME = "voicebridge.lnk"


def find_python_exe() -> Path:
    """Resolve a python.exe to use for shortcuts. Prefer pythonw.exe (no
    console window) when running headless / from a shortcut."""
    here = Path(sys.executable)
    candidates = [
        here.with_name("pythonw.exe"),
        here,
    ]
    which = shutil.which("pythonw") or shutil.which("python")
    if which:
        candidates.append(Path(which))
    for c in candidates:
        if c and c.exists():
            return c
    return Path("python.exe")


def _create_shortcut(target_lnk: Path, with_gui: bool, minimized: bool,
                     description: str) -> Path:
    import pythoncom
    from win32com.client import Dispatch
    pythoncom.CoInitialize()
    try:
        shell = Dispatch("WScript.Shell")
        sc = shell.CreateShortcut(str(target_lnk))
        py = find_python_exe()
        sc.Targetpath = str(py)
        if with_gui:
            sc.Arguments = f'"{ENTRY}"'
        else:
            sc.Arguments = f'"{ENTRY}" --no-gui'
        sc.WorkingDirectory = str(ENTRY.parent)
        sc.IconLocation = str(py)
        sc.Description = description
        sc.WindowStyle = 7 if minimized else 1
        sc.save()
    finally:
        pythoncom.CoUninitialize()
    return target_lnk


def enable_autostart() -> Path:
    return _create_shortcut(STARTUP_DIR / SHORTCUT_NAME,
                             with_gui=False, minimized=True,
                             description="Voicebridge (auto-start)")


def install_desktop() -> Path:
    return _create_shortcut(DESKTOP_DIR / SHORTCUT_NAME,
                             with_gui=True, minimized=False,
                             description="Voicebridge")


def install_startmenu() -> Path:
    return _create_shortcut(STARTMENU_DIR / SHORTCUT_NAME,
                             with_gui=True, minimized=False,
                             description="Voicebridge")


def disable_autostart() -> bool:
    p = STARTUP_DIR / SHORTCUT_NAME
    if p.exists(): p.unlink(); return True
    return False


def uninstall_desktop() -> bool:
    p = DESKTOP_DIR / SHORTCUT_NAME
    if p.exists(): p.unlink(); return True
    return False


def uninstall_startmenu() -> bool:
    p = STARTMENU_DIR / SHORTCUT_NAME
    if p.exists(): p.unlink(); return True
    return False


def status() -> Tuple[bool, bool, bool]:
    return (
        (STARTUP_DIR / SHORTCUT_NAME).exists(),
        (DESKTOP_DIR / SHORTCUT_NAME).exists(),
        (STARTMENU_DIR / SHORTCUT_NAME).exists(),
    )


# Backwards-compat aliases
def enable() -> Path: return enable_autostart()
def disable() -> bool: return disable_autostart()


if __name__ == "__main__":
    op = (sys.argv[1] if len(sys.argv) > 1 else "install-all").lower()
    if op in ("enable", "autostart"):
        p = enable_autostart()
        print(f"[autostart] enabled → {p}")
    elif op == "desktop":
        p = install_desktop()
        print(f"[desktop] shortcut → {p}")
    elif op == "startmenu":
        p = install_startmenu()
        print(f"[startmenu] shortcut → {p}")
    elif op == "install-all":
        a, d, s = enable_autostart(), install_desktop(), install_startmenu()
        print("[install-all] done. Three shortcuts installed:")
        print(f"  autostart : {a}")
        print(f"  desktop   : {d}")
        print(f"  startmenu : {s}")
        print()
        print("Double-click the desktop icon, or search 'voicebridge' in Start.")
    elif op == "disable":
        ok = disable_autostart()
        print(f"[autostart] {'removed' if ok else 'no shortcut found'}")
    elif op == "uninstall-all":
        a = disable_autostart(); d = uninstall_desktop(); s = uninstall_startmenu()
        print(f"[uninstall-all] autostart={a}, desktop={d}, startmenu={s}")
    elif op == "status":
        a, d, s = status()
        print(f"[autostart]  {'on' if a else 'off'}  ({STARTUP_DIR / SHORTCUT_NAME})")
        print(f"[desktop]    {'on' if d else 'off'}  ({DESKTOP_DIR / SHORTCUT_NAME})")
        print(f"[startmenu]  {'on' if s else 'off'}  ({STARTMENU_DIR / SHORTCUT_NAME})")
    else:
        print(f"unknown op: {op}")
        print(__doc__)
