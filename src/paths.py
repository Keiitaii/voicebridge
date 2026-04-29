"""Centralised paths so the package is portable.

Every file derives runtime / config locations from `INSTALL_DIR` (the parent
of this `src/` folder) so the package works from any installation path —
no user-specific drives baked in.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
INSTALL_DIR = SRC_DIR.parent

# Layout under INSTALL_DIR
CONFIG_PATH = INSTALL_DIR / "config.json"
EXAMPLE_CONFIG_PATH = INSTALL_DIR / "config.example.json"
VSCODE_DATA = INSTALL_DIR / "vscode-data"
VSCODE_EXT = INSTALL_DIR / "vscode-ext"
VOICE_BRIDGE_SRC = VSCODE_EXT / "voice-bridge"

# Runtime artefacts (debug + transcript dumps)
LAST_PROMPT = INSTALL_DIR / ".last_prompt.txt"
LAST_ASSISTANT = INSTALL_DIR / ".last_assistant.txt"
LAST_TRANSCRIBE = INSTALL_DIR / ".last_transcribe.txt"
FFPLAY_LOG = INSTALL_DIR / ".ffplay.log"

# User-scope locations (per-OS env vars)
HOME = Path.home()
APPDATA = Path(os.environ.get("APPDATA", str(HOME / "AppData" / "Roaming")))
LOCALAPPDATA = Path(os.environ.get("LOCALAPPDATA", str(HOME / "AppData" / "Local")))
USER_VSCODE_EXT = HOME / ".vscode" / "extensions"


def find_python_exe() -> Path:
    """Resolve a python.exe to use for shortcuts. Prefer pythonw.exe (no
    console window) when running headless / from a shortcut. Falls back to
    sys.executable, and finally to the first python on PATH."""
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
    return Path("python.exe")  # last resort, hope it's on PATH


def find_vscode_cli() -> str:
    """Resolve the VSCode CLI launcher (`code.cmd` on Windows). Tries PATH,
    then a few standard install locations."""
    found = shutil.which("code") or shutil.which("code.cmd")
    if found:
        return found
    candidates = [
        LOCALAPPDATA / "Programs" / "Microsoft VS Code" / "bin" / "code.cmd",
        Path(r"C:\Program Files\Microsoft VS Code\bin\code.cmd"),
        Path(r"C:\Program Files (x86)\Microsoft VS Code\bin\code.cmd"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise RuntimeError(
        "VSCode CLI 'code' not found. Install VSCode and ensure 'code' is on PATH "
        "(or its bin folder is reachable)."
    )
