"""Launch isolated VSCode instance with Claude Code + voice-bridge installed."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from paths import (
    INSTALL_DIR as ROOT,
    VSCODE_DATA,
    VSCODE_EXT,
    VOICE_BRIDGE_SRC,
    USER_VSCODE_EXT,
    find_vscode_cli,
)

DEFAULT_PORT = 43117


def find_user_claude_ext() -> Path:
    """Pick the highest-version anthropic.claude-code extension installed for the user."""
    candidates = sorted(USER_VSCODE_EXT.glob("anthropic.claude-code-*"))
    if not candidates:
        raise RuntimeError(
            f"No anthropic.claude-code extension found under {USER_VSCODE_EXT}. "
            "Install Claude Code in your main VSCode first."
        )
    return candidates[-1]


def ensure_claude_ext_installed():
    """Mirror the user's Claude Code extension into our isolated extensions-dir."""
    src = find_user_claude_ext()
    dst = VSCODE_EXT / src.name
    if dst.exists():
        # Refresh if user upgraded
        src_mtime = max((p.stat().st_mtime for p in src.rglob("*") if p.is_file()), default=0)
        dst_mtime = max((p.stat().st_mtime for p in dst.rglob("*") if p.is_file()), default=0)
        if src_mtime <= dst_mtime:
            return dst
        shutil.rmtree(dst)
    print(f"[launcher] copying {src.name} -> {dst}")
    shutil.copytree(src, dst, symlinks=False)
    return dst


def ensure_voice_bridge_present():
    """Sanity check that our bridge extension is in place. Source folder
    layout works as-is for VSCode (--extensions-dir scans subdirs); the
    voice-bridge.vsix in the same folder is for redistribution / installing
    into a non-isolated VSCode."""
    pkg = VOICE_BRIDGE_SRC / "package.json"
    ext = VOICE_BRIDGE_SRC / "extension.js"
    if not pkg.exists() or not ext.exists():
        raise RuntimeError(f"voice-bridge files missing under {VOICE_BRIDGE_SRC}")
    vsix = VOICE_BRIDGE_SRC / "voice-bridge.vsix"
    if vsix.exists():
        # info only — extension still loads from the source folder layout
        pass


def launch_vscode(workspace: Path, port: int = DEFAULT_PORT) -> subprocess.Popen:
    VSCODE_DATA.mkdir(parents=True, exist_ok=True)
    VSCODE_EXT.mkdir(parents=True, exist_ok=True)
    ensure_voice_bridge_present()
    ensure_claude_ext_installed()

    env = os.environ.copy()
    env["VOICE_BRIDGE_PORT"] = str(port)

    code_bin = find_vscode_cli()
    args = [
        code_bin,
        "--user-data-dir", str(VSCODE_DATA),
        "--extensions-dir", str(VSCODE_EXT),
        "--new-window",
        str(workspace),
    ]
    print(f"[launcher] launching: {' '.join(args)}")
    proc = subprocess.Popen(args, env=env, shell=False)
    return proc


def wait_for_bridge(port: int = DEFAULT_PORT, timeout: float = 60.0) -> dict:
    """Poll /status until bridge is up. Returns status dict."""
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=2) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            last_err = e
            time.sleep(0.5)
    raise RuntimeError(f"bridge did not come up on :{port} within {timeout}s: {last_err}")


def post_prompt(text: str, port: int = DEFAULT_PORT):
    body = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/prompt",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read().decode("utf-8"))


def stream_events(port: int = DEFAULT_PORT, on_assistant=None):
    """Block on SSE stream from /events. Calls on_assistant(text) for each assistant message."""
    url = f"http://127.0.0.1:{port}/events"
    with urllib.request.urlopen(url, timeout=None) as r:
        buf = b""
        while True:
            chunk = r.read(1)
            if not chunk:
                break
            buf += chunk
            if buf.endswith(b"\n\n"):
                event_block = buf.decode("utf-8", errors="replace")
                buf = b""
                event_name = None
                data_line = None
                for line in event_block.splitlines():
                    if line.startswith("event:"):
                        event_name = line[6:].strip()
                    elif line.startswith("data:"):
                        data_line = line[5:].strip()
                if event_name == "assistant" and data_line:
                    try:
                        payload = json.loads(data_line)
                        if on_assistant:
                            on_assistant(payload.get("text", ""))
                    except json.JSONDecodeError:
                        pass


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: launcher.py <workspace-path>")
        sys.exit(1)
    ws = Path(sys.argv[1]).resolve()
    if not ws.exists():
        print(f"workspace does not exist: {ws}")
        sys.exit(1)
    proc = launch_vscode(ws)
    print("[launcher] waiting for bridge to come up...")
    status = wait_for_bridge()
    print(f"[launcher] bridge ready: {status}")
