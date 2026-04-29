"""Tiny client for the voice-bridge VSCode extension's HTTP/SSE API."""
from __future__ import annotations

import asyncio
import json
import urllib.request
from typing import AsyncIterator, Awaitable, Callable, Optional


class BridgeClient:
    def __init__(self, url: str = "http://127.0.0.1:43117"):
        self.url = url.rstrip("/")

    async def status(self) -> dict:
        loop = asyncio.get_event_loop()
        def _go():
            with urllib.request.urlopen(f"{self.url}/status", timeout=3) as r:
                return json.loads(r.read())
        return await loop.run_in_executor(None, _go)

    async def post_prompt(self, text: str) -> dict:
        loop = asyncio.get_event_loop()
        def _go():
            body = json.dumps({"text": text}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.url}/prompt", data=body,
                headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                return json.loads(r.read())
        return await loop.run_in_executor(None, _go)

    async def shutdown(self) -> dict:
        """Tell the bridge extension to close its VSCode window."""
        loop = asyncio.get_event_loop()
        def _go():
            req = urllib.request.Request(
                f"{self.url}/shutdown", data=b"",
                headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(req, timeout=3) as r:
                return json.loads(r.read())
        try:
            return await loop.run_in_executor(None, _go)
        except Exception:
            return {"ok": False}

    async def stream_assistant(self, on_text: Callable[[str], Awaitable[None]],
                                stop_event: Optional[asyncio.Event] = None,
                                on_error: Optional[Callable[[dict], Awaitable[None]]] = None):
        """Long-running: pull SSE events.
            on_text(str)        — assistant text
            on_error({...})     — API error from claude (e.g. content filter
                                  block) — dict has 'status' and 'detail'
        """
        loop = asyncio.get_event_loop()
        stop_event = stop_event or asyncio.Event()

        def _read_loop():
            try:
                with urllib.request.urlopen(f"{self.url}/events", timeout=None) as r:
                    buf = b""
                    while not stop_event.is_set():
                        chunk = r.read(1)
                        if not chunk:
                            break
                        buf += chunk
                        if buf.endswith(b"\n\n"):
                            block = buf.decode("utf-8", errors="replace")
                            buf = b""
                            event = None
                            data = None
                            for line in block.splitlines():
                                if line.startswith("event:"):
                                    event = line[6:].strip()
                                elif line.startswith("data:"):
                                    data = line[5:].strip()
                            if not data:
                                continue
                            try:
                                obj = json.loads(data)
                            except json.JSONDecodeError:
                                continue
                            if event == "assistant":
                                text = obj.get("text", "")
                                if text:
                                    asyncio.run_coroutine_threadsafe(on_text(text), loop)
                            elif event == "error" and on_error is not None:
                                asyncio.run_coroutine_threadsafe(on_error(obj), loop)
            except Exception:
                pass

        await loop.run_in_executor(None, _read_loop)
