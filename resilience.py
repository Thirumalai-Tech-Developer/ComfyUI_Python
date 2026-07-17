"""
resilience.py

Small helpers that make generate.py safe to run against a ComfyUI instance
that might crash and restart mid-batch (e.g. a Kaggle notebook that runs out
of disk/VRAM and gets relaunched with a brand new Cloudflare tunnel URL).

Three pieces:
  - ntfy.sh discovery: read the ComfyUI server's *current* public URL from a
    free, no-signup pub/sub topic (https://ntfy.sh) that kaggle_watchdog.py
    publishes to every time the tunnel restarts.
  - retry_with_backoff: generic retry loop with exponential backoff, used to
    ride out a crash-and-restart cycle instead of just dying.
  - ProgressStore: a tiny JSON-backed "which scenes/jobs are already done"
    tracker so a batch can be safely re-run after a crash without
    regenerating (or re-downloading) work that already finished.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, Optional

import requests


# ---------------------------------------------------------------------------
# ntfy.sh based server-URL discovery
# ---------------------------------------------------------------------------

def ntfy_publish(topic: str, message: str, title: Optional[str] = None,
                  server: str = "https://ntfy.sh", timeout: float = 10.0) -> bool:
    headers = {}
    if title:
        headers["Title"] = title
    try:
        r = requests.post(f"{server.rstrip('/')}/{topic}", data=message.encode("utf-8"),
                           headers=headers, timeout=timeout)
        return r.status_code == 200
    except requests.RequestException:
        return False


def ntfy_latest(topic: str, server: str = "https://ntfy.sh", timeout: float = 10.0) -> Optional[str]:
    """Return the most recent message body published to an ntfy.sh topic, or None."""
    try:
        r = requests.get(f"{server.rstrip('/')}/{topic}/json", params={"poll": "1"}, timeout=timeout)
        r.raise_for_status()
    except requests.RequestException:
        return None

    latest = None
    for line in r.text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("event") == "message" and obj.get("message"):
            latest = obj["message"]
    return latest


class ServerResolver:
    """
    Wraps a ComfyUI server URL that may change over time. If an ntfy topic is
    configured, refresh() re-polls it for a newer URL; otherwise it's just a
    fixed string.
    """

    def __init__(self, initial_server: str, ntfy_topic: Optional[str] = None):
        self.server = initial_server
        self.ntfy_topic = ntfy_topic

    def refresh(self, verbose: bool = True) -> bool:
        """Returns True if the server URL changed."""
        if not self.ntfy_topic:
            return False
        latest = ntfy_latest(self.ntfy_topic)
        if latest and latest.strip() and latest.strip() != self.server:
            old = self.server
            self.server = latest.strip()
            if verbose:
                print(f"[resolver] server URL changed: {old} -> {self.server}")
            return True
        return False


# ---------------------------------------------------------------------------
# Retry with backoff
# ---------------------------------------------------------------------------

def retry_with_backoff(
    fn: Callable[[], Any],
    is_retryable: Callable[[Exception], bool] = lambda e: True,
    max_attempts: int = 0,               # 0 = retry forever
    initial_delay: float = 5.0,
    max_delay: float = 120.0,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
) -> Any:
    """
    Calls fn() and returns its result. On an exception where is_retryable(e)
    is True, waits with exponential backoff (capped at max_delay) and tries
    again, up to max_attempts times (0 = unlimited). Re-raises the last
    exception if attempts are exhausted or the exception isn't retryable.
    """
    attempt = 0
    delay = initial_delay
    while True:
        attempt += 1
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - intentionally broad, re-raised below
            if not is_retryable(e) or (max_attempts and attempt >= max_attempts):
                raise
            if on_retry:
                on_retry(attempt, e, delay)
            time.sleep(delay)
            delay = min(delay * 1.7, max_delay)


def is_connection_error(e: Exception) -> bool:
    """True for the kinds of errors a dead/restarting ComfyUI server raises."""
    if isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                       requests.exceptions.ChunkedEncodingError)):
        return True
    # ComfyUIError message text for timeouts / HTTP failures also counts
    msg = str(e).lower()
    return any(s in msg for s in (
        "connection", "timed out", "timeout", "refused", "reset by peer",
        "name or service not known", "temporarily unavailable", "502", "503", "504",
    ))


# ---------------------------------------------------------------------------
# Crash-safe progress tracking
# ---------------------------------------------------------------------------

class ProgressStore:
    """
    JSON-backed record of which named jobs/scenes have already completed
    successfully (and which files they produced), so re-running a batch
    after a crash skips finished work instead of regenerating it.
    """

    def __init__(self, path: str):
        self.path = path
        self.data: Dict[str, Any] = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except (json.JSONDecodeError, OSError):
                self.data = {}

    def is_done(self, name: str) -> bool:
        entry = self.data.get(name)
        if not entry or entry.get("status") != "done":
            return False
        # Also make sure the files it claims to have produced still exist
        # locally -- if you deleted your outputs folder, we should redo it.
        files = entry.get("files", [])
        return bool(files) and all(os.path.exists(f) for f in files)

    def mark_done(self, name: str, files):
        self.data[name] = {"status": "done", "files": list(files), "ts": time.time()}
        self._save()

    def mark_failed(self, name: str, error: str):
        self.data[name] = {"status": "failed", "error": error, "ts": time.time()}
        self._save()

    def _save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp, self.path)
