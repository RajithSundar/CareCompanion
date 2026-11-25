"""Per-visitor storage.

Each browser gets a random sandbox id in an HttpOnly cookie and its own copy of the
demo data, so one visitor can never see or change another visitor's records.

* Reads of a sandbox that was never written return a fresh copy of the seed data
  without touching disk, so crawlers do not create files.
* Writes are atomic (temp file + fsync + os.replace) and serialised per sandbox
  with a lock, so concurrent requests cannot lose updates or corrupt a file.
* Old sandboxes are pruned by age, and the total count is capped.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import secrets
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator

from demo_data import DEFAULT_APP_STATE, DEFAULT_MEMORY_STORE

log = logging.getLogger("carecompanion.sandbox")

SID_RE = re.compile(r"^[0-9a-f]{32}$")


def new_sid() -> str:
    return secrets.token_hex(16)


def valid_sid(sid: str | None) -> bool:
    return bool(sid and SID_RE.match(sid))


def seed() -> Dict[str, Any]:
    return {
        "state": copy.deepcopy(DEFAULT_APP_STATE),
        "memory": copy.deepcopy(DEFAULT_MEMORY_STORE),
    }


class SandboxStore:
    def __init__(self, root: Path, max_sandboxes: int = 2000, max_age_days: float = 14):
        self.root = Path(root)
        self.max_sandboxes = max_sandboxes
        self.max_age_s = max_age_days * 86400
        self._locks: Dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        self._last_prune = 0.0

    def _path(self, sid: str) -> Path:
        if not valid_sid(sid):
            raise ValueError("invalid sandbox id")
        return self.root / f"{sid}.json"

    def _lock(self, sid: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._locks.get(sid)
            if lock is None:
                lock = self._locks[sid] = threading.Lock()
            return lock

    def _load(self, sid: str) -> Dict[str, Any]:
        path = self._path(sid)
        if not path.exists():
            return seed()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            log.exception("sandbox %s unreadable; starting from seed data", sid[:8])
            return seed()
        # Personas added to the seed after this sandbox was created.
        for pid, default in DEFAULT_APP_STATE.items():
            data.setdefault("state", {}).setdefault(pid, copy.deepcopy(default))
        for pid, default in DEFAULT_MEMORY_STORE.items():
            data.setdefault("memory", {}).setdefault(pid, copy.deepcopy(default))
        return data

    def _save(self, sid: str, data: Dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(sid)
        is_new = not path.exists()
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, separators=(",", ":"))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        if is_new:
            self.prune()

    def read(self, sid: str) -> Dict[str, Any]:
        """A snapshot. Changes to it are not saved."""
        with self._lock(sid):
            return self._load(sid)

    @contextmanager
    def edit(self, sid: str) -> Iterator[Dict[str, Any]]:
        """Load, let the caller change the data, then save atomically. Nothing is saved on error."""
        with self._lock(sid):
            data = self._load(sid)
            yield data
            self._save(sid, data)

    def prune(self, force: bool = False) -> int:
        now = time.time()
        if not force and now - self._last_prune < 60:
            return 0
        self._last_prune = now
        removed = 0
        try:
            files = sorted(
                (p for p in self.root.glob("*.json") if valid_sid(p.stem)),
                key=lambda p: p.stat().st_mtime,
            )
        except OSError:
            return 0
        for p in list(files):
            try:
                if now - p.stat().st_mtime > self.max_age_s:
                    p.unlink()
                    files.remove(p)
                    removed += 1
            except OSError:
                pass
        while len(files) > self.max_sandboxes:
            try:
                files.pop(0).unlink()
                removed += 1
            except OSError:
                pass
        for tmp in self.root.glob(".tmp-*"):
            try:
                if now - tmp.stat().st_mtime > 3600:
                    tmp.unlink()
            except OSError:
                pass
        return removed
