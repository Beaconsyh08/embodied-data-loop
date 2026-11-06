"""Bounded, owner-specific CPU viewer sessions; computation remains in Agent jobs."""

from __future__ import annotations

import atexit
import json
import os
import re
import signal
import socket
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ViewerSession:
    session_id: str
    owner: str
    job_id: str
    artifact: str
    process: subprocess.Popen
    log: Path
    touched: float

    def describe(self):
        status, port = "starting", None
        if self.process.poll() is not None:
            status = "error"
        elif self.log.exists():
            match = re.search(r"Viser ready: http://127\.0\.0\.1:(\d+)", self.log.read_text(errors="replace"))
            if match:
                status, port = "ready", int(match[1])
        return {
            "session_id": self.session_id,
            "status": status,
            "port": port,
            "job_id": self.job_id,
            "artifact": self.artifact,
        }


class ViewerSessions:
    def __init__(self, root, *, idle_seconds=1800, maximum=4):
        self.root = Path(root)
        self.idle_seconds, self.maximum = idle_seconds, maximum
        self.sessions = {}
        self.lock = threading.RLock()
        self.stopped = threading.Event()
        self.thread = None

    def start(self, owner, job_id, artifact, bundle):
        from lerobot.data_platform.operators.implementations.umi import native_viewer
        from lerobot.data_platform.operators.implementations.umi.runtime import LAUNDRY_COMMIT

        filename = os.environ.get("DATA_PLATFORM_UMI_VIEWER_RUNTIME", "")
        if not filename or not Path(filename).is_absolute():
            raise ValueError("The server's UMI viewer runtime is not configured")
        config = json.loads(Path(filename).read_text())
        if config.get("laundry_commit") != LAUNDRY_COMMIT:
            raise ValueError("UMI viewer runtime must match the pinned laundry source")
        python, laundry = Path(config["python"]), Path(config["laundry_root"])
        if not python.is_absolute() or not os.access(python, os.X_OK) or not laundry.is_absolute():
            raise ValueError("UMI viewer runtime is unavailable")
        with self.lock:
            self.expire()
            for session in self.sessions.values():
                if (session.owner, session.job_id, session.artifact) == (owner, job_id, artifact):
                    session.touched = time.monotonic()
                    return session
            if (
                len(self.sessions) >= self.maximum
                or sum(s.owner == owner for s in self.sessions.values()) >= 2
            ):
                raise ValueError("Close an existing UMI viewer before opening another")
            self.root.mkdir(parents=True, exist_ok=True)
            session_id = str(uuid.uuid4())
            log = self.root / f"{session_id}.log"
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env.pop("PYTHONHOME", None)
            env["CUDA_VISIBLE_DEVICES"] = ""
            with log.open("w") as output:
                process = subprocess.Popen(
                    [str(python), "-u", native_viewer.__file__, str(bundle), str(laundry), str(port)],
                    env=env,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            session = ViewerSession(session_id, owner, job_id, artifact, process, log, time.monotonic())
            self.sessions[session_id] = session
            if self.thread is None:
                self.thread = threading.Thread(target=self._reaper, daemon=True, name="umi-viewer-reaper")
                self.thread.start()
                atexit.register(self.close)
            return session

    def get(self, session_id, owner):
        with self.lock:
            session = self.sessions.get(session_id)
            if session is None or session.owner != owner:
                raise KeyError(session_id)
            session.touched = time.monotonic()
            return session

    def remove(self, session_id, owner):
        with self.lock:
            session = self.get(session_id, owner)
            self.sessions.pop(session_id)
            if session.process.poll() is None:
                os.killpg(session.process.pid, signal.SIGTERM)
                try:
                    session.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(session.process.pid, signal.SIGKILL)
                    session.process.wait(timeout=5)
            session.log.unlink(missing_ok=True)

    def expire(self):
        with self.lock:
            for session in list(self.sessions.values()):
                if time.monotonic() - session.touched > self.idle_seconds:
                    self.remove(session.session_id, session.owner)

    def _reaper(self):
        while not self.stopped.wait(15):
            self.expire()

    def close(self):
        self.stopped.set()
        with self.lock:
            for session in list(self.sessions.values()):
                self.remove(session.session_id, session.owner)
