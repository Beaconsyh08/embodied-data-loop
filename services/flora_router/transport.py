"""Bounded-memory multipart forwarding with normal certificate verification."""

from __future__ import annotations

import os
import secrets
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter


class LocalTLSAdapter(HTTPAdapter):
    """Connect to a local forward while validating the backend's real certificate identity."""

    def __init__(self, expected_host: str):
        self.expected_host = expected_host
        super().__init__(max_retries=0)

    def init_poolmanager(self, connections, maxsize, block=False, **kwargs):
        kwargs.update(assert_hostname=self.expected_host, server_hostname=self.expected_host)
        super().init_poolmanager(connections, maxsize, block=block, **kwargs)


class Multipart:
    """A rewind-free stream; files are reopened on a deliberate idempotent retry."""

    def __init__(self, instruction: str, videos: dict[str, Path]):
        if not instruction or set(videos) not in ({"main"}, {"left", "right"}, {"left", "right", "main"}):
            raise ValueError("Invalid FLORA forwarding input")
        self.boundary = "flora-" + secrets.token_hex(16)
        self.stack = ExitStack()
        self.parts: list[bytes | object] = []
        self.sizes: list[int] = []
        self.index = 0
        self.offset = 0

        def add_bytes(value: bytes):
            self.parts.append(value)
            self.sizes.append(len(value))

        add_bytes(
            f'--{self.boundary}\r\nContent-Disposition: form-data; name="instruction"\r\n\r\n'.encode()
            + instruction.encode("utf-8")
            + b"\r\n"
        )
        try:
            for role in ("left", "right", "main"):
                if role not in videos:
                    continue
                path = videos[role]
                if path.is_symlink() or not path.is_file():
                    raise ValueError("Forwarding video is unavailable")
                add_bytes(
                    f'--{self.boundary}\r\nContent-Disposition: form-data; name="{role}_video"; '
                    f'filename="{role}.mp4"\r\nContent-Type: video/mp4\r\n\r\n'.encode()
                )
                handle = self.stack.enter_context(path.open("rb"))
                self.parts.append(handle)
                self.sizes.append(os.fstat(handle.fileno()).st_size)
                add_bytes(b"\r\n")
            add_bytes(f"--{self.boundary}--\r\n".encode())
        except BaseException:
            self.close()
            raise

    def __len__(self):
        return sum(self.sizes)

    def read(self, size: int = -1) -> bytes:
        # requests/urllib3 uses bounded reads; avoid buffering whole videos even for read(-1).
        remaining = 1024 * 1024 if size is None or size < 0 else size
        output = bytearray()
        while remaining and self.index < len(self.parts):
            part = self.parts[self.index]
            if isinstance(part, bytes):
                value = part[self.offset : self.offset + remaining]
                self.offset += len(value)
                finished = self.offset == len(part)
            else:
                value = part.read(remaining)
                finished = not value
            output.extend(value)
            remaining -= len(value)
            if finished:
                self.index += 1
                self.offset = 0
        return bytes(output)

    def close(self):
        self.stack.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


@dataclass(frozen=True)
class Backend:
    name: str
    url: str
    hostname: str
    ca: Path
    key_file: Path

    def session(self) -> requests.Session:
        if self.key_file.is_symlink() or self.key_file.stat().st_mode & 0o077:
            raise ValueError("Backend key must be a private regular file")
        token = self.key_file.read_text().strip()
        if not 32 <= len(token) <= 256:
            raise ValueError("Backend key is unavailable")
        session = requests.Session()
        session.trust_env = False
        session.verify = str(self.ca)
        session.headers["Authorization"] = "Bearer " + token
        session.mount(self.url.rstrip("/") + "/", LocalTLSAdapter(self.hostname))
        return session
