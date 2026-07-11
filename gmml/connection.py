"""
Newline-delimited JSON socket client to a GameMaker server.

Framing matches the engine side (json_stringify(data) + "\\n"): one JSON object
per line. Kept deliberately simple, hardened with a connect retry loop and timeouts
so a training run started before the game window is up doesn't immediately die.
"""

from __future__ import annotations

import json
import socket
import time
from typing import Any


class GameMakerConnection:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout: float = 30.0,
        connect_retries: int = 30,
        connect_backoff: float = 1.0,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = self._connect(connect_retries, connect_backoff)
        self.buf = b""

    def _connect(self, retries: int, backoff: float) -> socket.socket:
        last_err: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
                sock.settimeout(self.timeout)
                return sock
            except OSError as err:              # server not up yet
                last_err = err
                if attempt == 1:
                    print(f"Waiting for GameMaker on {self.host}:{self.port} …")
                time.sleep(backoff)
        raise ConnectionError(
            f"Could not connect to GameMaker at {self.host}:{self.port} after {retries} attempts"
        ) from last_err

    def send(self, data: dict[str, Any]) -> None:
        self.sock.sendall((json.dumps(data) + "\n").encode("utf-8"))

    def recv(self) -> dict[str, Any]:
        while b"\n" not in self.buf:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("GameMaker disconnected")
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        return json.loads(line.decode("utf-8"))

    def close(self) -> None:
        try:
            from .protocol import build_close
            self.send(build_close())
        except OSError:
            pass
        finally:
            self.sock.close()
