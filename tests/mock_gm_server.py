"""
A tiny in-process stand-in for the GameMaker engine, speaking the gmml protocol.

Lets us exercise the Python side (protocol / connection / GMEnv / GMVecEnv / training)
without launching GameMaker. Runs its accept/serve loop on a background thread.

Observation convention (so tests can tell episodes apart): a live obs is all 0.0; under
auto_reset, a 'done' step returns the *fresh* obs (all 0.0) plus 'terminal_obs' (all 1.0),
mirroring the engine resetting the agent in place and reporting the ended episode's final obs.
"""

from __future__ import annotations

import json
import socket
import threading

LIVE_OBS_VALUE = 0.0
TERMINAL_OBS_VALUE = 1.0


class MockGameMakerServer:
    def __init__(self, behaviors: dict, agents: list[dict],
                 episode_len: int = 5, host: str = "localhost"):
        """
        behaviors : {name: behavior-spec-dict} returned at handshake.
        agents    : list of {"agent_id", "behavior", "obs_size"} present each step.
        episode_len : steps before 'done' is reported true.
        """
        self.behaviors = behaviors
        self.agents = agents
        self.episode_len = episode_len
        self.auto_reset = False   # set from the handshake
        self.pause_count = 0      # one-way control messages received
        self.resume_count = 0

        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((host, 0))
        self._srv.listen(1)
        self.port = self._srv.getsockname()[1]

        self._t = threading.Thread(target=self._serve, daemon=True)
        self._t.start()

    # ── step packet construction ─────────────────────────────────────

    def _step_packet(self, done: bool) -> dict:
        records = []
        for a in self.agents:
            rec = {
                "agent_id": a["agent_id"],
                "behavior": a["behavior"],
                "obs": [LIVE_OBS_VALUE] * a["obs_size"],
                "reward": 1.0,
                "done": done,
            }
            # Auto-reset: the agent has been reset in place, so obs is already the fresh
            # next-episode obs and the ended episode's final obs rides along as terminal_obs.
            if done and self.auto_reset:
                rec["terminal_obs"] = [TERMINAL_OBS_VALUE] * a["obs_size"]
            records.append(rec)
        return {"type": "step", "agents": records}

    # ── serve loop ───────────────────────────────────────────────────

    def _serve(self) -> None:
        conn, _ = self._srv.accept()
        buf = b""
        step = 0
        with conn:
            while True:
                while b"\n" not in buf:
                    chunk = conn.recv(65536)
                    if not chunk:
                        return
                    buf += chunk
                line, buf = buf.split(b"\n", 1)
                msg = json.loads(line.decode())
                mtype = msg.get("type")

                if mtype == "handshake":
                    self.auto_reset = bool(msg.get("auto_reset", False))
                    conn.sendall((json.dumps({
                        "type": "spec", "protocol": 1, "behaviors": self.behaviors,
                    }) + "\n").encode())
                elif mtype == "reset":
                    step = 0
                    conn.sendall((json.dumps(self._step_packet(False)) + "\n").encode())
                elif mtype == "action":
                    step += 1
                    done = step >= self.episode_len
                    if done:
                        step = 0   # engine auto-resets in place; episodes keep cycling
                    conn.sendall((json.dumps(self._step_packet(done)) + "\n").encode())
                elif mtype == "pause":
                    self.pause_count += 1
                    conn.sendall((json.dumps({"type": "ack"}) + "\n").encode())
                elif mtype == "resume":
                    self.resume_count += 1
                    conn.sendall((json.dumps({"type": "ack"}) + "\n").encode())
                elif mtype == "close":
                    return

    def stop(self) -> None:
        try:
            self._srv.close()
        except OSError:
            pass
