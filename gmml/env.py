"""
Generic GameMaker environment.

A single GMEnv serves every domain: it discovers the observation/action shapes from
the engine's handshake spec, so there are no per-game subclasses. One agent of the
trained behavior is the *learner* (the one Gymnasium/SB3 sees); any other agents of the
same behavior are *opponents* driven by an externally supplied policy. Single-agent
tasks are just the degenerate case with no opponents.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from . import protocol
from .connection import GameMakerConnection


def _action_space(spec: protocol.ActionSpec) -> gym.Space:
    if spec.type == protocol.ACTION_DISCRETE:
        return gym.spaces.MultiDiscrete(spec.branches)
    return gym.spaces.Box(low=spec.low, high=spec.high,
                          shape=(spec.size,), dtype=np.float32)


def _obs_space(spec: protocol.BehaviorSpec) -> gym.Space:
    return gym.spaces.Box(low=spec.obs_low, high=spec.obs_high,
                          shape=(spec.obs_size,), dtype=np.float32)


def encode_action(action_spec: protocol.ActionSpec, action) -> list:
    """Flatten an SB3 action to the wire form: ints for discrete, floats for continuous.
    Shared by GMEnv and GMVecEnv."""
    arr = np.asarray(action).reshape(-1)
    if action_spec.type == protocol.ACTION_DISCRETE:
        return [int(x) for x in arr]
    return [float(x) for x in arr]


class GMEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, behavior: str | None = None,
                 host: str = "localhost", port: int = 5555, **conn_kwargs):
        super().__init__()
        self.conn = GameMakerConnection(host, port, **conn_kwargs)

        # Handshake -> self-describing behavior specs.
        self.conn.send(protocol.build_handshake())
        self.behaviors = protocol.parse_spec(self.conn.recv())

        if behavior is None:
            if len(self.behaviors) != 1:
                raise ValueError(
                    f"Multiple behaviors {list(self.behaviors)}; specify which to train."
                )
            behavior = next(iter(self.behaviors))
        if behavior not in self.behaviors:
            raise ValueError(
                f"Behavior {behavior!r} not declared by engine; have {list(self.behaviors)}"
            )

        self.behavior = behavior
        self.spec = self.behaviors[behavior]
        self.observation_space = _obs_space(self.spec)
        self.action_space = _action_space(self.spec.action)

        # Opponent wiring (self-play). None -> single-agent, no opponents.
        self.opponent_policy = None
        self.deterministic_opponent = False
        self._opponent_obs: dict[str, np.ndarray] = {}
        self._learner_id: str | None = None

    # ── self-play hooks ──────────────────────────────────────────────

    def set_opponent(self, policy) -> None:
        self.opponent_policy = policy

    # ── internal ─────────────────────────────────────────────────────

    def _split(self, records: list[protocol.AgentRecord]):
        """Return (learner_record, [opponent_records]) for the trained behavior."""
        mine = [r for r in records if r.behavior == self.behavior]
        if not mine:
            raise protocol.ProtocolError(
                f"No agents of behavior {self.behavior!r} in step packet"
            )
        mine.sort(key=lambda r: r.agent_id)          # deterministic learner pick
        return mine[0], mine[1:]

    def _remember_opponents(self, opponents: list[protocol.AgentRecord]) -> None:
        self._opponent_obs = {
            r.agent_id: np.asarray(r.obs, dtype=np.float32) for r in opponents
        }

    def _encode_action(self, action) -> list:
        return encode_action(self.spec.action, action)

    def _opponent_actions(self) -> dict[str, list]:
        if not self._opponent_obs:
            return {}
        if self.opponent_policy is None:
            raise RuntimeError(
                "Opponent agents present but no opponent policy set — call set_opponent()."
            )
        actions = {}
        for agent_id, obs in self._opponent_obs.items():
            act, _ = self.opponent_policy.predict(
                obs, deterministic=self.deterministic_opponent
            )
            actions[agent_id] = self._encode_action(act)
        return actions

    # ── Gymnasium interface ──────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.conn.send(protocol.build_reset())
        learner, opponents = self._split(protocol.parse_step(self.conn.recv()))
        self._learner_id = learner.agent_id
        self._remember_opponents(opponents)
        return np.asarray(learner.obs, dtype=np.float32), {}

    def step(self, action):
        actions = self._opponent_actions()
        actions[self._learner_id] = self._encode_action(action)
        self.conn.send(protocol.build_action(actions))

        learner, opponents = self._split(protocol.parse_step(self.conn.recv()))
        self._learner_id = learner.agent_id
        self._remember_opponents(opponents)
        return (
            np.asarray(learner.obs, dtype=np.float32),
            learner.reward,
            learner.done,
            False,
            {},
        )

    def close(self):
        self.conn.close()
