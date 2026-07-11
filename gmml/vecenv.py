"""
Vectorized GameMaker environment.

Runs N same-behavior agents living in ONE GameMaker instance behind an SB3 VecEnv. 
Uses the protocol's per-agent auto-reset: the engine resets each agent the instant *it* ends 
and reports the fresh obs, with the ended episode's final obs carried in "terminal_obs". 
That satisfies SB3's auto-reset contract (step returns the next episode's first obs, the final 
obs goes to "info["terminal_observation"]") without a global scene reset. 
One blocking socket serves all N agents: the engine batches them into a
single step packet, so there is exactly one round-trip per step regardless of N.
"""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from . import protocol
from .connection import GameMakerConnection
from .env import _action_space, _obs_space, encode_action


class GMVecEnv(VecEnv):
    render_mode = None

    def __init__(self, behavior: str | None = None,
                 host: str = "localhost", port: int = 5555, **conn_kwargs):
        self.conn = GameMakerConnection(host, port, **conn_kwargs)

        # Handshake announces the auto-reset (vectorized) model to the engine
        self.conn.send(protocol.build_handshake(auto_reset=True))
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
        self.behavior_spec = self.behaviors[behavior]
        obs_space = _obs_space(self.behavior_spec)
        act_space = _action_space(self.behavior_spec.action)

        # Discover N from a reset packet and cache the (sorted) agent order so an SB3 env
        # index maps to the same engine agent for the whole run (agents are never destroyed)
        records = self._my_records(self._reset_recv())
        self._agent_ids = [r.agent_id for r in records]
        num_envs = len(self._agent_ids)
        if num_envs == 0:
            raise protocol.ProtocolError(
                f"No agents of behavior {behavior!r} present — is the room populated?"
            )

        super().__init__(num_envs, obs_space, act_space)
        self._actions: np.ndarray | None = None

    # -------------------------------------- helpers ---

    def _reset_recv(self) -> list[protocol.AgentRecord]:
        self.conn.send(protocol.build_reset())
        return protocol.parse_step(self.conn.recv())

    def _my_records(self, records: list[protocol.AgentRecord]) -> list[protocol.AgentRecord]:
        """Records for the trained behavior, deterministically ordered by agent_id."""
        mine = [r for r in records if r.behavior == self.behavior]
        if not mine:
            raise protocol.ProtocolError(
                f"No agents of behavior {self.behavior!r} in step packet"
            )
        mine.sort(key=lambda r: r.agent_id)
        return mine

    def _ordered(self, records: list[protocol.AgentRecord]) -> list[protocol.AgentRecord]:
        """Reorder a step's records to the cached env-index order."""
        by_id = {r.agent_id: r for r in self._my_records(records)}
        return [by_id[aid] for aid in self._agent_ids]

    def _stack_obs(self, records: list[protocol.AgentRecord]) -> np.ndarray:
        return np.asarray([r.obs for r in records], dtype=np.float32)

    def _indices(self, indices):
        if indices is None:
            return range(self.num_envs)
        if isinstance(indices, int):
            return [indices]
        return indices

    # -------------------------------------- VecEnv interface ---

    def reset(self):
        return self._stack_obs(self._ordered(self._reset_recv()))

    def step_async(self, actions) -> None:
        self._actions = actions

    def step_wait(self):
        action_map = {
            aid: encode_action(self.behavior_spec.action, self._actions[i])
            for i, aid in enumerate(self._agent_ids)
        }
        self.conn.send(protocol.build_action(action_map))
        ordered = self._ordered(protocol.parse_step(self.conn.recv()))

        obs = self._stack_obs(ordered)
        rewards = np.array([r.reward for r in ordered], dtype=np.float32)
        dones = np.array([r.done for r in ordered], dtype=bool)
        infos = []
        for r in ordered:
            info: dict = {}
            if r.done and r.terminal_obs is not None:
                # SB3 auto-reset: obs already holds the fresh episode, stash the final obs
                info["terminal_observation"] = np.asarray(r.terminal_obs, dtype=np.float32)
            infos.append(info)
        return obs, rewards, dones, infos

    def close(self) -> None:
        self.conn.close()

    # -------------------------------------- abstracts that don't apply to a single-connection VecEnv ---

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._indices(indices)]

    def get_attr(self, attr_name, indices=None):
        if not hasattr(self, attr_name):
            raise AttributeError(
                f"GMVecEnv has no attribute {attr_name!r} to expose via get_attr"
            )
        value = getattr(self, attr_name)
        return [value for _ in self._indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError("GMVecEnv does not support set_attr")

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError("GMVecEnv does not support env_method")
