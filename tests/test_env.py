"""
Verifies the generic GMEnv against a mock GameMaker server for both a discrete
single-agent behavior and a continuous two-agent (self-play) behavior.

Run: python -m pytest tests/test_env.py   (or: python tests/test_env.py)
"""

from __future__ import annotations

import os
import sys

import gymnasium as gym
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gmml.env import GMEnv                       # noqa: E402
from tests.mock_gm_server import MockGameMakerServer   # noqa: E402


def test_discrete_single_agent():
    srv = MockGameMakerServer(
        behaviors={"platformer": {
            "obs_size": 13,
            "action": {"type": "discrete", "branches": [3, 2]},
        }},
        agents=[{"agent_id": "1001", "behavior": "platformer", "obs_size": 13}],
        episode_len=4,
    )
    env = GMEnv(port=srv.port, connect_retries=5)
    try:
        assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
        assert list(env.action_space.nvec) == [3, 2]
        assert env.observation_space.shape == (13,)

        obs, _ = env.reset()
        assert obs.shape == (13,)

        done = False
        steps = 0
        while not done and steps < 10:
            obs, reward, done, trunc, _ = env.step(env.action_space.sample())
            steps += 1
        assert done and steps == 4
    finally:
        env.close()
        srv.stop()


def test_continuous_selfplay():
    srv = MockGameMakerServer(
        behaviors={"topdown": {
            "obs_size": 26,
            "action": {"type": "continuous", "size": 4, "low": -1.0, "high": 1.0},
        }},
        agents=[
            {"agent_id": "2001", "behavior": "topdown", "obs_size": 26},
            {"agent_id": "2002", "behavior": "topdown", "obs_size": 26},
        ],
        episode_len=3,
    )
    env = GMEnv(port=srv.port, connect_retries=5)

    class ZeroPolicy:
        def predict(self, obs, deterministic=False):
            return np.zeros(env.action_space.shape, dtype=np.float32), None

    env.set_opponent(ZeroPolicy())
    try:
        assert isinstance(env.action_space, gym.spaces.Box)
        assert env.action_space.shape == (4,)

        obs, _ = env.reset()
        assert obs.shape == (26,)
        # one learner + one opponent remembered
        assert len(env._opponent_obs) == 1

        obs, reward, done, trunc, _ = env.step(env.action_space.sample())
        assert reward == 1.0
    finally:
        env.close()
        srv.stop()


if __name__ == "__main__":
    test_discrete_single_agent()
    test_continuous_selfplay()
    print("OK — both env tests passed")
