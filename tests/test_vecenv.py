"""
Verifies GMVecEnv against the mock GameMaker server: N agents behind one connection,
SB3 auto-reset semantics (fresh obs on done + info['terminal_observation']), and a short
PPO smoke test to confirm SB3 drives the VecEnv end-to-end.

Run: python -m pytest tests/test_vecenv.py   (or: python tests/test_vecenv.py)
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gmml.vecenv import GMVecEnv                       # noqa: E402
from tests.mock_gm_server import (                     # noqa: E402
    LIVE_OBS_VALUE, TERMINAL_OBS_VALUE, MockGameMakerServer,
)

N = 4
OBS = 17


def _server(episode_len=3):
    return MockGameMakerServer(
        behaviors={"drone": {
            "obs_size": OBS,
            "action": {"type": "continuous", "size": 2, "low": -1.0, "high": 1.0},
        }},
        agents=[
            {"agent_id": str(3000 + i), "behavior": "drone", "obs_size": OBS}
            for i in range(N)
        ],
        episode_len=episode_len,
    )


def test_shapes_and_num_envs():
    srv = _server()
    env = GMVecEnv(port=srv.port, connect_retries=5)
    try:
        assert env.num_envs == N
        assert env.observation_space.shape == (OBS,)
        assert env.action_space.shape == (2,)

        obs = env.reset()
        assert obs.shape == (N, OBS)

        actions = np.zeros((N, 2), dtype=np.float32)
        obs, rewards, dones, infos = env.step(actions)
        assert obs.shape == (N, OBS)
        assert rewards.shape == (N,)
        assert dones.shape == (N,)
        assert len(infos) == N
        assert np.allclose(rewards, 1.0)
    finally:
        env.close()
        srv.stop()


def test_auto_reset_terminal_observation():
    srv = _server(episode_len=3)
    env = GMVecEnv(port=srv.port, connect_retries=5)
    try:
        env.reset()
        actions = np.zeros((N, 2), dtype=np.float32)
        # steps 1, 2 are live; step 3 ends the episode for every agent
        env.step(actions)
        env.step(actions)
        obs, rewards, dones, infos = env.step(actions)

        assert dones.all()
        for i in range(N):
            assert "terminal_observation" in infos[i]
            # terminal obs = the ended episode's final obs (marker)
            assert np.allclose(infos[i]["terminal_observation"], TERMINAL_OBS_VALUE)
            # returned obs = the FRESH next episode's obs
            assert np.allclose(obs[i], LIVE_OBS_VALUE)
    finally:
        env.close()
        srv.stop()


def test_ppo_smoke():
    from stable_baselines3 import PPO

    srv = _server(episode_len=3)
    env = GMVecEnv(port=srv.port, connect_retries=5)
    try:
        model = PPO(
            "MlpPolicy", env, n_steps=8, batch_size=16, n_epochs=1,
            policy_kwargs={"net_arch": [16]}, verbose=0,
        )
        model.learn(total_timesteps=64)
    finally:
        env.close()
        srv.stop()


def test_physics_pause():
    # A pause/resume around every weight update must not desync the request/response loop.
    from stable_baselines3 import PPO

    from gmml.pause import PhysicsPauseCallback

    srv = _server(episode_len=3)
    env = GMVecEnv(port=srv.port, connect_retries=5)
    try:
        model = PPO(
            "MlpPolicy", env, n_steps=8, batch_size=16, n_epochs=1,
            policy_kwargs={"net_arch": [16]}, verbose=0,
        )
        # n_steps=8 × 4 envs = 32 per rollout - 96 timesteps -> 3 rollouts -> ≥2 updates.
        model.learn(total_timesteps=96, callback=PhysicsPauseCallback(env.conn))
        assert srv.pause_count >= 1
        assert srv.resume_count >= 1
    finally:
        env.close()
        srv.stop()


if __name__ == "__main__":
    test_shapes_and_num_envs()
    test_auto_reset_terminal_observation()
    test_ppo_smoke()
    test_physics_pause()
    print("OK — all vecenv tests passed")
