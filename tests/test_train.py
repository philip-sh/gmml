"""
End-to-end smoke test of the config-driven training path against the mock server:
config load -> GMEnv -> PPO -> checkpoint + self-play callback -> final save.

Run: python tests/test_train.py
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gmml.config import load_config          # noqa: E402
from gmml.train import train                 # noqa: E402
from tests.mock_gm_server import MockGameMakerServer   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = os.path.join(os.path.dirname(HERE), "configs")


def _tiny(overrides, port, run_dir):
    return [
        f"connection.port={port}",
        "total_timesteps=64",
        "hyperparameters.n_steps=32",
        "hyperparameters.batch_size=16",
        "output.checkpoint_every=32",
        f"output.run_dir={run_dir}",
        *overrides,
    ]


def test_train_discrete():
    srv = MockGameMakerServer(
        behaviors={"platformer": {
            "obs_size": 13, "action": {"type": "discrete", "branches": [3, 2]}}},
        agents=[{"agent_id": "1", "behavior": "platformer", "obs_size": 13}],
    )
    with tempfile.TemporaryDirectory() as d:
        cfg = load_config(os.path.join(CONFIGS, "platformer.yaml"),
                          _tiny([], srv.port, d))
        train(cfg)
        assert os.path.exists(os.path.join(d, "final_policy.zip"))
    srv.stop()


def test_train_selfplay():
    srv = MockGameMakerServer(
        behaviors={"topdown": {
            "obs_size": 26,
            "action": {"type": "continuous", "size": 4, "low": -1.0, "high": 1.0}}},
        agents=[{"agent_id": "1", "behavior": "topdown", "obs_size": 26},
                {"agent_id": "2", "behavior": "topdown", "obs_size": 26}],
    )
    with tempfile.TemporaryDirectory() as d:
        cfg = load_config(os.path.join(CONFIGS, "topdown.yaml"),
                          _tiny(["self_play.swap_every=32"], srv.port, d))
        train(cfg)
        assert os.path.exists(os.path.join(d, "final_policy.zip"))
        # self-play pool got seeded + a snapshot
        pool = os.listdir(os.path.join(d, "opponent_pool"))
        assert any(f.endswith(".zip") for f in pool)
    srv.stop()


def test_train_vectorized():
    # N drones behind one connection -> GMVecEnv path selected by cfg["vectorized"].
    srv = MockGameMakerServer(
        behaviors={"drone": {
            "obs_size": 17,
            "action": {"type": "continuous", "size": 2, "low": -1.0, "high": 1.0}}},
        agents=[{"agent_id": str(3000 + i), "behavior": "drone", "obs_size": 17}
                for i in range(4)],
    )
    with tempfile.TemporaryDirectory() as d:
        # n_steps is per-env here (× 4 = 32 rollout), matching the vectorized semantics.
        cfg = load_config(os.path.join(CONFIGS, "drone.yaml"),
                          _tiny(["hyperparameters.n_steps=8"], srv.port, d))
        assert cfg["vectorized"] is True
        train(cfg)
        assert os.path.exists(os.path.join(d, "final_policy.zip"))
    srv.stop()


def test_infer_vectorized():
    # Inference must use the vectorized/auto-reset path (not GMEnv) for a multi-agent behavior.
    from gmml.infer import infer

    behaviors = {"drone": {
        "obs_size": 17,
        "action": {"type": "continuous", "size": 2, "low": -1.0, "high": 1.0}}}
    agents = [{"agent_id": str(3000 + i), "behavior": "drone", "obs_size": 17}
              for i in range(4)]

    with tempfile.TemporaryDirectory() as d:
        # 1) train briefly to produce runs/.../final_policy.zip
        srv1 = MockGameMakerServer(behaviors=behaviors, agents=agents)
        cfg = load_config(os.path.join(CONFIGS, "drone.yaml"),
                          _tiny(["hyperparameters.n_steps=8"], srv1.port, d))
        train(cfg)
        srv1.stop()

        # 2) vectorized inference over a fresh connection, bounded by episode count
        srv2 = MockGameMakerServer(behaviors=behaviors, agents=agents, episode_len=3)
        cfg2 = load_config(os.path.join(CONFIGS, "drone.yaml"),
                           _tiny(["hyperparameters.n_steps=8"], srv2.port, d))
        infer(cfg2, episodes=4)   # 4 agents finish together -> returns promptly
        srv2.stop()


if __name__ == "__main__":
    test_train_discrete()
    test_train_selfplay()
    test_train_vectorized()
    test_infer_vectorized()
    print("OK — training smoke tests passed")
