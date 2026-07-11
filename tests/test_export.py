"""
Round-trip test for '.gmpolicy' export: train a tiny policy against the mock server,
export it, then parse the bytes with a pure-numpy reference reader that runs the exact
forward pass the GML runtime (scr_gmml_inference) performs. The exported policy must
reproduce model.predict(deterministic=True) - continuous means match to a tolerance,
discrete argmax indices match exactly.

The numpy reader below doubles as the executable spec for the GML port: if GML matches
this reader on a fixed observation, it matches SB3.

Run: python tests/test_export.py
"""

from __future__ import annotations

import os
import struct
import sys
import tempfile

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gmml.config import load_config          # noqa: E402
from gmml.export import export               # noqa: E402
from gmml.train import _algo_cls, train      # noqa: E402
from tests.mock_gm_server import MockGameMakerServer   # noqa: E402

ACTION_DISCRETE = 0
ACTION_CONTINUOUS = 1


# --------------------------------------------- pure-numpy reference reader (GML mirror) ---

def read_gmpolicy(path: str) -> dict:
    with open(path, "rb") as f:
        data = f.read()
    assert data[:8] == b"GMPOLICY", "bad magic"
    off = 8
    version, obs_size = struct.unpack_from("<II", data, off); off += 8
    (action_type,) = struct.unpack_from("<B", data, off); off += 1

    branches = None
    low = high = None
    if action_type == ACTION_DISCRETE:
        (n,) = struct.unpack_from("<I", data, off); off += 4
        branches = list(struct.unpack_from(f"<{n}I", data, off)); off += 4 * n
    else:
        _size, low, high = struct.unpack_from("<Iff", data, off); off += 12

    (activation,) = struct.unpack_from("<B", data, off); off += 1
    (n_layers,) = struct.unpack_from("<I", data, off); off += 4

    layers = []
    for _ in range(n_layers):
        in_f, out_f = struct.unpack_from("<II", data, off); off += 8
        w = np.frombuffer(data, dtype="<f4", count=out_f * in_f, offset=off).reshape(out_f, in_f)
        off += 4 * out_f * in_f
        b = np.frombuffer(data, dtype="<f4", count=out_f, offset=off).copy(); off += 4 * out_f
        layers.append((np.array(w), b))

    assert off == len(data), f"trailing bytes: read {off} of {len(data)}"
    return dict(version=version, obs_size=obs_size, action_type=action_type,
                branches=branches, low=low, high=high, activation=activation, layers=layers)


def gmpolicy_forward(pol: dict, obs) -> np.ndarray:
    x = np.asarray(obs, dtype=np.float32)
    n = len(pol["layers"])
    for idx, (w, b) in enumerate(pol["layers"]):
        x = w @ x + b
        if idx < n - 1:   # activation after every layer except the output head
            x = np.maximum(0.0, x) if pol["activation"] == 1 else np.tanh(x)
    if pol["action_type"] == ACTION_CONTINUOUS:
        return np.clip(x, pol["low"], pol["high"])
    out, off = [], 0
    for size in pol["branches"]:
        out.append(int(np.argmax(x[off:off + size])))
        off += size
    return np.array(out)


# --------------------------------------------- helpers ---

def _train_tiny(base: dict, behaviors: dict, agents: list, run_dir: str) -> dict:
    """Write 'base' to a temp YAML, train briefly against the mock server, return the cfg.

    Hermetic on purpose - it does not rely on any file under configs/.
    """
    cfg_path = os.path.join(run_dir, "cfg.yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(base, f)
    srv = MockGameMakerServer(behaviors=behaviors, agents=agents)
    cfg = load_config(cfg_path, [
        f"connection.port={srv.port}",
        "total_timesteps=64",
        "hyperparameters.n_steps=8",
        "hyperparameters.batch_size=16",
        "output.checkpoint_every=32",
        f"output.run_dir={run_dir}",
    ])
    train(cfg)
    srv.stop()
    return cfg


def _assert_matches(cfg: dict, gmpolicy_path: str, obs_size: int, n: int = 32) -> None:
    model = _algo_cls(cfg["algo"]).load(os.path.join(cfg["output"]["run_dir"], "final_policy"))
    pol = read_gmpolicy(gmpolicy_path)
    assert pol["obs_size"] == obs_size
    rng = np.random.default_rng(0)
    for _ in range(n):
        obs = rng.uniform(-1.0, 1.0, size=obs_size).astype(np.float32)
        sb3_action, _ = model.predict(obs, deterministic=True)
        ref_action = gmpolicy_forward(pol, obs)
        if pol["action_type"] == ACTION_CONTINUOUS:
            assert np.allclose(sb3_action, ref_action, atol=1e-4), (sb3_action, ref_action)
        else:
            assert np.array_equal(sb3_action, ref_action), (sb3_action, ref_action)


# --------------------------------------------- tests ---

def test_export_continuous():
    behaviors = {"drone": {
        "obs_size": 17,
        "action": {"type": "continuous", "size": 2, "low": -1.0, "high": 1.0}}}
    agents = [{"agent_id": str(3000 + i), "behavior": "drone", "obs_size": 17}
              for i in range(4)]
    base = {"behavior": "drone", "algo": "PPO", "vectorized": True,
            "hyperparameters": {"policy": "MlpPolicy", "net_arch": [64, 64]}}
    with tempfile.TemporaryDirectory() as d:
        cfg = _train_tiny(base, behaviors, agents, d)
        out = os.path.join(d, "drone.gmpolicy")
        export(cfg, fmt="gmpolicy", out=out)
        assert os.path.exists(out)
        _assert_matches(cfg, out, obs_size=17)


def test_export_discrete():
    behaviors = {"platformer": {
        "obs_size": 13, "action": {"type": "discrete", "branches": [3, 2]}}}
    agents = [{"agent_id": "1", "behavior": "platformer", "obs_size": 13}]
    base = {"behavior": "platformer", "algo": "PPO",
            "hyperparameters": {"policy": "MlpPolicy", "net_arch": [32, 32]}}
    with tempfile.TemporaryDirectory() as d:
        cfg = _train_tiny(base, behaviors, agents, d)
        out = os.path.join(d, "platformer.gmpolicy")
        export(cfg, fmt="gmpolicy", out=out)
        pol = read_gmpolicy(out)
        assert pol["action_type"] == ACTION_DISCRETE and pol["branches"] == [3, 2]
        _assert_matches(cfg, out, obs_size=13)


if __name__ == "__main__":
    test_export_continuous()
    test_export_discrete()
    print("OK — gmpolicy export round-trips against model.predict")
