"""
Config-driven inference: run a trained policy against GameMaker. This is a socket-driven inference path
for evaluation during development. To run a policy inside a shipped game with no Python, export it with
'gmml export --format onnx' and use the in-engine runtime (ONNX export not yet implemented).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import load_config
from .env import GMEnv
from .train import _algo_cls
from .vecenv import GMVecEnv


def _default_model_path(cfg: dict) -> str:
    return str(Path(cfg["output"]["run_dir"]) / cfg["output"]["final_name"])


def infer(cfg: dict, *, model_path: str | None = None,
          episodes: int | None = None) -> None:
    if cfg.get("vectorized"):
        _infer_vectorized(cfg, model_path=model_path, episodes=episodes)
        return

    algo_cls = _algo_cls(cfg["algo"])
    env = GMEnv(
        behavior=cfg["behavior"],
        host=cfg["connection"]["host"],
        port=cfg["connection"]["port"],
    )
    model = algo_cls.load(model_path or _default_model_path(cfg))
    print("Model loaded, running inference…")

    # For self-play behaviors the same policy also drives the opponent(s).
    if cfg["self_play"].get("enabled"):
        env.set_opponent(model)
        env.deterministic_opponent = True

    ep = 0
    obs, _ = env.reset()
    try:
        while episodes is None or ep < episodes:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)
            if done:
                obs, _ = env.reset()
                ep += 1
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


def _infer_vectorized(cfg: dict, *, model_path: str | None,
                      episodes: int | None) -> None:
    """Vectorized inference: run the trained policy across all N agents on the SAME auto-reset path as training. 
    'episodes' counts total episodes finished across all agents (or run until Ctrl-C)."""
    algo_cls = _algo_cls(cfg["algo"])
    env = GMVecEnv(
        behavior=cfg["behavior"],
        host=cfg["connection"]["host"],
        port=cfg["connection"]["port"],
    )
    model = algo_cls.load(model_path or _default_model_path(cfg))
    print(f"Model loaded, running inference across {env.num_envs} agent(s)…")

    finished = 0
    obs = env.reset()
    try:
        while episodes is None or finished < episodes:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = env.step(action)   # engine auto-resets each done agent
            finished += int(np.sum(dones))
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


def main(config: str, overrides=None, **kw) -> None:
    infer(load_config(config, overrides), **kw)


if __name__ == "__main__":   # 'python -m gmml.infer ...'
    import sys
    from .cli import main as cli_main
    sys.exit(cli_main(["infer", *sys.argv[1:]]))
