"""
Config loading for the gmml CLI.

A training/inference run is fully described by a YAML file, so no per-domain Python is
needed. '--set a.b=c' overrides let you tweak a single field without editing the file.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

# Defaults merged under every loaded config, so YAMLs can stay terse.
DEFAULTS: dict[str, Any] = {
    "algo": "PPO",
    "total_timesteps": 1_000_000,
    "vectorized": False,                # true -> GMVecEnv (N per-agent-auto-reset envs)
    "hyperparameters": {
        "policy": "MlpPolicy",
    },
    "self_play": {
        "enabled": False,
        "pool_dir": None,               # defaults to <run_dir>/opponent_pool
        "swap_every": 10_000,
        "deterministic_opponent": False,
    },
    "output": {
        "run_dir": None,                # required in practice, see resolve()
        "checkpoint_every": 100_000,
        "final_name": "final_policy",
    },
    "connection": {
        "host": "localhost",
        "port": 5555,
    },
}


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _set_dotted(cfg: dict, dotted: str) -> None:
    """Apply 'a.b.c=value'; value parsed as a YAML scalar (so 1e-4, true, [1,2] work)."""
    key, sep, value = dotted.partition("=")
    if not sep:
        raise ValueError(f"--set expects key=value, got {dotted!r}")
    parsed = yaml.safe_load(value)
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = parsed


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    if "behavior" not in user:
        raise ValueError(f"{path}: config must set a top-level 'behavior' name")
    cfg = _deep_merge(DEFAULTS, user)
    for ov in overrides or []:
        _set_dotted(cfg, ov)
    return resolve(cfg)


def resolve(cfg: dict) -> dict:
    """Fill in derived defaults that depend on other fields."""
    if cfg.get("vectorized") and cfg["self_play"].get("enabled"):
        raise ValueError(
            "'vectorized' and 'self_play.enabled' are mutually exclusive: self-play needs "
            "the global paired reset, vectorized uses per-agent auto-reset."
        )
    run_dir = cfg["output"].get("run_dir") or f"runs/{cfg['behavior']}"
    cfg["output"]["run_dir"] = run_dir
    if cfg["self_play"].get("pool_dir") is None:
        cfg["self_play"]["pool_dir"] = str(Path(run_dir) / "opponent_pool")
    return cfg
