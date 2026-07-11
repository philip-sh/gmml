"""
Config-driven training. One entry point for every behavior. The domain lives in
GameMaker, the hyperparameters live in YAML.
"""

from __future__ import annotations

import re
from pathlib import Path

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

from .config import load_config
from .env import GMEnv
from .pause import PhysicsPauseCallback
from .selfplay import SelfPlayCallback
from .vecenv import GMVecEnv

ALGOS = {"PPO": PPO, "A2C": A2C}

# hyperparameter keys that are not passed directly to the algo constructor
_POLICY_KEYS = {"policy", "net_arch", "activation_fn"}


def _algo_cls(name: str):
    try:
        return ALGOS[name.upper()]
    except KeyError:
        raise ValueError(f"Unknown algo {name!r}; supported: {sorted(ALGOS)}")


def _split_hyperparams(hp: dict):
    """Return (policy, policy_kwargs, algo_kwargs)."""
    policy = hp.get("policy", "MlpPolicy")
    policy_kwargs = {}
    if "net_arch" in hp:
        policy_kwargs["net_arch"] = hp["net_arch"]
    algo_kwargs = {k: v for k, v in hp.items() if k not in _POLICY_KEYS}
    return policy, policy_kwargs, algo_kwargs


def _latest_checkpoint(ckpt_dir: Path) -> Path | None:
    if not ckpt_dir.is_dir():
        return None
    ckpts = list(ckpt_dir.glob("ckpt_*_steps.zip"))
    if not ckpts:
        return None
    # order by embedded step count, falling back to mtime
    def steps(p: Path) -> int:
        m = re.search(r"ckpt_(\d+)_steps", p.stem)
        return int(m.group(1)) if m else 0
    return max(ckpts, key=steps)


def _clear_pool(pool_dir: str) -> None:
    """Delete opponent snapshots so self-play reseeds v0 from the current learner."""
    pool = Path(pool_dir)
    snapshots = list(pool.glob("*.zip")) if pool.is_dir() else []
    for f in snapshots:
        f.unlink()
    print(f"Reset opponent pool: removed {len(snapshots)} snapshot(s) from {pool_dir}")


def train(cfg: dict, *, resume: bool = False,
          resume_from: str | None = None, reset_timesteps: bool = False,
          reset_pool: bool = False) -> None:
    algo_cls = _algo_cls(cfg["algo"])
    run_dir = Path(cfg["output"]["run_dir"])
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = run_dir / "tensorboard"

    # Vectorized (GMVecEnv): N per-agent-auto-reset envs behind one connection
    # Otherwise the single-learner GMEnv (also the self-play path)
    vectorized = bool(cfg.get("vectorized"))
    env_cls = GMVecEnv if vectorized else GMEnv
    env = env_cls(
        behavior=cfg["behavior"],
        host=cfg["connection"]["host"],
        port=cfg["connection"]["port"],
    )
    conn = env.conn   # grab before VecMonitor hides it. used by the physics-pause callback
    # SB3 auto-wraps a single Gym env in Monitor but leaves a VecEnv alone
    # Add VecMonitor here so the vectorized path logs the same episode stats
    if vectorized:
        env = VecMonitor(env)

    # Resolve a checkpoint to resume from, if any
    resume_path = resume_from
    if resume and resume_path is None:
        latest = _latest_checkpoint(ckpt_dir)
        if latest is None:
            print("--resume: no checkpoint found, starting fresh.")
        else:
            resume_path = str(latest)

    policy, policy_kwargs, algo_kwargs = _split_hyperparams(cfg["hyperparameters"])

    if resume_path:
        print(f"Resuming from {resume_path}")
        # let YAML hyperparams override what was baked into the checkpoint
        model = algo_cls.load(
            resume_path, env=env, tensorboard_log=str(tb_dir),
            custom_objects=algo_kwargs,
        )
    else:
        model = algo_cls(
            policy, env, verbose=1, tensorboard_log=str(tb_dir),
            policy_kwargs=policy_kwargs, **algo_kwargs,
        )

    callbacks = [
        CheckpointCallback(
            save_freq=cfg["output"]["checkpoint_every"],
            save_path=str(ckpt_dir), name_prefix="ckpt",
        ),
        PhysicsPauseCallback(conn, verbose=1),
    ]

    sp = cfg["self_play"]
    if sp.get("enabled"):
        if reset_pool:
            _clear_pool(sp["pool_dir"])
        env.deterministic_opponent = bool(sp.get("deterministic_opponent", False))
        callbacks.append(SelfPlayCallback(
            env=env, algo_cls=algo_cls,
            pool_dir=sp["pool_dir"], swap_every=sp["swap_every"], verbose=1,
        ))
    elif reset_pool:
        print("--reset-pool ignored: self_play is not enabled for this config.")

    print(f"Training behavior '{cfg['behavior']}' for {cfg['total_timesteps']} steps…")
    try:
        model.learn(
            total_timesteps=cfg["total_timesteps"],
            callback=CallbackList(callbacks),
            reset_num_timesteps=reset_timesteps or not resume_path,
        )
    except KeyboardInterrupt:
        print("Interrupted, saving…")

    final = run_dir / cfg["output"]["final_name"]
    model.save(str(final))
    print(f"Saved final policy to {final}.zip")
    env.close()


def main(config: str, overrides=None, **kw) -> None:
    train(load_config(config, overrides), **kw)


if __name__ == "__main__":   # 'python -m gmml.train ...'
    import sys
    from .cli import main as cli_main
    sys.exit(cli_main(["train", *sys.argv[1:]]))
