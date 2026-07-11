"""
Self-play opponent management.

The learner periodically snapshots itself into 'pool_dir' and swaps in a randomly sampled 
past self as the opponent. The pool directory is the single source for both storing and 
loading opponents, so a resumed run keeps accumulating rather than restarting the pool.
"""

from __future__ import annotations

import os
import random

from stable_baselines3.common.callbacks import BaseCallback


class SelfPlayCallback(BaseCallback):
    def __init__(self, env, algo_cls, pool_dir="./opponent_pool/",
                 swap_every=10_000, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.algo_cls = algo_cls
        self.pool_dir = pool_dir
        self.swap_every = swap_every
        self.last_swap = 0
        os.makedirs(pool_dir, exist_ok=True)

    def _pool(self) -> list[str]:
        return [f for f in os.listdir(self.pool_dir) if f.endswith(".zip")]

    def _load_opponent(self, filename: str) -> None:
        self.env.set_opponent(self.algo_cls.load(os.path.join(self.pool_dir, filename)))

    def _on_training_start(self) -> None:
        # Seed the pool only if empty (a resumed run reuses the accumulated pool)
        if not self._pool():
            path = os.path.join(self.pool_dir, "opponent_v0")
            self.model.save(path)
            if self.verbose:
                print("Self-play: pool initialized with v0")
        self._load_opponent(random.choice(self._pool()))

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_swap >= self.swap_every:
            snapshot = f"opponent_v{self.num_timesteps}"
            self.model.save(os.path.join(self.pool_dir, snapshot))
            chosen = random.choice(self._pool())
            self._load_opponent(chosen)
            self.last_swap = self.num_timesteps
            if self.verbose:
                print(f"Self-play: opponent swapped to {chosen}")
        return True
