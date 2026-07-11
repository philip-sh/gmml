"""
Physics-pause callback.

While PPO runs a gradient update Python stops calling env.step(). GameMaker keeps running its
game loop, but agents don't receive new actions (which can often lead to agents ending up in 
unwanted states). This can produce noise in the training data and run stats. With the callbacks 
here you can you can choose to handle rollout_end and rollout_start as you see fit, such as 
freezing any physics updates for the duration and resuming afterwards.

See agent_pause()/agent_resume() hooks in the gmproject.
"""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback

from . import protocol


class PhysicsPauseCallback(BaseCallback):
    def __init__(self, conn, verbose: int = 0):
        super().__init__(verbose)
        self.conn = conn

    def _on_rollout_end(self) -> None:
        # Rollout collected, the gradient update is imminent -> freeze the engine.
        self._send_and_ack(protocol.build_pause())

    def _on_rollout_start(self) -> None:
        # Next rollout beginning (update done) -> unfreeze. The very first call, before any
        # pause, is a harmless no-op resume.
        self._send_and_ack(protocol.build_resume())

    def _send_and_ack(self, msg) -> None:
        # Recv the ack so the engine (which has no message framing) never sees this control
        # message coalesced with the next action which could drop the action and deadlock.
        self.conn.send(msg)
        self.conn.recv()

    def _on_step(self) -> bool:
        return True
