"""
gmml wire protocol - domain-agnostic messages between GameMaker and Python.

GameMaker is the TCP *server*, Python is the *client*. Messages are newline-delimited
JSON. The engine owns all domain logic (what an observation is, how an action maps to
gameplay), Python only sees generic vectors described by a self-reported behavior spec.

Message flow
---------------------------------------------
    Python -> GM : {"type": "handshake", "protocol": 1, "auto_reset": false}
    GM -> Python : {"type": "spec", "protocol": 1, "behaviors": {<name>: <behavior spec>}}
    Python -> GM : {"type": "reset"}
    GM -> Python : {"type": "step", "agents": [<agent record>, ...]}
    Python -> GM : {"type": "action", "actions": {<agent_id>: [floats], ...}}
    GM -> Python : {"type": "step", "agents": [<agent record>, ...]}
    Python -> GM : {"type": "close"}

Control messages (sent between step loops, e.g. from a callback). GM's raw TCP socket has no
framing, so every Python->GM message MUST be followed by a reply before the next send or two
messages coalesce and one is lost, hence pause/resume are ack'd, not one-way:
    Python -> GM : {"type": "pause"}    # freeze physics/game-loop during a weight update
    GM -> Python : {"type": "ack"}
    Python -> GM : {"type": "resume"}   # unfreeze
    GM -> Python : {"type": "ack"}

A behavior spec:
    {"obs_size": 13,
     "action":  {"type": "discrete",   "branches": [3, 2]}  # or
                {"type": "continuous", "size": 4, "low": -1.0, "high": 1.0},
     "obs_low": -1.0, 
     "obs_high": 1.0}

An agent record (one per controllable agent, per step):
    {"agent_id": "1001",
    "behavior": "topdown",
     "obs": [...],
     "reward": 0.1,
     "done": false}

Under auto_reset (GMVecEnv), a done record carries the fresh next-episode obs in "obs"
and the final obs of the ended episode in "terminal_obs":
    {..., "done": true, "obs": [<fresh>], "terminal_obs": [<final>]}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PROTOCOL_VERSION = 1

# --------------------------------------------- Message type tags ---
MSG_HANDSHAKE = "handshake"
MSG_SPEC      = "spec"
MSG_RESET     = "reset"
MSG_STEP      = "step"
MSG_ACTION    = "action"
MSG_CLOSE     = "close"
MSG_PAUSE     = "pause"
MSG_RESUME    = "resume"
MSG_ACK       = "ack"       # engine's reply to pause/resume (keeps strict request/response)

ACTION_DISCRETE   = "discrete"
ACTION_CONTINUOUS = "continuous"


class ProtocolError(RuntimeError):
    """Raised when a packet is malformed or the protocol version disagrees."""


# --------------------------------------------- Specs ---

@dataclass(frozen=True)
class ActionSpec:
    """Describes a behavior's action space, engine-declared."""
    type: str                                           # ACTION_DISCRETE | ACTION_CONTINUOUS
    branches: list[int] = field(default_factory=list)   # discrete only
    size: int = 0                                       # continuous only
    low: float = -1.0                                   # continuous only
    high: float = 1.0                                   # continuous only

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ActionSpec":
        t = d.get("type")
        if t == ACTION_DISCRETE:
            branches = [int(b) for b in d["branches"]]
            if not branches:
                raise ProtocolError("discrete action spec has empty 'branches'")
            return cls(type=ACTION_DISCRETE, branches=branches)
        if t == ACTION_CONTINUOUS:
            return cls(
                type=ACTION_CONTINUOUS,
                size=int(d["size"]),
                low=float(d.get("low", -1.0)),
                high=float(d.get("high", 1.0)),
            )
        raise ProtocolError(f"unknown action spec type: {t!r}")


@dataclass(frozen=True)
class BehaviorSpec:
    """Full description of a trainable behavior, as declared by the engine."""
    name: str
    obs_size: int
    action: ActionSpec
    obs_low: float = -1.0
    obs_high: float = 1.0

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any]) -> "BehaviorSpec":
        return cls(
            name=name,
            obs_size=int(d["obs_size"]),
            action=ActionSpec.from_dict(d["action"]),
            obs_low=float(d.get("obs_low", -1.0)),
            obs_high=float(d.get("obs_high", 1.0)),
        )


@dataclass
class AgentRecord:
    """A single agent's observation/reward/done for one step.

    Under auto-reset (GMVecEnv), when "done" is true the engine has already reset the
    agent: "obs" is the fresh next-episode observation and "terminal_obs" carries the
    final observation of the episode that just ended (for "info['terminal_observation']").
    """
    agent_id: str
    behavior: str
    obs: list[float]
    reward: float
    done: bool
    terminal_obs: list[float] | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AgentRecord":
        term = d.get("terminal_obs")
        return cls(
            agent_id=str(d["agent_id"]),
            behavior=str(d["behavior"]),
            obs=[float(x) for x in d["obs"]],
            reward=float(d.get("reward", 0.0)),
            done=bool(d.get("done", False)),
            terminal_obs=[float(x) for x in term] if term is not None else None,
        )


# --------------------------------------------- Outgoing message builders (Python -> GameMaker) ---

def build_handshake(auto_reset: bool = False) -> dict[str, Any]:
    """Handshake packet. "auto_reset" selects the per-agent reset model (GMVecEnv);
    the default False keeps the global-reset model used by GMEnv/self-play."""
    return {"type": MSG_HANDSHAKE, "protocol": PROTOCOL_VERSION, "auto_reset": auto_reset}


def build_reset() -> dict[str, Any]:
    return {"type": MSG_RESET}


def build_action(actions: dict[str, list]) -> dict[str, Any]:
    # keys must be strings for JSON, values are plain number lists
    return {"type": MSG_ACTION,
            "actions": {str(k): list(v) for k, v in actions.items()}}


def build_close() -> dict[str, Any]:
    return {"type": MSG_CLOSE}


def build_pause() -> dict[str, Any]:
    """One-way: tell the engine to freeze physics during a weight update. No reply."""
    return {"type": MSG_PAUSE}


def build_resume() -> dict[str, Any]:
    """One-way: tell the engine to unfreeze physics. No reply."""
    return {"type": MSG_RESUME}


# --------------------------------------------- Incoming message parsers (GameMaker -> Python) ---

def parse_spec(packet: dict[str, Any]) -> dict[str, BehaviorSpec]:
    """Parse a handshake 'spec' packet into {behavior_name: BehaviorSpec}."""
    if packet.get("type") != MSG_SPEC:
        raise ProtocolError(f"expected '{MSG_SPEC}' packet, got {packet.get('type')!r}")
    version = int(packet.get("protocol", -1))
    if version != PROTOCOL_VERSION:
        raise ProtocolError(
            f"protocol version mismatch: engine={version} python={PROTOCOL_VERSION}"
        )
    behaviors = packet.get("behaviors")
    if not behaviors:
        raise ProtocolError("spec packet declares no behaviors — are agents registered?")
    return {name: BehaviorSpec.from_dict(name, spec) for name, spec in behaviors.items()}


def parse_step(packet: dict[str, Any]) -> list[AgentRecord]:
    """Parse a 'step' packet into a list of AgentRecord (one per agent)."""
    if packet.get("type") != MSG_STEP:
        raise ProtocolError(f"expected '{MSG_STEP}' packet, got {packet.get('type')!r}")
    agents = packet.get("agents")
    if agents is None:
        raise ProtocolError("step packet missing 'agents'")
    return [AgentRecord.from_dict(a) for a in agents]
