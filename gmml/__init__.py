"""
gmml - a reusable framework for training reinforcement-learning agents in GameMaker games 
inspired by Unity ML-Agents. Domain logic lives in the GameMaker project (gmproject), 
this package is generic and config-driven.
"""

from .connection import GameMakerConnection
from .env import GMEnv

__all__ = ["GMEnv", "GameMakerConnection", "protocol"]
__version__ = "0.1.0"

from . import protocol  # noqa: E402
