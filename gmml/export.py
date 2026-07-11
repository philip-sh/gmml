"""
Model export for in-engine inference (run a trained policy inside GameMaker, no Python).

gmpolicy: a self-contained binary format (weights + architecture + action spec) read by
the pure-GML runtime in scr_gmml_inference. Deterministic acting only (continuous returns
the Gaussian mean, discrete returns per-branch argmax), which matches 'gmml infer'.

onnx: not yet implemented but will be the preferred inference solution (native ONNX
Runtime backends for HTML5 and Windows).
"""

from __future__ import annotations

import struct
from pathlib import Path

FORMATS = ("gmpolicy", "onnx")

_EXT = {"gmpolicy": ".gmpolicy", "onnx": ".onnx"}

# .gmpolicy binary format (see scr_gmml_inference in the gmproject for the reader)
GMPOLICY_MAGIC = b"GMPOLICY"
GMPOLICY_VERSION = 1
ACTION_DISCRETE = 0
ACTION_CONTINUOUS = 1
ACT_TANH = 0
ACT_RELU = 1


def _default_out(cfg: dict, fmt: str) -> str:
    run_dir = Path(cfg["output"]["run_dir"])
    return str(run_dir / (cfg["output"]["final_name"] + _EXT[fmt]))


def _default_model_path(cfg: dict) -> str:
    return str(Path(cfg["output"]["run_dir"]) / cfg["output"]["final_name"])


def _export_gmpolicy(cfg: dict, model_path: str | None, out: str | None) -> None:
    import numpy as np
    import torch.nn as nn
    from gymnasium import spaces

    from .train import _algo_cls

    algo_cls = _algo_cls(cfg["algo"])
    model = algo_cls.load(model_path or _default_model_path(cfg), device="cpu")

    # The acting path is obs -> mlp_extractor.policy_net -> action_net. Walk policy_net in
    # order collecting Linear layers, the interleaved modules tell us the activation.
    policy = model.policy
    linears: list[nn.Linear] = []
    activation_id = ACT_TANH   # SB3 ActorCriticPolicy default
    for module in policy.mlp_extractor.policy_net:
        if isinstance(module, nn.Linear):
            linears.append(module)
        elif isinstance(module, nn.ReLU):
            activation_id = ACT_RELU
        elif isinstance(module, nn.Tanh):
            activation_id = ACT_TANH
    linears.append(policy.action_net)   # output head, no activation after it

    obs_space = model.observation_space
    if not isinstance(obs_space, spaces.Box) or len(obs_space.shape) != 1:
        raise ValueError(
            f"gmpolicy export supports only 1-D Box observations, got {obs_space}"
        )
    obs_size = int(obs_space.shape[0])

    action_space = model.action_space
    buf = bytearray()
    buf += GMPOLICY_MAGIC
    buf += struct.pack("<II", GMPOLICY_VERSION, obs_size)

    if isinstance(action_space, spaces.MultiDiscrete):
        branches = [int(n) for n in action_space.nvec]
        buf += struct.pack("<BI", ACTION_DISCRETE, len(branches))
        buf += struct.pack(f"<{len(branches)}I", *branches)
    elif isinstance(action_space, spaces.Discrete):
        buf += struct.pack("<BII", ACTION_DISCRETE, 1, int(action_space.n))
    elif isinstance(action_space, spaces.Box):
        low = float(np.asarray(action_space.low).flat[0])
        high = float(np.asarray(action_space.high).flat[0])
        buf += struct.pack("<BIff", ACTION_CONTINUOUS, int(action_space.shape[0]), low, high)
    else:
        raise ValueError(f"gmpolicy export: unsupported action space {action_space}")

    buf += struct.pack("<BI", activation_id, len(linears))
    for layer in linears:
        w = layer.weight.detach().cpu().numpy().astype("<f4")   # (out, in), row-major
        b = layer.bias.detach().cpu().numpy().astype("<f4")     # (out,)
        out_features, in_features = w.shape
        buf += struct.pack("<II", int(in_features), int(out_features))
        buf += w.tobytes(order="C")
        buf += b.tobytes(order="C")

    out_path = Path(out or _default_out(cfg, "gmpolicy"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(buf)
    print(f"Exported gmpolicy to {out_path} ({len(buf)} bytes, {len(linears)} layers)")


def export(cfg: dict, *, fmt: str, model_path: str | None = None,
           out: str | None = None) -> None:
    if fmt not in FORMATS:
        raise ValueError(f"Unknown export format {fmt!r}; choose from {FORMATS}")

    if fmt == "gmpolicy":
        _export_gmpolicy(cfg, model_path, out)
    elif fmt == "onnx":
        raise NotImplementedError("ONNX export format is not yet implemented.")


def main(config: str, overrides=None, *, fmt: str = "gmpolicy", **kw) -> None:
    from .config import load_config
    export(load_config(config, overrides), fmt=fmt, **kw)
