"""
Model export for in-engine inference (run a trained policy inside GameMaker, no Python).

onnx: not yet implemented but will be the preferred inference solution (native ONNX
Runtime backends for HTML5 and Windows).
"""

from __future__ import annotations

from pathlib import Path

FORMATS = ("onnx",)


def _default_out(cfg: dict, fmt: str) -> str:
    ext = {"onnx": ".onnx"}[fmt]
    run_dir = Path(cfg["output"]["run_dir"])
    return str(run_dir / (cfg["output"]["final_name"] + ext))


def export(cfg: dict, *, fmt: str, model_path: str | None = None,
           out: str | None = None) -> None:
    if fmt not in FORMATS:
        raise ValueError(f"Unknown export format {fmt!r}; choose from {FORMATS}")

    if fmt == "onnx":
        raise NotImplementedError(
            "ONNX export is a planned fast-follow (native ONNX Runtime backends for HTML5 and "
            "Windows)."
        )


def main(config: str, overrides=None, *, fmt: str = "onnx", **kw) -> None:
    from .config import load_config
    export(load_config(config, overrides), fmt=fmt, **kw)
