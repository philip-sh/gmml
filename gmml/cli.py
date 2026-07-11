"""
gmml command-line interface.

    gmml train  --config configs/yourconfig.yaml [--resume | --resume-from CKPT]
    gmml infer  --config configs/yourconfig.yaml [--model PATH]
    gmml export --config configs/yourconfig.yaml [--format onnx]

Any config field can be overridden with '--set key.path=value'.
"""

from __future__ import annotations

import argparse
import sys

from .export import FORMATS


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", required=True, help="path to the behavior YAML")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE", help="override a config field (repeatable)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gmml", description="GameMaker ML-Agents framework")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train a behavior from a YAML config")
    _add_common(t)
    t.add_argument("--resume", action="store_true",
                   help="resume from the latest checkpoint in the run dir")
    t.add_argument("--resume-from", metavar="CKPT", help="resume from a specific checkpoint")
    t.add_argument("--reset-timesteps", action="store_true",
                   help="restart the timestep counter instead of continuing it")
    t.add_argument("--reset-pool", action="store_true",
                   help="clear the opponent pool before training (self-play starts fresh from v0)")

    i = sub.add_parser("infer", help="run a trained policy against GameMaker")
    _add_common(i)
    i.add_argument("--model", help="policy path (default: <run_dir>/<final_name>)")
    i.add_argument("--episodes", type=int, help="stop after N episodes (default: run forever)")

    e = sub.add_parser("export", help="export a policy for in-engine inference")
    _add_common(e)
    e.add_argument("--format", choices=FORMATS, default="onnx", help="export format")
    e.add_argument("--model", help="policy path (default: <run_dir>/<final_name>)")
    e.add_argument("--out", help="output file path")

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "train":
        from .train import main as run
        run(args.config, args.overrides,
            resume=args.resume, resume_from=args.resume_from,
            reset_timesteps=args.reset_timesteps, reset_pool=args.reset_pool)
    elif args.command == "infer":
        from .infer import main as run
        run(args.config, args.overrides, model_path=args.model, episodes=args.episodes)
    elif args.command == "export":
        from .export import main as run
        run(args.config, args.overrides, fmt=args.format, model_path=args.model, out=args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
