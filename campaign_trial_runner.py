"""
Internal module used by campaign_runner to execute a single trial config.

Usage:
    torchrun --standalone --nproc-per-node=auto -m campaign_trial_runner --cfg-path <path>
"""

from argparse import ArgumentParser
import json
from pathlib import Path

from train import run_training
from utils.config import get_config_train


def _parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--cfg-path", required=True, help="Path to serialized trial config JSON")
    return parser


def main() -> None:
    parser = _parse_args()
    args = parser.parse_args()

    cfg_path = Path(args.cfg_path)
    with open(cfg_path) as f:
        cfg_dict = json.load(f)

    cfg = get_config_train(cfg_dict=cfg_dict)
    run_training(cfg)


if __name__ == "__main__":
    main()
