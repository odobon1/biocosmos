"""
Internal module used by campaign_runner to execute a single trial config.
"""

from argparse import ArgumentParser
import json

from train import run_training
from utils.config import get_config_train


def _parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--cfg-json", required=True, help="Trial config as JSON string")
    return parser


def main() -> None:
    parser = _parse_args()
    args = parser.parse_args()

    cfg_dict = json.loads(args.cfg_json)
    cfg = get_config_train(cfg_dict=cfg_dict)
    run_training(cfg)


if __name__ == "__main__":
    main()
