"""
Internal module used by campaign_runner to execute a single trial config.
"""

from argparse import ArgumentParser
import json
from pathlib import Path
import torch.distributed as dist

from train import run_training
from utils.config import get_config_train


def _parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--fpath-cfg", required=True, help="Path to serialized trial config JSON")
    return parser


def main() -> None:
    parser = _parse_args()
    args = parser.parse_args()

    fpath_cfg = Path(args.fpath_cfg)
    with open(fpath_cfg) as f:
        cfg_dict = json.load(f)

    if dist.is_initialized():
        dist.barrier()
    if not dist.is_initialized() or dist.get_rank() == 0:
        fpath_cfg.unlink(missing_ok=True)

    cfg = get_config_train(cfg_dict=cfg_dict)
    run_training(cfg)


if __name__ == "__main__":
    main()
