"""
python -m tools.clear_preprocessing all
python -m tools.clear_preprocessing splits
"""

import shutil
import sys
from pathlib import Path

from utils.utils import paths


DATASETS = ("bryo", "cub", "lepid", "nymph")


def remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
        return
    path.unlink()

def clear_dataset_metadata(dpath_dataset: Path) -> None:
    dpath_splits = dpath_dataset / "splits"
    fpath_gitkeep = dpath_splits / ".gitkeep"

    if not fpath_gitkeep.is_file():
        raise FileNotFoundError(f"Refusing to clear {dpath_dataset}: missing {fpath_gitkeep}")

    for child in dpath_dataset.iterdir():
        if child == dpath_splits:
            continue
        remove_path(child)

    for child in dpath_splits.iterdir():
        if child == fpath_gitkeep:
            continue
        remove_path(child)

def clear_dataset_splits(dpath_dataset: Path) -> None:
    dpath_splits = dpath_dataset / "splits"
    fpath_gitkeep = dpath_splits / ".gitkeep"

    if not fpath_gitkeep.is_file():
        raise FileNotFoundError(f"Refusing to clear {dpath_splits}: missing {fpath_gitkeep}")

    for child in dpath_splits.iterdir():
        if child == fpath_gitkeep:
            continue
        remove_path(child)

def clear_all_preprocessing() -> None:
    for ds_name in DATASETS:
        clear_dataset_metadata(paths["metadata"][ds_name])

def clear_all_splits() -> None:
    for ds_name in DATASETS:
        clear_dataset_splits(paths["metadata"][ds_name])

def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ("all", "splits"):
        print("Usage: python -m tools.clear_preprocessing all|splits")
        sys.exit(1)

    if sys.argv[1] == "all":
        clear_all_preprocessing()
    else:
        clear_all_splits()


if __name__ == "__main__":
    main()