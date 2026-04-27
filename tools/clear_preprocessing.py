"""
python -m tools.clear_preprocessing
"""

import shutil
from pathlib import Path

from utils.utils import paths


DATASET_NAMES = ("bryo", "cub", "lepid", "nymph")


def remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
        return
    path.unlink()

def clear_dataset_metadata(dataset_dir: Path) -> None:
    splits_dir = dataset_dir / "splits"
    gitkeep_path = splits_dir / ".gitkeep"

    if not gitkeep_path.is_file():
        raise FileNotFoundError(f"Refusing to clear {dataset_dir}: missing {gitkeep_path}")

    for child in dataset_dir.iterdir():
        if child == splits_dir:
            continue
        remove_path(child)

    for child in splits_dir.iterdir():
        if child == gitkeep_path:
            continue
        remove_path(child)

def clear_all_preprocessing() -> None:
    for ds_name in DATASET_NAMES:
        clear_dataset_metadata(paths["metadata"][ds_name])

def main() -> None:
    clear_all_preprocessing()


if __name__ == "__main__":
    main()