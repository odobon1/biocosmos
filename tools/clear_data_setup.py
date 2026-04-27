"""
python -m tools.clear_data_setup
"""

from utils.utils import paths
from tools.clear_preprocessing import clear_all_preprocessing, remove_path


def clear_cub_data() -> None:
    preserved_path = paths["cub_tree_raw"]
    for child in paths["data"]["cub"].iterdir():
        if child == preserved_path:
            continue
        remove_path(child)

def main() -> None:
    clear_all_preprocessing()
    clear_cub_data()


if __name__ == "__main__":
    main()