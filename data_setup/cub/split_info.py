"""
python -m data_setup.cub.split_info
"""

import os
import zipfile
import urllib.request

from utils.utils import paths


SPLITS_URL = "http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip"


def download_splits(url: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"[skip] {dest} already exists - skipping download.")
        return
    print(f"[download] {url}")
    print(f"        → {dest}")
 
    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            bar = "█" * int(pct // 2) + "░" * (50 - int(pct // 2))
            print(f"\r  [{bar}] {pct:5.1f}%", end="", flush=True)
 
    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print()  # newline after progress bar
    print("[download] complete.")

def extract_splits(zip_path: str, dest_dir: str) -> None:
    marker = os.path.join(dest_dir, "xlsa17")
    if os.path.isdir(marker):
        print(f"[skip] {marker} already exists - skipping extraction.")
        return
    print(f"[extract] {zip_path} → {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    os.remove(zip_path)
    print("[extract] done.")

def main() -> None:
    print("Downloading Attribute Splits...")
    download_splits(SPLITS_URL, paths["data"]["cub"] / "xlsa17.zip")

    print("Extracting Attribute Splits...")
    extract_splits(paths["data"]["cub"] / "xlsa17.zip", paths["data"]["cub"])


if __name__ == "__main__":
    main()