"""
Downloads xlsa17.zip, extracts it in the same directory,
and reads the attribute/class mapping .mat file inside.
"""
 
import os
import zipfile
import urllib.request
from scipy.io import loadmat
import numpy as np
 
# ── Config ────────────────────────────────────────────────────────────────────
URL        = "http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip"
DEST_DIR   = os.path.dirname(os.path.abspath(__file__))   # same dir as this script
ZIP_PATH   = os.path.join(DEST_DIR, "xlsa17.zip")
 
# Known location of the split/mapping mat file inside the archive
# (xlsa17/data/<DATASET>/att_splits.mat  – AWA2 used as example)
DATASETS   = ["AWA1", "AWA2", "CUB", "SUN", "APY"]
 
# ── Download ──────────────────────────────────────────────────────────────────
def download(url: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"[skip] {dest} already exists – skipping download.")
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
 
 
# ── Extract ───────────────────────────────────────────────────────────────────
def extract(zip_path: str, dest_dir: str) -> None:
    marker = os.path.join(dest_dir, "xlsa17")
    if os.path.isdir(marker):
        print(f"[skip] {marker} already exists – skipping extraction.")
        return
    print(f"[extract] {zip_path} → {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    print("[extract] done.")
 
 
# ── Read .mat ─────────────────────────────────────────────────────────────────
def read_mat(mat_path: str) -> dict:
    """Load a MATLAB v5/v7.3 .mat file and return a clean dict."""
    try:
        mat = loadmat(mat_path)
    except NotImplementedError:
        print("Failed to open .mat file")
    # Remove MATLAB meta-keys
    return {k: v for k, v in mat.items() if not k.startswith("__")}
  
 
# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # example use case
    download(URL, ZIP_PATH)
    extract(ZIP_PATH, DEST_DIR)
 
    mat_path = os.path.join(DEST_DIR, "xlsa17", "data", "CUB", "att_splits.mat")
    if not os.path.isfile(mat_path):
        print(f"[warn] {mat_path} not found.")
        return
    data = read_mat(mat_path)
 
 
if __name__ == "__main__":
    main()
 
