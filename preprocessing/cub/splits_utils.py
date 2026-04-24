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
SPLITS_URL        = "http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip"
 
# Known location of the split/mapping mat file inside the archive
# (xlsa17/data/CUB/att_splits.mat  – CUB used as example)
# datasets other than CUB not necessary
DATASETS   = ["AWA1", "AWA2", "CUB", "SUN", "APY"]
 
# ── Download ──────────────────────────────────────────────────────────────────
def download_splits(url: str, dest: str) -> None:
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
def extract_splits(zip_path: str, dest_dir: str) -> None:
    marker = os.path.join(dest_dir, "xlsa17")
    if os.path.isdir(marker):
        print(f"[skip] {marker} already exists – skipping extraction.")
        return
    print(f"[extract] {zip_path} → {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    os.remove(zip_path)
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
    
