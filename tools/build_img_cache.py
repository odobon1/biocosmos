"""
python -m tools.build_img_cache

Pack each dataset's individual image files into one indexable blob per dataset, so training opens ONE file
instead of one small file per sample. These datasets are 100k+ tiny images, and the loader does one
`Image.open()` per sample per epoch (utils/data.py `__getitem__`); collapsing them into a single mmap-indexed
pack removes that per-sample filesystem overhead. This is a win on any filesystem (fewer inodes, less metadata
traffic, better read locality, and one big file is trivially staged to node-local disk), and it is *critical* on
shared parallel filesystems where per-file metadata ops dominate -- see CLUSTER NOTES for the HiPerGator case.

CLUSTER-AGNOSTIC BY DESIGN: image roots come from utils.utils.paths (which switch on the CLUSTER flag in
utils/utils.py), IMG_CACHE_DIR is derived relative to the project root, and the only cluster-specific step --
Lustre OST striping of the cache dir -- auto-skips when `lfs` is absent. So this runs as-is on PACE or anywhere.

For each dataset in DATASETS this writes, under <project_root>/../../img_cache/<dataset>/:
  - pack.bin    concatenated raw ENCODED image bytes (unchanged jpg/png bytes), in sorted-rfpath order
  - index.pkl   pickle dict {rfpath (str): (offset (int), length (int))}
  - meta.json   provenance (source root, counts, bytes, stripe, format); written LAST as a completion marker

The pack is keyed by rfpath -- the path relative to paths["imgs"][dataset], the SAME key utils/data.py already
uses (index_data[idx]["rfpath"]) -- and holds the FULL image corpus under that root, so it is split-agnostic
(one pack serves every split/setting/seed). Random access by rfpath is preserved, so the existing map-style
global shuffle (EpochEncodingDistributedSampler + torch.randperm) keeps working unchanged once reads are
pointed at the pack. Source images are never modified. The FIRST build of a dataset is slow -- it reads every
small source file once (the per-sample cost we are amortizing away) -- and is metadata/IO-bound.

Reading a sample later (for the eventual utils/data.py integration -- NOT done in this script):
    import io, mmap, pickle
    from PIL import Image
    index = pickle.load(open(out_dir / "index.pkl", "rb"))
    mm = mmap.mmap(open(out_dir / "pack.bin", "rb").fileno(), 0, access=mmap.ACCESS_READ)
    off, ln = index[rfpath]
    img = Image.open(io.BytesIO(mm[off:off + ln])).convert("RGB")

--------------------------------------------------------------------------------------------------------------
CLUSTER NOTES (HiPerGator today; what to check when re-running on PACE or elsewhere)
--------------------------------------------------------------------------------------------------------------
Why packing matters most on HiPerGator: the source images are stored Data-on-MDT (a small file's bytes live on
the metadata server, not the data servers) on a SINGLE MDT, so every per-sample read is a metadata-server RPC
and 100k+/epoch all hit one server -- the dominant cause of the variable per-epoch wall-clock. One packed file
collapses that to a handful of RPCs. As an added Lustre-only optimization, ensure_striped_dir() runs
`lfs setstripe -c STRIPE_COUNT -S STRIPE_SIZE` on the cache dir so the big pack spreads across OSTs instead of
landing on one (the default layout there is itself DoM + single-OST); verified on HPG that files inherit
stripe_count=8, pattern raid0.

Re-running on PACE (or any new cluster): it runs unchanged (striping no-ops when `lfs` is missing), but before
relying on the striping, CHECK:
  1. FS type of the cache + source dirs:  `df -T <dir>`  (+ `lfs df` / `lfs getstripe <a_source_image>` if Lustre).
       - NOT Lustre (e.g. PACE /storage is often GPFS): `lfs` is absent, striping auto-skips, and there is no
         DoM/OST knob -- packing STILL helps (many files -> one file); just ignore the striping paragraph.
       - IS Lustre: keep the striping; `lfs getstripe` a source image to see if it is also DoM (same motivation),
         and set STRIPE_COUNT to that FS's OST count if it differs.
  2. Cache location: IMG_CACHE_DIR defaults to <project_root>/../../img_cache. Confirm that lands somewhere
     sane (on PACE you may prefer shared project storage, e.g. under /storage/ice-shared/cs8903onl).
  3. DATASETS: set to the datasets defined in that cluster's paths["imgs"] (the PACE branch has bryo/cub/lepid,
     no nymph).
"""

import io
import json
import mmap
import os
import pickle
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from utils.utils import paths


# ----- config ---------------------------------------------------------------------------------------------
DATASETS = ["bryo", "cub", "lepid", "nymph"]  # datasets to pack; any of: bryo, cub, lepid, nymph
OVERWRITE = False  # False -> skip a dataset already fully packed (meta.json present)
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}  # file extensions treated as packable images
STRIPE_COUNT = 8  # Lustre OST stripe count for the cache dir (ignored on non-Lustre FS, e.g. PACE)
STRIPE_SIZE = "1M"  # Lustre stripe size (ignored on non-Lustre FS)
PROGRESS_EVERY = 5000  # files between progress prints
# ----------------------------------------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]     # <root>/tools/this.py -> <root>
IMG_CACHE_DIR = PROJECT_ROOT.parent.parent / "img_cache"


def ensure_striped_dir(d: Path) -> None:
    """Create dir d and set a wide OST stripe as its default so files written under it inherit it (Lustre only)."""
    d.mkdir(parents=True, exist_ok=True)
    if shutil.which("lfs") is None:
        print(f"  [stripe] 'lfs' not found -- skipping stripe setup (non-Lustre FS?): {d}")
        return
    res = subprocess.run(["lfs", "setstripe", "-c", str(STRIPE_COUNT), "-S", STRIPE_SIZE, str(d)], check=False)
    if res.returncode == 0:
        print(f"  [stripe] default layout set on {d}  (-c {STRIPE_COUNT} -S {STRIPE_SIZE})")
    else:
        print(f"  [stripe] WARNING lfs setstripe failed (exit {res.returncode}) on {d} -- continuing unstriped")


def list_images(img_root: Path):
    """Walk img_root; return (sorted list of rfpath strings, {skipped_ext: count}). Deterministic order.

    Packs the FULL corpus under img_root (a superset of any single split), so one pack serves every split. Each
    rfpath key is str(path relative to img_root), "/"-separated -- matching how utils/data.py builds index_data
    rfpaths (verified for bryo/cub/lepid/nymph). os.walk uses followlinks=False (default) on purpose; a symlinked
    class dir would be omitted -> a training-time KeyError, which is the desired fail-loud.
    """
    rfpaths, skipped = [], {}
    for dirpath, dirnames, filenames in os.walk(img_root):
        dirnames.sort()
        for fn in sorted(filenames):
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXTS:
                rfpaths.append(str(Path(dirpath, fn).relative_to(img_root)))
            else:
                skipped[ext] = skipped.get(ext, 0) + 1
    return rfpaths, skipped


def pack_dataset(dataset: str) -> None:
    if dataset not in paths["imgs"]:
        raise KeyError(f"dataset '{dataset}' not in paths['imgs'] (have: {sorted(paths['imgs'])})")
    img_root = Path(paths["imgs"][dataset])
    if not img_root.is_dir():
        raise FileNotFoundError(f"image root for '{dataset}' does not exist: {img_root}")

    out_dir = IMG_CACHE_DIR / dataset
    pack_path, index_path, meta_path = out_dir / "pack.bin", out_dir / "index.pkl", out_dir / "meta.json"

    if meta_path.exists() and not OVERWRITE:
        print(f"[{dataset}] already packed (meta.json present) -- skipping (set OVERWRITE=True to rebuild)")
        return

    print(f"[{dataset}] scanning {img_root} ...")
    rfpaths, skipped = list_images(img_root)
    if not rfpaths:
        raise RuntimeError(f"no images {sorted(IMG_EXTS)} found under {img_root}")
    if skipped:
        print(f"[{dataset}] WARNING skipped non-image extensions (add to IMG_EXTS if any of these are images): {skipped}")
    print(f"[{dataset}] packing {len(rfpaths):,} images -> {out_dir}")

    ensure_striped_dir(out_dir)
    meta_path.unlink(missing_ok=True)                  # drop any stale completion marker before (re)building
    index, offset = {}, 0
    t0 = time.time()
    pack_tmp = pack_path.with_name("pack.bin.tmp")
    with open(pack_tmp, "wb", buffering=8 * 1024 * 1024) as out:
        for i, rf in enumerate(rfpaths, 1):
            data = (img_root / rf).read_bytes()
            if not data:
                raise RuntimeError(f"empty/unreadable source image (0 bytes): {img_root / rf}")
            out.write(data)
            index[rf] = (offset, len(data))
            offset += len(data)
            if i % PROGRESS_EVERY == 0 or i == len(rfpaths):
                dt = time.time() - t0 or 1e-9
                print(f"[{dataset}]   {i:,}/{len(rfpaths):,}  {offset / 1e9:.2f} GB  "
                      f"{i / dt:.0f} files/s  {offset / 1e6 / dt:.0f} MB/s")
    os.replace(pack_tmp, pack_path)                    # pack is only "seen" complete once atomically renamed

    index_tmp = index_path.with_name("index.pkl.tmp")
    with open(index_tmp, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(index_tmp, index_path)                  # commit index atomically too, before the meta marker

    meta = {
        "dataset": dataset,
        "source_img_root": str(img_root),
        "n_images": len(rfpaths),
        "total_bytes": offset,
        "img_exts": sorted(IMG_EXTS),
        "skipped_ext_counts": skipped,
        "pack_format": "concatenated raw encoded image bytes, sorted-rfpath order",
        "index_format": "pickle dict {rfpath: [offset, length]}",
        "stripe": {"count": STRIPE_COUNT, "size": STRIPE_SIZE},
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(meta_path, "w") as f:                    # written LAST -> presence == a complete, valid pack
        json.dump(meta, f, indent=2)

    print(f"[{dataset}] DONE  {len(rfpaths):,} images  {offset / 1e9:.2f} GB  ({time.time() - t0:.0f}s)")
    verify_pack(dataset)


def verify_pack(dataset: str, n_samples: int = 6) -> None:
    """Sanity check: mmap the pack and decode a spread of samples looked up through the index."""
    out_dir = IMG_CACHE_DIR / dataset
    with open(out_dir / "index.pkl", "rb") as f:
        index = pickle.load(f)
    keys = list(index)
    idxs = sorted({0, len(keys) - 1, *[(len(keys) * k) // (n_samples + 1) for k in range(1, n_samples + 1)]})
    with open(out_dir / "pack.bin", "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for j in idxs:
                off, ln = index[keys[j]]
                Image.open(io.BytesIO(mm[off:off + ln])).load()   # force full decode
        finally:
            mm.close()
    print(f"[{dataset}] verified {len(idxs)} samples decode OK")


def main() -> None:
    # cub's source root is cwd-relative (paths["root"] = os.getcwd()); other datasets are absolute. Run from the
    # project root (python -m tools.build_img_cache) so every dataset's img_root resolves.
    assert Path.cwd().resolve() == PROJECT_ROOT, f"run from the project root: {PROJECT_ROOT}"
    print(f"img cache dir: {IMG_CACHE_DIR}")
    ensure_striped_dir(IMG_CACHE_DIR)                 # stripe the root so per-dataset subdirs/files inherit it
    for dataset in DATASETS:
        pack_dataset(dataset)
    print("all done.")


if __name__ == "__main__":
    main()
