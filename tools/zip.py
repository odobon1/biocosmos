"""
python -m tools.zip [<campaign> ...]

Bundle a snapshot of the source tree -- every git-tracked file (per the index, so staged adds count and a `git rm`
drops out), plus the untracked files config/render/zip.yaml lists -- and, optionally, whole campaigns
(artifacts/<campaign>/, every phase incl. test/) into one zip under temp/:
temp/biocosmos_<YYYYMMDD-HHMMSS>[_<campaign>...].zip, laid out as a checkout (everything under a top-level
biocosmos/ dir, campaigns at biocosmos/artifacts/<campaign>/). Which of a campaign's bulky contents ship is
governed by config/render/zip.yaml's toggles (weights, manifold_viz, manifold_viz_cache, batch_logs); the rest of a campaign --
metrics, metadata, stats tables and plots, learning curves, the epoch/eval/init logs -- is always included.
"""

from pathlib import Path
import subprocess
import sys
import time
import zipfile

from utils.config import load_zip_config_dict
from utils.utils import paths


MANIF_VIZ_CACHE = ("embs.npz", "projections.npz", "projections_pooled.npz", "orient_ref.pkl")


def _bulky_toggle(rel):
    """The zip.yaml toggle governing a campaign file at campaign-relative path `rel`, or None for a file that is
    always included. Bulky contents all live inside trial dirs (.../_seeds/<seed>/), so the match is on the
    trial-relative path."""
    parts = rel.parts
    if "_seeds" not in parts:
        return None
    sub = parts[parts.index("_seeds") + 2:]  # trial-relative
    if sub[0] in ("model.pt", "chkpts"):  # trainval product weights; in-progress resume state (model + optimizer)
        return "weights"
    if any(p in ("viz", "viz_pooled") for p in sub[:-1]):  # per-eval plots (evals/<eval>/viz*/) + evolving GIFs (<trial>/viz*/)
        return "manifold_viz"
    if sub[0] == "evals" and sub[-1] in MANIF_VIZ_CACHE:
        return "manifold_viz_cache"
    if sub[:2] == ("logs", "batch"):
        return "batch_logs"
    return None

def source_files(cfg_zip):
    """Root-relative paths of the source snapshot: every git-tracked file, plus zip.yaml's `untracked` entries --
    each a file, or a dir taken whole."""
    out = subprocess.run(["git", "ls-files", "-z"], cwd=paths["root"], check=True, capture_output=True).stdout
    files = [Path(p) for p in out.decode().split("\0") if p]
    for entry in cfg_zip["untracked"]:
        p = paths["root"] / entry
        if not p.exists():
            raise FileNotFoundError(f"zip.yaml untracked entry not found: {entry}")
        files += [q.relative_to(paths["root"]) for q in ([p] if p.is_file() else sorted(p.rglob("*"))) if q.is_file()]
    return files

def campaign_files(campaign, cfg_zip):
    """Root-relative paths of a campaign's contents, minus the bulky categories zip.yaml switches off."""
    dpath = paths["artifacts"] / campaign
    if not dpath.is_dir():
        sys.exit(f"no such campaign: {dpath}")
    files = []
    for f in sorted(dpath.rglob("*")):
        if f.is_file():
            toggle = _bulky_toggle(f.relative_to(dpath))
            if toggle is None or cfg_zip[toggle]:
                files.append(f.relative_to(paths["root"]))
    return files

def zip_snapshot(campaigns):
    cfg_zip = load_zip_config_dict()
    files = source_files(cfg_zip)
    for campaign in campaigns:
        files += campaign_files(campaign, cfg_zip)
    dpath_out = paths["root"] / "temp"
    dpath_out.mkdir(exist_ok=True)
    fpath_zip = dpath_out / ("_".join(["biocosmos", time.strftime("%Y%m%d-%H%M%S"), *campaigns]) + ".zip")
    with zipfile.ZipFile(fpath_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in files:
            zf.write(paths["root"] / rel, f"biocosmos/{rel.as_posix()}")
    return fpath_zip, len(files)

def main():
    fpath_zip, n_files = zip_snapshot([c.strip("/") for c in sys.argv[1:]])
    print(f"{fpath_zip} -- {n_files} files, {fpath_zip.stat().st_size / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
