import subprocess
import zipfile
from pathlib import Path

import pytest
import yaml

from tools import zip as zip_tool


TRIAL = "artifacts/camp/_phase/screen/_datasets/cub/_arms/mp/_coords/LR-1e-5/_seeds/42"
CORE = {  # always included, whatever the toggles
    "artifacts/camp/_phase/screen/phase_metadata.json",
    "artifacts/camp/_phase/screen/_datasets/cub/_arms/mp/_coords/LR-1e-5/config.json",
    f"{TRIAL}/trial_metadata.json",
    f"{TRIAL}/data_trial.pkl",
    f"{TRIAL}/evals/evals/1/metrics/metrics.json",
    f"{TRIAL}/evals/_selected/metrics/metrics.json",
    f"{TRIAL}/learning_curves/scores.png",
    f"{TRIAL}/logs/epoch.log",
}
BULKY = {  # toggle -> the files it governs
    "weights": {f"{TRIAL}/model.pt", f"{TRIAL}/chkpts/in_progress/train_state.pt"},
    "manifold_viz": {
        f"{TRIAL}/evals/evals/1/viz/vanilla/8panel/joint.png",
        f"{TRIAL}/evals/evals/1/viz/pooled/8panel/joint.png",
        f"{TRIAL}/evals/_selected/viz/vanilla/8panel/joint.png",
        f"{TRIAL}/evals/_best/viz/pooled/8panel/joint.png",
        f"{TRIAL}/viz/vanilla/8panel/joint.gif",
        f"{TRIAL}/viz/pooled/8panel/joint.gif",
    },
    "manifold_viz_cache": {
        f"{TRIAL}/evals/evals/1/viz/cache/embs.npz",
        f"{TRIAL}/evals/evals/1/viz/cache/projections.npz",
        f"{TRIAL}/evals/evals/1/viz/cache/projections_pooled.npz",
        f"{TRIAL}/evals/evals/1/viz/cache/orient_ref.pkl",
        f"{TRIAL}/evals/_selected/viz/cache/projections.npz",
        f"{TRIAL}/evals/_best/viz/cache/embs.npz",
    },
    "batch_logs": {f"{TRIAL}/logs/batch/grad_norm.log", f"{TRIAL}/logs/batch/sim_targ.log"},
}


def _write(fpath: Path) -> None:
    fpath.parent.mkdir(parents=True, exist_ok=True)
    fpath.write_text("x")


def _set_cfg(root: Path, **toggles) -> None:
    cfg = {"weights": False, "manifold_viz": False, "manifold_viz_cache": False, "batch_logs": False,
           "untracked": ["tools/readable_tree.txt", "tools/image_aug"], **toggles}
    _write(root / "config" / "render" / "zip.yaml")
    (root / "config" / "render" / "zip.yaml").write_text(yaml.safe_dump(cfg))


def _make_campaign(root: Path, campaign: str = "camp") -> None:
    for rel in CORE | set().union(*BULKY.values()):
        _write(root / rel.replace("artifacts/camp/", f"artifacts/{campaign}/"))


def _names(fpath_zip: Path) -> set[str]:
    with zipfile.ZipFile(fpath_zip) as zf:
        return set(zf.namelist())


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway repo root: a git index with two tracked files, untracked files (one listed in zip.yaml, one
    not) and an untracked dir (listed), plus the zip.yaml tools.zip reads; paths[...] redirected there."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    _write(tmp_path / "train.py")
    _write(tmp_path / "utils" / "config.py")
    subprocess.run(["git", "add", "train.py", "utils/config.py"], cwd=tmp_path, check=True)
    _write(tmp_path / "notes.txt")  # untracked, unlisted -> out
    _write(tmp_path / "tools" / "readable_tree.txt")  # untracked, listed as a file
    _write(tmp_path / "tools" / "image_aug" / "a.png")  # untracked, listed as a dir
    _write(tmp_path / "tools" / "image_aug" / "sub" / "b.png")
    monkeypatch.setitem(zip_tool.paths, "root", tmp_path)
    monkeypatch.setitem(zip_tool.paths, "config", tmp_path / "config")
    monkeypatch.setitem(zip_tool.paths, "artifacts", tmp_path / "artifacts")
    _set_cfg(tmp_path)
    return tmp_path


def test_source_snapshot_is_tracked_plus_listed_untracked(root: Path) -> None:
    fpath_zip, n_files = zip_tool.zip_snapshot([])

    assert fpath_zip.parent == root / "temp"
    assert fpath_zip.name.startswith("biocosmos_") and fpath_zip.suffix == ".zip"
    assert n_files == 5
    assert _names(fpath_zip) == {
        "biocosmos/train.py",
        "biocosmos/utils/config.py",
        "biocosmos/tools/readable_tree.txt",
        "biocosmos/tools/image_aug/a.png",
        "biocosmos/tools/image_aug/sub/b.png",
    }


def test_campaign_core_ships_with_every_toggle_off(root: Path) -> None:
    _make_campaign(root)

    fpath_zip, _ = zip_tool.zip_snapshot(["camp"])
    names = _names(fpath_zip)

    assert fpath_zip.name.endswith("_camp.zip")
    assert {f"biocosmos/{rel}" for rel in CORE} <= names
    assert not {f"biocosmos/{rel}" for files in BULKY.values() for rel in files} & names


@pytest.mark.parametrize("toggle", sorted(BULKY))
def test_each_toggle_admits_only_its_files(root: Path, toggle: str) -> None:
    _make_campaign(root)
    _set_cfg(root, **{toggle: True})

    names = _names(zip_tool.zip_snapshot(["camp"])[0])

    for kind, files in BULKY.items():
        for rel in files:
            assert (f"biocosmos/{rel}" in names) == (kind == toggle), rel


def test_multiple_campaigns(root: Path) -> None:
    _make_campaign(root, "camp")
    _make_campaign(root, "camp2")

    fpath_zip, _ = zip_tool.zip_snapshot(["camp", "camp2"])
    names = _names(fpath_zip)

    assert fpath_zip.name.endswith("_camp_camp2.zip")
    assert "biocosmos/artifacts/camp/_phase/screen/phase_metadata.json" in names
    assert "biocosmos/artifacts/camp2/_phase/screen/phase_metadata.json" in names


def test_unknown_campaign_exits(root: Path) -> None:
    with pytest.raises(SystemExit):
        zip_tool.zip_snapshot(["nope"])


def test_missing_untracked_entry_raises(root: Path) -> None:
    _set_cfg(root, untracked=["nope.txt"])

    with pytest.raises(FileNotFoundError):
        zip_tool.zip_snapshot([])
