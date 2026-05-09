from types import SimpleNamespace

import torch  # type: ignore[import]
import utils.train as train_utils

from utils.train import ArtifactManager, TrainImageDumper
from utils.utils import load_json


def make_cfg(view_imgs: int):
    return SimpleNamespace(
        campaign_name="campaign",
        setting_name="setting",
        dataset="lepid",
        seed=42,
        dev={
            "allow_overwrite_trial": True,
            "view_imgs": view_imgs,
        },
    )


def test_artifact_manager_creates_train_imgs_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setitem(train_utils.paths, "artifacts", tmp_path / "artifacts")

    cfg = make_cfg(view_imgs=3)
    ArtifactManager.set_paths(cfg)
    ArtifactManager.create_trial_dirs()

    assert ArtifactManager.dpath_train_imgs is not None
    assert ArtifactManager.dpath_train_imgs.exists()
    assert ArtifactManager.fpath_train_imgs_manifest is not None


def test_artifact_manager_skips_train_imgs_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setitem(train_utils.paths, "artifacts", tmp_path / "artifacts")

    cfg = make_cfg(view_imgs=0)
    ArtifactManager.set_paths(cfg)
    ArtifactManager.create_trial_dirs()

    assert ArtifactManager.dpath_train_imgs is None
    assert ArtifactManager.fpath_train_imgs_manifest is None
    assert not (ArtifactManager.dpath_trial / "train_imgs").exists()


class _Normalize:

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std


class _Compose:

    def __init__(self, transforms):
        self.transforms = transforms


def test_train_image_dumper_saves_exact_target_and_manifest(tmp_path, monkeypatch):
    monkeypatch.setitem(train_utils.paths, "artifacts", tmp_path / "artifacts")

    cfg = make_cfg(view_imgs=3)
    ArtifactManager.set_paths(cfg)
    ArtifactManager.create_trial_dirs()

    img_pp_train = _Compose([_Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    dumper = TrainImageDumper(cfg=cfg, gpu_rank=0, gpu_world_size=1, img_pp_train=img_pp_train)

    imgs_sb = torch.zeros(5, 3, 4, 4)
    targ_data_sb = (
        {"cid": "c-1", "class_enc": 11},
        {"cid": "c-2", "class_enc": 12},
        {"cid": "c-3", "class_enc": 13},
        {"cid": "c-4", "class_enc": 14},
        {"cid": "c-5", "class_enc": 15},
    )

    dumper.dump(imgs_sb, targ_data_sb)

    saved_imgs = sorted(ArtifactManager.dpath_train_imgs.glob("*.png"))
    assert len(saved_imgs) == 3
    assert saved_imgs[0].name.startswith("000000_cid-c-1_class-11")
    assert saved_imgs[-1].name.startswith("000002_cid-c-3_class-13")

    manifest = load_json(ArtifactManager.fpath_train_imgs_manifest)
    assert manifest["view_imgs_target"] == 3
    assert manifest["saved"] == 3


def test_train_image_dumper_is_rank_zero_only(tmp_path, monkeypatch):
    monkeypatch.setitem(train_utils.paths, "artifacts", tmp_path / "artifacts")

    cfg = make_cfg(view_imgs=2)
    ArtifactManager.set_paths(cfg)
    ArtifactManager.create_trial_dirs()

    img_pp_train = _Compose([_Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    dumper = TrainImageDumper(cfg=cfg, gpu_rank=1, gpu_world_size=2, img_pp_train=img_pp_train)

    imgs_sb = torch.zeros(2, 3, 4, 4)
    targ_data_sb = ({"cid": "c-1", "class_enc": 11}, {"cid": "c-2", "class_enc": 12})

    dumper.dump(imgs_sb, targ_data_sb)

    assert len(list(ArtifactManager.dpath_train_imgs.glob("*.png"))) == 0
    assert not ArtifactManager.fpath_train_imgs_manifest.exists()