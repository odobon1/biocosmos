import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from utils.train import ArtifactManager, TrialData, format_mem, merge_mem
from utils.utils import save_pickle, load_pickle


def _full_loss_cfg(crit="bce", targ="sw"):
    return {
        "crit": crit,
        "sim": "cos",
        "targ": targ,
        "wting": {
            "cls_imb": {
                "type": "inv_freq",
                "inv_freq": {"gamma": 0.5},
                "class_bal": {"beta": 0.9999},
                "freq_type_2d": "naive",
                "wt_mean_type": "per_class",
            },
            "focal": {"gamma": 2.0, "comp_type": 1},
            "bce": {"dsmr": True, "agg": "prod", "norm": {"cls_imb": True, "agg": False}},
        },
        "logits": {
            "scale": {"init": None, "freeze": False, "clamp": False},
            "bias": {"init": None, "freeze": False},
        },
    }


@dataclass
class _FakeSettingCfg:
    campaign: str = "c"
    setting: str = "iw"
    seed: int = 42
    idx_seed: int = 0
    dataset: str = "cub"
    split: str = "D10"
    dev: dict = field(default_factory=dict)
    arch: dict = field(default_factory=lambda: {
        "model_type": "siglip_vitb16", "clip": {"non_causal": False}, "siglip": {"vis_proj_head": None},
    })
    dropout: dict = field(default_factory=lambda: {
        "patch_dropout": 0.0, "siglip": {"proj_head": 0.0, "stoch_depth": None},
    })
    loss: dict = field(default_factory=_full_loss_cfg)
    loss2: dict = field(default_factory=lambda: {"mix": 0.0, "mix_unit_scale": False, **_full_loss_cfg(targ="phylo")})


def test_save_metadata_setting_splits_config_and_crash_count(tmp_path, monkeypatch) -> None:
    # setting-level config params go to config.json; setting_metadata.json holds only n_crashes (a mutable
    # counter the campaign runner bumps on crashes). A later trial of the same setting must re-assert config.json
    # unchanged and must not reset the crash count.
    monkeypatch.setattr(ArtifactManager, "dpath_setting", tmp_path)
    cfg = _FakeSettingCfg()

    ArtifactManager.save_metadata_setting(cfg)
    config = json.loads((tmp_path / "config.json").read_text())
    assert "loss" in config and "setting" not in config  # config params kept, identity keys stripped
    assert json.loads((tmp_path / "setting_metadata.json").read_text()) == {"n_crashes": {"ram": 0, "vram": 0, "other": 0}}

    (tmp_path / "setting_metadata.json").write_text(json.dumps({"n_crashes": {"ram": 1, "vram": 2, "other": 4}}))  # runner bumps it across crashes
    ArtifactManager.save_metadata_setting(cfg)  # a later trial re-saves: must not raise, must not reset the counts
    assert json.loads((tmp_path / "setting_metadata.json").read_text()) == {"n_crashes": {"ram": 1, "vram": 2, "other": 4}}
    assert json.loads((tmp_path / "config.json").read_text()) == config


def test_save_metadata_setting_prunes_inert_params(tmp_path, monkeypatch) -> None:
    # absence in config.json is the inert signal (the stats overrides table renders absent params
    # as '-'): every param another param renders inert must be pruned from the saved dict

    # SigLIP + BCE + inv_freq + loss2 off (the fake's defaults)
    (tmp_path / "s1").mkdir()
    monkeypatch.setattr(ArtifactManager, "dpath_setting", tmp_path / "s1")
    ArtifactManager.save_metadata_setting(_FakeSettingCfg())
    config = json.loads((tmp_path / "s1" / "config.json").read_text())
    assert "clip" not in config["arch"]  # non_causal is CLIP-only
    assert "proj_head" not in config["dropout"]["siglip"]  # arch.siglip.vis_proj_head null -> no head to drop out
    assert "stoch_depth" in config["dropout"]["siglip"]
    assert "loss2" not in config  # mix 0.0
    cls_imb = config["loss"]["wting"]["cls_imb"]
    assert "class_bal" not in cls_imb and cls_imb["inv_freq"] == {"gamma": 0.5}  # type inv_freq
    assert cls_imb["freq_type_2d"] == "naive" and cls_imb["wt_mean_type"] == "per_class"  # BCE weights 2D, no self-norm
    assert "comp_type" not in config["loss"]["wting"]["focal"]  # bce + sw: binary targets -> comp forms coincide
    assert config["loss"]["wting"]["bce"]["norm"] == {"cls_imb": True, "agg": False}  # no unit-scale / norm.agg -> the rescale sticks
    assert "freeze" in config["loss"]["logits"]["bias"]  # SigLIP logit_bias is a real Parameter

    # CLIP + InfoNCE1 + class_bal: the 1D path reads none of the 2D/BCE-only machinery
    (tmp_path / "s2").mkdir()
    monkeypatch.setattr(ArtifactManager, "dpath_setting", tmp_path / "s2")
    cfg = _FakeSettingCfg()
    cfg.arch = {"model_type": "clip_vitb16", "clip": {"non_causal": True}, "siglip": {"vis_proj_head": None}}
    cfg.loss = _full_loss_cfg(crit="infonce1")
    cfg.loss["wting"]["cls_imb"]["type"] = "class_bal"
    ArtifactManager.save_metadata_setting(cfg)
    config = json.loads((tmp_path / "s2" / "config.json").read_text())
    assert "siglip" not in config["arch"] and "siglip" not in config["dropout"]
    assert config["arch"]["clip"] == {"non_causal": True}
    wting = config["loss"]["wting"]
    assert "bce" not in wting  # BCE-only
    assert "comp_type" not in wting["focal"] and wting["focal"]["gamma"] == 2.0  # 1D focal path
    assert wting["cls_imb"] == {"type": "class_bal", "class_bal": {"beta": 0.9999}}  # inv_freq/freq_type_2d/wt_mean_type inert
    assert "bias" not in config["loss"]["logits"]  # CLIP + bias.init null -> fixed 0.0 buffer

    # all weight factors off -> whole wting block inert; loss2 unit-scale cancels its norm scalars
    (tmp_path / "s3").mkdir()
    monkeypatch.setattr(ArtifactManager, "dpath_setting", tmp_path / "s3")
    cfg = _FakeSettingCfg()
    cfg.loss["wting"]["cls_imb"]["type"] = None
    cfg.loss["wting"]["focal"]["gamma"] = 0.0
    cfg.loss["wting"]["bce"]["dsmr"] = False
    cfg.loss2["mix"] = 0.3
    cfg.loss2["mix_unit_scale"] = True
    ArtifactManager.save_metadata_setting(cfg)
    config = json.loads((tmp_path / "s3" / "config.json").read_text())
    assert "wting" not in config["loss"]
    assert config["loss2"]["mix"] == 0.3 and config["loss2"]["mix_unit_scale"] is True
    assert "norm" not in config["loss2"]["wting"]["bce"]  # cls_imb (prod agg) and agg both cancelled -> emptied out
    assert config["loss2"]["wting"]["focal"]["comp_type"] == 1  # bce + phylo: continuous targets keep comp_type live


def test_update_eval_appends_none_leaves_from_base_eval(tmp_path) -> None:
    # base eval computes no loss -> loss_raw/sim/targ carry None leaves; update_eval must append
    # them as placeholders (not crash on a bare None) so eval curves stay index-aligned across evals
    data = TrialData(tmp_path)
    data.eval_metrics = {
        "scores": {"comp": {"map": {"all": 0.5}}},
        "loss_raw": {"id": None},
        "sim": {"min": None, "max": None, "median": None, "mean": None},
        "targ": {"min": None, "max": None, "median": None, "mean": None},
    }
    data.update_eval(0)
    data.eval_metrics = {
        "scores": {"comp": {"map": {"all": 0.6}}},
        "loss_raw": {"id": 0.7},
        "sim": {"min": -0.02, "max": 0.16, "median": 0.09, "mean": 0.09},
        "targ": {"min": -1.0, "max": 1.0, "median": -1.0, "mean": -0.99},
    }
    data.update_eval(1000)

    assert data.data_eval["n_samps_seen"] == [0, 1000]
    assert data.data_eval["loss_raw"]["id"] == [None, 0.7]
    assert data.data_eval["sim"]["mean"] == [None, 0.09]
    assert data.data_eval["targ"]["min"] == [None, -1.0]


def test_load_base_eval_cache_misses_when_entry_lacks_needed_pieces(tmp_path, monkeypatch) -> None:
    # entries carry only what the caching trial computed -- a trial must read an entry missing a
    # piece it needs (projections for viz, embs for pooled) as a miss (recompute + upgrade the
    # entry) rather than hit the missing piece downstream in _write_base_eval; leaner trials
    # still reuse the entry
    fpath = tmp_path / "combo.pkl"
    monkeypatch.setattr(ArtifactManager, "base_eval_cache_fpath", lambda cfg: fpath)
    entry = {"metrics": {"scores": {"comp": {"map": {"all": "0.50"}}}}, "projections": None, "embs": None}

    assert ArtifactManager.load_base_eval_cache(None, require_projections=False, require_embs=False) is None  # no file for this combo

    save_pickle(entry, fpath)
    assert ArtifactManager.load_base_eval_cache(None, require_projections=True, require_embs=False) is None  # metrics-only entry, viz trial
    assert ArtifactManager.load_base_eval_cache(None, require_projections=False, require_embs=False) == entry

    entry_viz = {**entry, "projections": {"pca_id": [0.0]}}
    save_pickle(entry_viz, fpath)
    assert ArtifactManager.load_base_eval_cache(None, require_projections=True, require_embs=False) == entry_viz
    assert ArtifactManager.load_base_eval_cache(None, require_projections=True, require_embs=True) is None  # no embs, pooled trial

    entry_pooled = {**entry_viz, "embs": {"embs_id": [0.0]}}
    save_pickle(entry_pooled, fpath)
    assert ArtifactManager.load_base_eval_cache(None, require_projections=True, require_embs=True) == entry_pooled


def test_save_base_eval_cache_writes_per_combo_file(tmp_path, monkeypatch) -> None:
    # each save ingests the npz files compute_projections wrote into this trial's evals/_base/
    # (absent for non-viz trials -> None) and writes its combo's entry to that combo's own file,
    # leaving other combos' files untouched
    dpath_cache = tmp_path / "base_eval_cache"
    monkeypatch.setattr(ArtifactManager, "base_eval_cache_fpath", lambda cfg: dpath_cache / "combo.pkl")
    monkeypatch.setattr(ArtifactManager, "dpath_trial", tmp_path / "trial")
    dpath_base = tmp_path / "trial" / "evals" / "_base"
    dpath_base.mkdir(parents=True)
    np.savez(dpath_base / "projections.npz", pca_id=np.arange(3))
    eval_metrics = {"scores": {"comp": {"map": {"all": 0.5}}}, "loss_raw": {"id": 0.7, "ood": None}}

    ArtifactManager.save_base_eval_cache(None, eval_metrics)

    entry = load_pickle(dpath_cache / "combo.pkl")
    assert entry["metrics"] == {"scores": {"comp": {"map": {"all": "0.5000"}}}}  # loss_raw stripped
    assert list(entry["projections"]) == ["pca_id"]
    assert entry["embs"] is None

    monkeypatch.setattr(ArtifactManager, "base_eval_cache_fpath", lambda cfg: dpath_cache / "combo2.pkl")
    monkeypatch.setattr(ArtifactManager, "dpath_trial", tmp_path / "trial2")  # no _base npzs -> non-viz trial

    ArtifactManager.save_base_eval_cache(None, eval_metrics)

    assert sorted(p.name for p in dpath_cache.iterdir()) == ["combo.pkl", "combo2.pkl"]
    assert load_pickle(dpath_cache / "combo2.pkl")["projections"] is None


def test_base_eval_key_normalizes_family_inert_components() -> None:
    # non_causal is CLIP-only, vis_proj_head is SigLIP-only, and seed only enters through the random
    # init of a linear/mlp vis_proj_head -- inert components read as None so equivalent configs
    # share one cache entry
    def cfg(model_type, non_causal=False, vis_proj_head=None):
        return SimpleNamespace(
            arch={"model_type": model_type, "clip": {"non_causal": non_causal}, "siglip": {"vis_proj_head": vis_proj_head}},
            img_norm="default", dataset="cub", split="dev",
            text_template={"train": "train", "eval": "sci"}, seed=42,
        )

    assert ArtifactManager.base_eval_key(cfg("siglip_vitb16")) == \
        ("siglip_vitb16", "default", "cub", "dev", None, "sci", None, None)  # headless: seed shared
    assert ArtifactManager.base_eval_key(cfg("siglip_vitb16", vis_proj_head="mlp")) == \
        ("siglip_vitb16", "default", "cub", "dev", None, "sci", "mlp", 42)  # random head: seed kept
    assert ArtifactManager.base_eval_key(cfg("clip_vitb16", non_causal=True)) == \
        ("clip_vitb16", "default", "cub", "dev", True, "sci", None, None)
    # non_causal true vs false are two separate cached readings
    assert ArtifactManager.base_eval_key(cfg("clip_vitb16", non_causal=True)) != \
        ArtifactManager.base_eval_key(cfg("clip_vitb16", non_causal=False))
    # the combo key serializes to the flat per-combo cache filename
    fpath = ArtifactManager.base_eval_cache_fpath(cfg("siglip_vitb16", vis_proj_head="mlp"))
    assert fpath.parent.name == "base_eval_cache"
    assert fpath.name == "siglip_vitb16__default__cub__dev__None__sci__mlp__42.pkl"


def test_format_and_merge_mem_running_max() -> None:
    # bytes -> 'used/total GB' (GiB), and merge keeps the higher-used reading per key -- a running
    # max across snapshots; None (no reading yet) is always superseded
    snap = format_mem({"ram": (4.2 * 2**30, 128 * 2**30), "vram": (37.5 * 2**30, 79.3 * 2**30)})
    assert snap == {"ram": "4.2/128.0 GB", "vram": "37.5/79.3 GB"}

    assert merge_mem({"ram": None, "vram": None}, snap) == snap

    later = {"ram": "6.0/128.0 GB", "vram": "12.0/79.3 GB"}
    assert merge_mem(snap, later) == {"ram": "6.0/128.0 GB", "vram": "37.5/79.3 GB"}
