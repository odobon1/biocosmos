import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from train import TrainPipeline, kill_chkpt, pass_epoch_span, samps_stop
from utils.train import ArtifactManager, TrialData, format_mem, merge_mem
from utils.utils import save_pickle, load_pickle


def _full_loss_cfg(crit="bce", lambda_=0.0):
    return {
        "crit": crit,
        "sim": "cos",
        "blend": {"lambda": lambda_, "type": "targ"},
        "unitless": False,
        "bce": {"targ_mass_neut": False},
        "wting": {
            "cls_imb": {
                "type": "inv_freq",
                "inv_freq": {"gamma": 0.5},
                "class_bal": {"beta": 0.9999},
                "norm": True,
            },
            "focal": {"gamma": 2.0},
            "bce": {"dsmr": True},
        },
        "logits": {
            "shared": True,
            "scale": {"init": None, "freeze": False, "clamp": False},
            "bce": {"center": None, "bias": {"init": None, "freeze": False}},
        },
        "loss1": _targ_cfg("mp"),
        "loss2": _targ_cfg("phylo"),
    }


def _targ_cfg(targ):
    return {"targ": targ, "infonce": {"tsm": {"type": "linear", "sm_scale": "pinned"}}}


@dataclass
class _FakeCoordCfg:
    campaign: str = "c"
    phase: str = "_screen"
    arm: str = "sp"
    coord: str = "base"
    seed: int = 42
    idx_seed: int = 0
    idx_trial: int = 1
    n_trials_total: int = 8
    dataset: str = "cub"
    split: str = "D10"
    n_epochs: int = 5
    n_chkpts: int = 5
    batch_size: int = 1_024
    dev: bool = False
    reporting: dict = field(default_factory=dict)
    kill_thresh: float | None = None
    del_base_eval_cache: str | None = None
    arch: dict = field(default_factory=lambda: {
        "model_type": "siglip_vitb16", "clip": {"non_causal": False}, "siglip": {"vis_proj_head": None},
    })
    dropout: dict = field(default_factory=lambda: {
        "patch_dropout": 0.0, "siglip": {"proj_head": 0.0, "stoch_depth": None},
    })
    loss: dict = field(default_factory=_full_loss_cfg)
    opt: dict = field(default_factory=dict)
    lr: dict = field(default_factory=lambda: {"warmup": 0.04})

    def __post_init__(self):
        self.sample_volume = 102_500  # derived in TrainConfig.__post_init__, not a config field


def test_save_metadata_coord_splits_config_and_crash_count(tmp_path, monkeypatch) -> None:
    # coord-level config params go to config.json; coord_metadata.json holds the mutable state --
    # n_crashes (bumped by the campaign runner), best_chkpt (rewritten at each trial end), and the
    # precomputed horizon (sample/step totals with their LR-warmup shares). A later
    # trial of the same coord must re-assert config.json unchanged and must not reset the state.
    monkeypatch.setattr(ArtifactManager, "dpath_coord", tmp_path)
    cfg = _FakeCoordCfg()

    ArtifactManager.save_metadata_coord(cfg)
    config = json.loads((tmp_path / "config.json").read_text())
    assert "loss" in config and "loss1" in config["loss"] and "phase" not in config and "arm" not in config and "coord" not in config  # config params kept, identity keys stripped
    assert "n_epochs" not in config and "n_chkpts" not in config  # dataset-resolved, not coord params
    assert json.loads((tmp_path / "coord_metadata.json").read_text()) == {
        "n_crashes": {"ram": 0, "vram": 0, "other": 0},
        "horizon": {
            # warmup is the share OF each total: round(0.04 x 102_500) samples, ceil'd to steps
            "n_samps": {"total": 102_500, "warmup": 4_100},
            # ceil(102_500 / 1_024): the partial final batch still steps
            "n_steps": {"total": 101, "warmup": 5},
        },
        "best_chkpt": {},
    }

    metadata = {
        "n_crashes": {"ram": 1, "vram": 2, "other": 4},
        "horizon": {
            "n_samps": {"total": 102_500, "warmup": 4_100},
            "n_steps": {"total": 101, "warmup": 5},
        },
        "best_chkpt": {"map": {"native": {"idx": 3}}},
    }
    (tmp_path / "coord_metadata.json").write_text(json.dumps(metadata))  # runner/trials mutate it
    ArtifactManager.save_metadata_coord(cfg)  # a later trial re-saves: must not raise, must not reset the state
    assert json.loads((tmp_path / "coord_metadata.json").read_text()) == metadata
    assert json.loads((tmp_path / "config.json").read_text()) == config

    cfg.n_epochs = 2  # n_epochs is pruned from config.json, so a differing resolved duration still matches it
    ArtifactManager.save_metadata_coord(cfg)
    assert json.loads((tmp_path / "config.json").read_text()) == config


def test_save_metadata_coord_prunes_inert_params(tmp_path, monkeypatch) -> None:
    # absence in config.json is the inert signal (the stats overrides table renders absent params
    # as '-'): every param another param renders inert must be pruned from the saved dict

    # SigLIP + BCE + inv_freq + loss2 off (the fake's defaults)
    (tmp_path / "s1").mkdir()
    monkeypatch.setattr(ArtifactManager, "dpath_coord", tmp_path / "s1")
    ArtifactManager.save_metadata_coord(_FakeCoordCfg())
    config = json.loads((tmp_path / "s1" / "config.json").read_text())
    assert "clip" not in config["arch"]  # non_causal is CLIP-only
    assert "proj_head" not in config["dropout"]["siglip"]  # arch.siglip.vis_proj_head null -> no head to drop out
    assert "stoch_depth" in config["dropout"]["siglip"]
    assert "loss2" not in config["loss"]  # lambda 0.0
    assert "type" not in config["loss"]["blend"] and config["loss"]["unitless"] is False  # a lone target: nothing to blend
    assert "shared" not in config["loss"]["logits"]  # ... and no second term to give its own logit scalars
    assert config["loss"]["loss1"] == {"targ": "mp"}  # the InfoNCE-only tsm sub-block pruned under a BCE crit
    assert "bce" not in config["loss"]  # targ_mass_neut is bif_bce-only
    cls_imb = config["loss"]["wting"]["cls_imb"]
    assert "class_bal" not in cls_imb and cls_imb["inv_freq"] == {"gamma": 0.5}  # type inv_freq
    assert cls_imb["norm"] is True
    assert "freeze" in config["loss"]["logits"]["bce"]["bias"]  # SigLIP logit_bias is a real Parameter

    # CLIP + InfoNCE + class_bal: the 1D path reads none of the BCE-only machinery
    (tmp_path / "s2").mkdir()
    monkeypatch.setattr(ArtifactManager, "dpath_coord", tmp_path / "s2")
    cfg = _FakeCoordCfg()
    cfg.arch = {"model_type": "clip_vitb16", "clip": {"non_causal": True}, "siglip": {"vis_proj_head": None}}
    cfg.loss = _full_loss_cfg(crit="infonce")
    cfg.loss["wting"]["cls_imb"]["type"] = "class_bal"
    cfg.loss["wting"]["cls_imb"]["norm"] = False
    ArtifactManager.save_metadata_coord(cfg)
    config = json.loads((tmp_path / "s2" / "config.json").read_text())
    assert "siglip" not in config["arch"] and "siglip" not in config["dropout"]
    assert config["arch"]["clip"] == {"non_causal": True}
    assert config["loss"]["loss1"]["infonce"] == {"tsm": {"type": "linear", "sm_scale": "pinned"}}  # infonce + mp: block live
    wting = config["loss"]["wting"]
    assert "bce" not in wting  # BCE-only
    assert wting["cls_imb"] == {  # inv_freq inert (type class_bal)
        "type": "class_bal", "class_bal": {"beta": 0.9999}, "norm": False,
    }
    assert "bias" not in config["loss"]["logits"]["bce"]  # CLIP + bias.init null -> fixed 0.0 buffer

    # bif_bce: 1D per-anchor weighting, and the BCE-family blocks stay live
    (tmp_path / "s4").mkdir()
    monkeypatch.setattr(ArtifactManager, "dpath_coord", tmp_path / "s4")
    cfg = _FakeCoordCfg()
    cfg.loss = _full_loss_cfg(crit="bif_bce")
    ArtifactManager.save_metadata_coord(cfg)
    config = json.loads((tmp_path / "s4" / "config.json").read_text())
    assert "infonce" not in config["loss"]["loss1"]
    assert config["loss"]["bce"] == {"targ_mass_neut": False}  # bif_bce reads it
    assert config["loss"]["wting"]["bce"] == {"dsmr": True}  # dsmr applies to bif_bce too

    # all weight factors off -> whole wting block inert; a live blend keeps both target specs
    (tmp_path / "s3").mkdir()
    monkeypatch.setattr(ArtifactManager, "dpath_coord", tmp_path / "s3")
    cfg = _FakeCoordCfg()
    cfg.loss["wting"]["cls_imb"]["type"] = None
    del cfg.loss["wting"]["focal"]  # config load prunes the block when gamma = 0.0
    cfg.loss["wting"]["bce"]["dsmr"] = False
    cfg.loss["blend"]["lambda"] = 0.3
    ArtifactManager.save_metadata_coord(cfg)
    config = json.loads((tmp_path / "s3" / "config.json").read_text())
    assert "wting" not in config["loss"]
    assert config["loss"]["blend"]["lambda"] == 0.3
    assert "type" not in config["loss"]["blend"]  # no loss factor reads the target: the blend types coincide
    assert config["loss"]["loss1"] == {"targ": "mp"} and config["loss"]["loss2"] == {"targ": "phylo"}

    # lambda 1.0: the primary target spec is never read
    (tmp_path / "s5").mkdir()
    monkeypatch.setattr(ArtifactManager, "dpath_coord", tmp_path / "s5")
    cfg = _FakeCoordCfg()
    cfg.loss["blend"]["lambda"] = 1.0
    ArtifactManager.save_metadata_coord(cfg)
    config = json.loads((tmp_path / "s5" / "config.json").read_text())
    assert "loss1" not in config["loss"] and config["loss"]["loss2"] == {"targ": "phylo"}

    # InfoNCE: the tsm sub-block is read for every live target, except under sp (the linear mapping is a no-op)
    (tmp_path / "s6").mkdir()
    monkeypatch.setattr(ArtifactManager, "dpath_coord", tmp_path / "s6")
    cfg = _FakeCoordCfg()
    cfg.loss = _full_loss_cfg(crit="infonce", lambda_=0.3)
    cfg.loss["loss1"] = _targ_cfg("sp")
    ArtifactManager.save_metadata_coord(cfg)
    config = json.loads((tmp_path / "s6" / "config.json").read_text())
    assert config["loss"]["loss1"] == {"targ": "sp"}
    assert config["loss"]["loss2"]["infonce"] == {"tsm": {"type": "linear", "sm_scale": "pinned"}}
    assert config["loss"]["blend"]["type"] == "targ"  # a live blend under focal: the blend types differ
    assert "shared" not in config["loss"]["logits"]  # a target blend is one loss on one set of logits

    # unitless: its rescale cancels cls_imb.norm's per-batch normalizer, and makes a blend's type matter
    (tmp_path / "s7").mkdir()
    monkeypatch.setattr(ArtifactManager, "dpath_coord", tmp_path / "s7")
    cfg = _FakeCoordCfg()
    cfg.loss = _full_loss_cfg(lambda_=0.3)
    cfg.loss["unitless"], cfg.loss["blend"]["type"] = True, "loss"
    del cfg.loss["wting"]["focal"]
    cfg.loss["wting"]["bce"]["dsmr"] = False
    ArtifactManager.save_metadata_coord(cfg)
    config = json.loads((tmp_path / "s7" / "config.json").read_text())
    assert (config["loss"]["blend"]["type"], config["loss"]["unitless"]) == ("loss", True)
    assert config["loss"]["logits"]["shared"] is True  # a live loss blend: its terms could run on separate scalars
    assert config["loss"]["wting"]["cls_imb"] == {"type": "inv_freq", "inv_freq": {"gamma": 0.5}}


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


def _cfg_eval_groups(**toggles):
    """A stub config carrying just reporting.eval -- what load_base_eval_cache reads off it."""
    eval_cfg = {"native_macro": False, "joint": False, "joint_macro": False, **toggles}
    return SimpleNamespace(reporting={"eval": eval_cfg})


def test_load_base_eval_cache_misses_when_entry_lacks_needed_pieces(tmp_path, monkeypatch) -> None:
    # entries carry only what the caching trial computed -- a trial must read an entry missing a
    # piece it needs (projections for viz, embs for pooled) as a miss (recompute + upgrade the
    # entry) rather than hit the missing piece downstream in _write_base_eval; leaner trials
    # still reuse the entry
    fpath = tmp_path / "combo.pkl"
    cfg = _cfg_eval_groups()
    monkeypatch.setattr(ArtifactManager, "base_eval_cache_fpath", lambda cfg: fpath)
    entry = {"metrics": {"scores": {"native": {"comp": {"map": {"all": "0.50"}}}}}, "projections": None, "embs": None}

    assert ArtifactManager.load_base_eval_cache(cfg, require_projections=False, require_embs=False) is None  # no file for this combo

    save_pickle(entry, fpath)
    assert ArtifactManager.load_base_eval_cache(cfg, require_projections=True, require_embs=False) is None  # metrics-only entry, viz trial
    assert ArtifactManager.load_base_eval_cache(cfg, require_projections=False, require_embs=False) == entry

    entry_viz = {**entry, "projections": {"pca_id": [0.0]}}
    save_pickle(entry_viz, fpath)
    assert ArtifactManager.load_base_eval_cache(cfg, require_projections=True, require_embs=False) == entry_viz
    assert ArtifactManager.load_base_eval_cache(cfg, require_projections=True, require_embs=True) is None  # no embs, pooled trial

    entry_pooled = {**entry_viz, "embs": {"embs_id": [0.0]}}
    save_pickle(entry_pooled, fpath)
    assert ArtifactManager.load_base_eval_cache(cfg, require_projections=True, require_embs=True) == entry_pooled


def test_load_base_eval_cache_misses_when_entry_lacks_an_eval_group_in_play(tmp_path, monkeypatch) -> None:
    # an entry written by a campaign with fewer eval groups in play carries only those groups' scores;
    # a campaign needing more must recompute rather than KeyError on the missing group downstream, while
    # the leaner campaign still reuses the richer entry
    fpath = tmp_path / "combo.pkl"
    monkeypatch.setattr(ArtifactManager, "base_eval_cache_fpath", lambda cfg: fpath)
    scores = {"native": {"comp": {"map": {"all": "0.50"}}}, "joint_macro": {"comp": {"map": {"all": "0.40"}}}}
    entry = {"metrics": {"scores": scores}, "projections": None, "embs": None}
    save_pickle(entry, fpath)

    cfg_lean = _cfg_eval_groups(joint_macro=True)
    cfg_rich = _cfg_eval_groups(joint_macro=True, joint=True)

    assert ArtifactManager.load_base_eval_cache(cfg_lean, require_projections=False, require_embs=False) == entry
    assert ArtifactManager.load_base_eval_cache(cfg_rich, require_projections=False, require_embs=False) is None


def test_save_base_eval_cache_writes_per_combo_file(tmp_path, monkeypatch) -> None:
    # each save ingests the npz files compute_projections wrote into this trial's evals/base/
    # (absent for non-viz trials -> None) and writes its combo's entry to that combo's own file,
    # leaving other combos' files untouched
    dpath_cache = tmp_path / "base_eval_cache"
    monkeypatch.setattr(ArtifactManager, "base_eval_cache_fpath", lambda cfg: dpath_cache / "combo.pkl")
    monkeypatch.setattr(ArtifactManager, "dpath_trial", tmp_path / "trial")
    dpath_base = tmp_path / "trial" / "evals" / "base"
    dpath_base.mkdir(parents=True)
    np.savez(dpath_base / "projections.npz", pca_id=np.arange(3))
    eval_metrics = {"scores": {"comp": {"map": {"all": 0.5}}}, "loss_raw": {"id": 0.7, "ood": None}}

    ArtifactManager.save_base_eval_cache(None, eval_metrics)

    entry = load_pickle(dpath_cache / "combo.pkl")
    assert entry["metrics"] == {"scores": {"comp": {"map": {"all": "0.5000"}}}}  # loss_raw stripped
    assert list(entry["projections"]) == ["pca_id"]
    assert entry["embs"] is None

    monkeypatch.setattr(ArtifactManager, "base_eval_cache_fpath", lambda cfg: dpath_cache / "combo2.pkl")
    monkeypatch.setattr(ArtifactManager, "dpath_trial", tmp_path / "trial2")  # no base npzs -> non-viz trial

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
            dataset="cub", split="dev",
            text_template={"train": "train", "eval": "sci"}, seed=42,
        )

    assert ArtifactManager.base_eval_key(cfg("siglip_vitb16")) == \
        ("siglip_vitb16", "cub", "dev", None, "sci", None, None)  # headless: seed shared
    assert ArtifactManager.base_eval_key(cfg("siglip_vitb16", vis_proj_head="mlp")) == \
        ("siglip_vitb16", "cub", "dev", None, "sci", "mlp", 42)  # random head: seed kept
    assert ArtifactManager.base_eval_key(cfg("clip_vitb16", non_causal=True)) == \
        ("clip_vitb16", "cub", "dev", True, "sci", None, None)
    # non_causal true vs false are two separate cached readings
    assert ArtifactManager.base_eval_key(cfg("clip_vitb16", non_causal=True)) != \
        ArtifactManager.base_eval_key(cfg("clip_vitb16", non_causal=False))
    # the combo key serializes to the flat per-combo cache filename
    fpath = ArtifactManager.base_eval_cache_fpath(cfg("siglip_vitb16", vis_proj_head="mlp"))
    assert fpath.parent.name == "base_eval_cache"
    assert fpath.name == "siglip_vitb16__cub__dev__None__sci__mlp__42.pkl"


def test_format_and_merge_mem_running_max() -> None:
    # bytes -> 'used/total GB' (GiB), and merge keeps the higher-used reading per key -- a running
    # max across snapshots; None (no reading yet) is always superseded
    snap = format_mem({"ram": (4.2 * 2**30, 128 * 2**30), "vram": (37.5 * 2**30, 79.3 * 2**30)})
    assert snap == {"ram": "4.2/128.0 GB", "vram": "37.5/79.3 GB"}

    assert merge_mem({"ram": None, "vram": None}, snap) == snap

    later = {"ram": "6.0/128.0 GB", "vram": "12.0/79.3 GB"}
    assert merge_mem(snap, later) == {"ram": "6.0/128.0 GB", "vram": "37.5/79.3 GB"}


def _fake_pipe(loss_crit, requires_grad, sep=False):
    """A stand-in TrainPipeline carrying just what _tracked_logit_scalars reads: the loss config and an
    unwrapped model whose logit scalars have the given requires_grad flags -- under `sep` a live loss blend
    on separate logit scalars, the model then carrying the second pair (same flags)."""
    attrs = {attr: SimpleNamespace(requires_grad=requires_grad[attr]) for attr in ("logit_scale", "logit_bias")}
    if sep:
        attrs.update({f"{attr}2": scalar for attr, scalar in attrs.items()})
    loss = {"crit": loss_crit, "blend": {"lambda": 0.3 if sep else 0.0, "type": "loss"}, "logits": {"shared": not sep}}
    return SimpleNamespace(cfg=SimpleNamespace(loss=loss), modelw=SimpleNamespace(_unwrapped_model=SimpleNamespace(**attrs)))


def test_tracked_logit_scalars_skips_frozen_and_inert() -> None:
    # a scalar gets a learning-curve series only when it's learnable AND meaningful: the bias is
    # BCE-family-only (inert under InfoNCE). The scale parameter feeds two series: scale (alpha) and
    # logit_scale (the parameter itself, log alpha)
    all_learnable = {"logit_scale": True, "logit_bias": True}
    tracked = TrainPipeline._tracked_logit_scalars

    assert tracked(_fake_pipe("bce", all_learnable)) == {"scale": "logit_scale", "logit_scale": "logit_scale", "bias": "logit_bias"}
    assert tracked(_fake_pipe("bif_bce", all_learnable)) == {"scale": "logit_scale", "logit_scale": "logit_scale", "bias": "logit_bias"}
    assert tracked(_fake_pipe("infonce", all_learnable)) == {"scale": "logit_scale", "logit_scale": "logit_scale"}

    # frozen scalars are dropped -- a flat line says nothing
    assert tracked(_fake_pipe("bce", {"logit_scale": False, "logit_bias": True})) == {"bias": "logit_bias"}
    assert tracked(_fake_pipe("infonce", {"logit_scale": False, "logit_bias": True})) == {}

    # separate logit scalars: loss2's term's pair gets its own series, under the same rules
    assert tracked(_fake_pipe("bce", all_learnable, sep=True)) == {
        "scale": "logit_scale", "logit_scale": "logit_scale", "bias": "logit_bias",
        "scale2": "logit_scale2", "logit_scale2": "logit_scale2", "bias2": "logit_bias2"}
    assert tracked(_fake_pipe("infonce", all_learnable, sep=True)) == {
        "scale": "logit_scale", "logit_scale": "logit_scale", "scale2": "logit_scale2", "logit_scale2": "logit_scale2"}
    assert tracked(_fake_pipe("bce", {"logit_scale": False, "logit_bias": True}, sep=True)) == {"bias": "logit_bias", "bias2": "logit_bias2"}


def test_logit_scalar_values_cap_the_scale_under_the_clamp() -> None:
    # the scale series carries the alpha the logits carry, exp(logit_scale) held at 100 by
    # loss.logits.scale.clamp (compute_logits' cap) once the raw parameter sits above ln(100); the
    # logit_scale series the parameter as the model holds it (log alpha: no exp, no clamp), the
    # bias series the raw bias
    model = SimpleNamespace(logit_scale=torch.tensor(140.0).log(), logit_bias=torch.tensor(-0.5))
    pipe = SimpleNamespace(
        cfg=SimpleNamespace(loss={"logits": {"scale": {"clamp": True}}}),
        modelw=SimpleNamespace(_unwrapped_model=model),
        _logit_scalars_tracked={"scale": "logit_scale", "logit_scale": "logit_scale", "bias": "logit_bias"},
    )
    values = TrainPipeline._logit_scalar_values(pipe)
    assert values == pytest.approx({"scale": 100.0, "logit_scale": torch.tensor(140.0).log().item(), "bias": -0.5}, rel=1e-5)
    # the second pair's scale (separate logit scalars) reads the same way: alpha, under the same cap
    model.logit_scale2 = torch.tensor(50.0).log()
    pipe._logit_scalars_tracked = {"scale2": "logit_scale2"}
    assert TrainPipeline._logit_scalar_values(pipe) == pytest.approx({"scale2": 50.0}, rel=1e-5)
    pipe._logit_scalars_tracked = {"scale": "logit_scale", "bias": "logit_bias"}
    pipe.cfg.loss["logits"]["scale"]["clamp"] = False  # unbounded: the raw scale, wherever it sits
    assert TrainPipeline._logit_scalar_values(pipe)["scale"] == pytest.approx(140.0, rel=1e-5)


def test_pass_epoch_span_shares_the_straddled_epoch_between_passes() -> None:
    # chain_floor 16_000, batch_size 512, train set 4_935, n_epochs 5 -> chain_perms 4, so a pass is
    # 4 x 4_935 = 19_740 nominal, batch-aligned down to 19_456 (38 batches); the 284-sample tail of
    # permutation 4 carries into pass 2. Pass 1 therefore covers epochs 1-4 (ending 284 samples shy
    # of epoch 4's end) and pass 2 picks epoch 4 back up and runs to 5 -- the boundary epoch counted
    # in BOTH passes, not skipped.
    cfg = SimpleNamespace(samps_per_pass=19_456, samps_per_epoch=4_935, sample_volume=24_675)

    assert pass_epoch_span(cfg, 1) == (1, 4)
    assert pass_epoch_span(cfg, 2) == (4, 5)


def test_pass_epoch_span_without_chaining_is_one_epoch_per_pass() -> None:
    # no chain-shuffle: an epoch IS a batch-truncated pass, so every pass spans exactly its own epoch
    cfg = SimpleNamespace(samps_per_pass=4_928, samps_per_epoch=4_928, sample_volume=24_640)

    assert [pass_epoch_span(cfg, p) for p in range(1, 6)] == [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]


def _fake_targ_pipe(targ1, targ2, lambda_):
    return SimpleNamespace(cfg=SimpleNamespace(
        loss={"blend": {"lambda": lambda_}, "loss1": {"targ": targ1}, "loss2": {"targ": targ2}}))


def test_tracked_targ_stats_graded_blends_only() -> None:
    # the blended targets are curved when graded: a live phylo/tax target, or two distinct targets
    # blended (their disagreements sit at lambda / 1 - lambda); a lone sp/mp target is a 0/1 indicator whose
    # spread says nothing, and a spec with zero weight does not count
    tracked = TrainPipeline._tracked_targ_stats

    assert tracked(_fake_targ_pipe("phylo", "phylo", 0.3)) is True
    assert tracked(_fake_targ_pipe("mp", "phylo", 0.3)) is True
    assert tracked(_fake_targ_pipe("tax", "sp", 0.3)) is True
    assert tracked(_fake_targ_pipe("sp", "mp", 0.3)) is True  # distinct 0/1 targets blend to a graded matrix
    assert tracked(_fake_targ_pipe("sp", "sp", 0.3)) is False
    # a zero-weight spec is irrelevant however it's configured
    assert tracked(_fake_targ_pipe("phylo", "phylo", 0.0)) is True
    assert tracked(_fake_targ_pipe("mp", "phylo", 0.0)) is False
    assert tracked(_fake_targ_pipe("mp", "phylo", 1.0)) is True
    assert tracked(_fake_targ_pipe("phylo", "mp", 1.0)) is False


def test_samps_stop_is_the_selected_checkpoint_threshold() -> None:
    # the trainval phase stops at chkpt_stop's threshold; null (every other phase) and the last index run to
    # sample_volume itself, which covers the skipped last mid-train threshold the way the final eval does
    cfg = SimpleNamespace(sample_volume=1_000, n_chkpts=10, chkpt_interval=100, chkpt_stop=None)
    assert samps_stop(cfg) == 1_000
    cfg.chkpt_stop = 3
    assert samps_stop(cfg) == 300
    cfg.chkpt_stop = 10
    assert samps_stop(cfg) == 1_000


def test_kill_chkpt_rounds_the_threshold_up_to_the_nearest_eval() -> None:
    # kill_thresh is a fraction of the run; the kill check runs at the train-time eval nearest it,
    # rounded up (ceil(kill_thresh * n_chkpts)); null turns it off
    cfg = SimpleNamespace(n_chkpts=10, kill_thresh=None)
    assert kill_chkpt(cfg) is None
    cfg.kill_thresh = 0.05
    assert kill_chkpt(cfg) == 1
    cfg.kill_thresh = 0.25
    assert kill_chkpt(cfg) == 3
    cfg.kill_thresh = 0.3
    assert kill_chkpt(cfg) == 3
    cfg.kill_thresh = 0.7  # 0.7 * 10 is 7.000000000000001 in floats: still eval 7, not 8
    assert kill_chkpt(cfg) == 7


def test_save_model_writes_unwrapped_state_dict(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ArtifactManager, "dpath_trial", tmp_path)
    model = torch.nn.Linear(2, 1)

    ArtifactManager.save_model(SimpleNamespace(_unwrapped_model=model))

    state = torch.load(tmp_path / "model.pt")
    assert set(state) == {"weight", "bias"}
    assert torch.equal(state["weight"], model.weight.detach())
