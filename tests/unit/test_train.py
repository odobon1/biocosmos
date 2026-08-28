import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from train import TrainPipeline, pass_epoch_span
from utils.train import ArtifactManager, TrialData, format_mem, merge_mem
from utils.utils import save_pickle, load_pickle


def _full_loss_cfg(crit="bce", targ="mp"):
    return {
        "crit": crit,
        "infonce": {"tsm": {"type": "linear", "sm_temp": "pinned"}},
        "bce": {"targ_mass_neut": False},
        "sim": "cos",
        "targ": targ,
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
            "temp": {"init": None, "freeze": False, "clamp": False},
            "bce": {"center": None, "bias": {"init": None, "freeze": False}},
        },
    }


@dataclass
class _FakeCoordCfg:
    campaign: str = "c"
    phase: str = "screening"
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
    dev: dict = field(default_factory=dict)
    arch: dict = field(default_factory=lambda: {
        "model_type": "siglip_vitb16", "clip": {"non_causal": False}, "siglip": {"vis_proj_head": None},
    })
    dropout: dict = field(default_factory=lambda: {
        "patch_dropout": 0.0, "siglip": {"proj_head": 0.0, "stoch_depth": None},
    })
    loss: dict = field(default_factory=_full_loss_cfg)
    loss2: dict = field(default_factory=lambda: {"mix": 0.0, "mix_unit_scale": False, **_full_loss_cfg(targ="phylo")})
    opt: dict = field(default_factory=lambda: {"lr": {"warmup": 0.04}})

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
    assert "loss" in config and "phase" not in config and "arm" not in config and "coord" not in config  # config params kept, identity keys stripped
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
    assert "loss2" not in config  # mix 0.0
    assert "infonce" not in config["loss"]  # InfoNCE-only sub-block
    assert "bce" not in config["loss"]  # targ_mass_neut is bif_bce-only
    cls_imb = config["loss"]["wting"]["cls_imb"]
    assert "class_bal" not in cls_imb and cls_imb["inv_freq"] == {"gamma": 0.5}  # type inv_freq
    assert cls_imb["norm"] is True  # no unit-scale -> the rescale sticks
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
    assert config["loss"]["infonce"] == {"tsm": {"type": "linear", "sm_temp": "pinned"}}  # infonce + mp: block live
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
    assert "infonce" not in config["loss"]
    assert config["loss"]["bce"] == {"targ_mass_neut": False}  # bif_bce reads it
    assert config["loss"]["wting"]["bce"] == {"dsmr": True}  # dsmr applies to bif_bce too

    # all weight factors off -> whole wting block inert; loss2 unit-scale cancels its norm scalars
    (tmp_path / "s3").mkdir()
    monkeypatch.setattr(ArtifactManager, "dpath_coord", tmp_path / "s3")
    cfg = _FakeCoordCfg()
    cfg.loss["wting"]["cls_imb"]["type"] = None
    del cfg.loss["wting"]["focal"]  # config load prunes the block when gamma = 0.0
    cfg.loss["wting"]["bce"]["dsmr"] = False
    cfg.loss2["mix"] = 0.3
    cfg.loss2["mix_unit_scale"] = True
    ArtifactManager.save_metadata_coord(cfg)
    config = json.loads((tmp_path / "s3" / "config.json").read_text())
    assert "wting" not in config["loss"]
    assert config["loss2"]["mix"] == 0.3 and config["loss2"]["mix_unit_scale"] is True
    cls_imb2 = config["loss2"]["wting"]["cls_imb"]
    assert "norm" not in cls_imb2  # its rescale is cancelled by unit-scaling


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


def _fake_pipe(loss_crit, loss2_crit, mix, requires_grad):
    """A stand-in TrainPipeline carrying just what _tracked_logit_scalars reads: the loss configs and
    an unwrapped model whose logit scalars have the given requires_grad flags."""
    model = SimpleNamespace(**{
        attr: SimpleNamespace(requires_grad=requires_grad[attr])
        for attr in ("logit_scale", "logit_bias", "logit_scale2", "logit_bias2")
    })
    return SimpleNamespace(
        cfg=SimpleNamespace(loss={"crit": loss_crit}, loss2={"crit": loss2_crit, "mix": mix}),
        modelw=SimpleNamespace(_unwrapped_model=model),
    )


def test_tracked_logit_scalars_skips_frozen_inert_and_inactive() -> None:
    # a scalar gets a learning-curve series only when it's learnable AND meaningful: bias is
    # BCE-family-only (inert under InfoNCE), loss2's pair only when loss2 is mixed in
    all_learnable = dict.fromkeys(("logit_scale", "logit_bias", "logit_scale2", "logit_bias2"), True)
    tracked = TrainPipeline._tracked_logit_scalars

    # loss2 off: only loss1's pair, and its bias only because crit is BCE-family
    assert tracked(_fake_pipe("bce", "bce", 0.0, all_learnable)) == {
        "temp1": "logit_scale", "bias1": "logit_bias",
    }
    assert tracked(_fake_pipe("infonce", "bce", 0.0, all_learnable)) == {"temp1": "logit_scale"}

    # loss2 mixed in: both pairs, each loss's bias gated by its OWN crit
    assert tracked(_fake_pipe("infonce", "bce", 0.3, all_learnable)) == {
        "temp1": "logit_scale", "temp2": "logit_scale2", "bias2": "logit_bias2",
    }

    # frozen scalars are dropped -- a flat line says nothing
    frozen_t1_b2 = {**all_learnable, "logit_scale": False, "logit_bias2": False}
    assert tracked(_fake_pipe("bce", "bce", 0.3, frozen_t1_b2)) == {
        "bias1": "logit_bias", "temp2": "logit_scale2",
    }


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


def _fake_targ_pipe(targ1, targ2, mix):
    return SimpleNamespace(cfg=SimpleNamespace(loss={"targ": targ1}, loss2={"targ": targ2, "mix": mix}))


def test_tracked_targ_stats_only_graded_targets_of_active_branches() -> None:
    # target stats are curved only for graded targets (phylo/tax); sp/mp are 0/1 indicators whose
    # spread says nothing, and loss2 counts only when it's mixed in
    tracked = TrainPipeline._tracked_targ_stats

    assert tracked(_fake_targ_pipe("phylo", "phylo", 0.3)) == {"targ1", "targ2"}
    assert tracked(_fake_targ_pipe("mp", "phylo", 0.3)) == {"targ2"}  # only loss2 qualifies
    assert tracked(_fake_targ_pipe("tax", "sp", 0.3)) == {"targ1"}
    assert tracked(_fake_targ_pipe("sp", "mp", 0.3)) == set()  # neither -> no Y panel at all
    # loss2 off: its targ is irrelevant however it's configured
    assert tracked(_fake_targ_pipe("phylo", "phylo", 0.0)) == {"targ1"}
    assert tracked(_fake_targ_pipe("mp", "phylo", 0.0)) == set()
