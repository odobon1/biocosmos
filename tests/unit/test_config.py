import pytest

from utils.config import GenSplitConfig, StatsConfig, TrainConfig
from utils.config import apply_overrides
from utils.config import apply_model_specific_opt_defaults


def make_train_config_dummy(**overrides):
    config = {
        "campaign": "campaign",
        "setting": "exp",
        "seed": 7,
        "dataset": "cub",
        "split": "D10",
        "train_pt": "train",
        "n_epochs": 1,
        "n_chkpts": 10,
        "batch_size": 8,
        "chain_floor": None,
        "dv_batching": False,
        "htarg_shuf": False,
        "dev": {"logging": False, "plot_every": "trial", "manifold_viz": {"n_trials": 1, "pooled": {"enabled": True, "budget": 1.0, "pca_bounds": None}}},
        "arch": {"model_type": "clip_vitb16", "clip": {"non_causal": False}, "siglip": {"vis_proj_head": None}},
        "dropout": {"patch_dropout": 0.0, "siglip": {"proj_head": 0.0, "stoch_depth": None}},
        "loss": {"crit": "bce", "sim": "cos", "targ": "sp", "wting": {"focal": {"gamma": 0.0}}, "logits": {"scalar_lr_factor": 1.0, "temp": {"init": None}, "bce": {"center": None, "bias": {"init": None}}}},
        "loss2": {"crit": "bce", "sim": "cos", "targ": "sp", "mix": 0.0, "wting": {"focal": {"gamma": 0.0}}, "logits": {"scalar_lr_factor": 1.0, "temp": {"init": None}, "bce": {"center": None, "bias": {"init": None}}}},
        "opt": {
            "lr": {"init": 1.0e-5, "decay_factor": 1.0e-3, "warmup": 0.02},
            "wd": 0.0,
            "beta1": 0.9,
            "beta2": 0.95,
            "eps": 1.0e-6,
        },
        "freeze": {"text": False, "image": True},
        "text_template": {"train": "train", "eval": "sci"},
        "hw": {
            "mixed_prec": True,
            "act_chkpt": False,
            "loss_chunk_size": None,
            "cudnn_benchmark": False,
            "prefetch_factor": 4,
            "max_n_workers_gpu": None,
            "pin_memory": True,
            "persistent_workers": {"train": True, "eval": True},
            "use_img_cache": False,
            "eval": {"map_chunk_size": {"img2img": 512, "cross_modal": 512}, "tsne_chunk_log2": 28},
            "ram_poll_interval": 1.0,
            "pg_timeout": 300,
            "max_retries": 2,
        },
    }
    config.update(overrides)
    return config


def patch_hw(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "utils.config.compute_dataloader_workers_prefetch",
        lambda *args, **kwargs: (2, 2, {"n_gpus": 1, "n_cpus": 4, "ram": 32}),
    )


def test_train_config_rejects_head_dropout_without_proj_head(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    with pytest.raises(ValueError, match="requires arch.siglip.vis_proj_head"):
        TrainConfig(**make_train_config_dummy(
            arch={"model_type": "siglip_vitb16", "clip": {"non_causal": False}, "siglip": {"vis_proj_head": None}},
            dropout={"patch_dropout": 0.0, "siglip": {"proj_head": 0.3, "stoch_depth": None}},
        ))


def test_train_config_rejects_freezing_both_encoders(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    with pytest.raises(ValueError, match="both set to frozen"):
        TrainConfig(**make_train_config_dummy(freeze={"text": True, "image": True}))


def test_train_config_rejects_invalid_secondary_mix(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    with pytest.raises(ValueError, match="Secondary loss mix out of bounds"):
        TrainConfig(**make_train_config_dummy(loss2={"crit": "bce", "sim": "cos", "targ": "sp", "mix": 1.5,
                                                     "logits": {"scalar_lr_factor": 1.0, "temp": {"init": None}, "bce": {"center": None, "bias": {"init": None}}}}))


def test_train_config_rejects_negative_viz_n_trials(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    with pytest.raises(ValueError, match="dev.manifold_viz.n_trials must be >= 0"):
        TrainConfig(**make_train_config_dummy(dev={"logging": False, "manifold_viz": {"n_trials": -1}}))


def test_train_config_rejects_nonpositive_pooled_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    with pytest.raises(ValueError, match="dev.manifold_viz.pooled.budget must be > 0"):
        TrainConfig(**make_train_config_dummy(
            dev={"logging": False, "manifold_viz": {"n_trials": 1, "pooled": {"enabled": True, "budget": 0.0}}}))


def test_train_config_rejects_invalid_pca_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    with pytest.raises(ValueError, match="dev.manifold_viz.pooled.pca_bounds must be null or 'final'"):
        TrainConfig(**make_train_config_dummy(
            dev={"logging": False, "manifold_viz": {"n_trials": 1, "pooled": {"enabled": True, "budget": 1.0, "pca_bounds": "first"}}}))


def test_train_config_rejects_yaml_string_scientific_notation(monkeypatch: pytest.MonkeyPatch) -> None:
    # YAML parses "1e-6" (no decimal point in the mantissa) as a STRING; a swept opt.lr.init like
    # that must fail at config time, not deep inside AdamW
    patch_hw(monkeypatch)

    with pytest.raises(ValueError, match="opt.lr.init must be numeric"):
        TrainConfig(**make_train_config_dummy(opt={
            "lr": {"init": "1e-6", "decay_factor": 1.0e-3, "warmup": 0.02},
            "wd": 0.0,
            "beta1": 0.9,
            "beta2": 0.95,
            "eps": 1.0e-6,
        }))


def test_train_config_rejects_warmup_out_of_range(monkeypatch: pytest.MonkeyPatch) -> None:
    # opt.lr.warmup is a fraction of sample_volume -- a stale absolute sample count must fail loudly
    patch_hw(monkeypatch)

    with pytest.raises(ValueError, match="opt.lr.warmup must be a fraction of sample_volume"):
        TrainConfig(**make_train_config_dummy(opt={
            "lr": {"init": 1.0e-5, "decay_factor": 1.0e-3, "warmup": 200_000},
            "wd": 0.0,
            "beta1": 0.9,
            "beta2": 0.95,
            "eps": 1.0e-6,
        }))


def test_train_config_rejects_unknown_plot_every(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    cfg_dict = make_train_config_dummy()
    cfg_dict["dev"]["plot_every"] = "epoch"
    with pytest.raises(ValueError, match="dev.plot_every must be 'trial' or 'chkpt'"):
        TrainConfig(**cfg_dict)


def test_train_config_rejects_htarg_shuf_without_phylo_target(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    with pytest.raises(ValueError, match="requires an active phylo target"):
        TrainConfig(**make_train_config_dummy(htarg_shuf=True))


def test_train_config_accepts_htarg_shuf_with_secondary_phylo(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    cfg = TrainConfig(**make_train_config_dummy(
        htarg_shuf=True,
        loss2={"crit": "bce", "sim": "cos", "targ": "phylo", "mix": 0.3, "wting": {"focal": {"gamma": 0.0}}, "logits": {"scalar_lr_factor": 1.0, "temp": {"init": None}, "bce": {"center": None, "bias": {"init": None}}}},
    ))

    assert cfg.htarg_shuf is True


def test_train_config_rejects_htarg_shuf_with_null_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    with pytest.raises(ValueError, match="requires a non-null seed"):
        TrainConfig(**make_train_config_dummy(
            htarg_shuf=True,
            seed=None,
            loss={"crit": "bce", "sim": "cos", "targ": "phylo", "logits": {"scalar_lr_factor": 1.0, "temp": {"init": None}, "bce": {"center": None, "bias": {"init": None}}}},
        ))


def test_train_config_populates_runtime_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    cfg = TrainConfig(**make_train_config_dummy())

    assert cfg.n_workers == 2
    assert cfg.prefetch_factor == 2
    assert cfg.n_gpus == 1
    assert str(cfg.device) == "cuda"


def test_train_config_reads_hw_from_cfg_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    # hw comes from cfg_dict (frozen baseline / hw.* override), not re-read live from hardware.yaml
    patch_hw(monkeypatch)

    cfg = TrainConfig(**make_train_config_dummy(hw={
        "mixed_prec": False,
        "act_chkpt": True,
        "loss_chunk_size": None,
        "cudnn_benchmark": True,
        "prefetch_factor": 8,
        "max_n_workers_gpu": 3,
        "pin_memory": False,
        "persistent_workers": {"train": False, "eval": False},
        "use_img_cache": False,
        "eval": {"map_chunk_size": {"img2img": 1024, "cross_modal": 1024}, "tsne_chunk_log2": 30},
        "ram_poll_interval": 0.5,
        "pg_timeout": 300,
        "max_retries": 2,
    }))

    assert cfg.use_img_cache is False
    assert cfg.hw.mixed_prec is False
    assert cfg.hw.act_chkpt is True
    assert cfg.hw.prefetch_factor == 8
    assert cfg.hw.max_n_workers_gpu == 3
    assert cfg.hw.pin_memory is False
    assert cfg.hw.eval["tsne_chunk_log2"] == 30
    assert cfg.hw.ram_poll_interval == 0.5


def test_apply_overrides_dot_path_sets_single_nested_field() -> None:
    base = {
        "loss": {
            "targ": "sp",
            "sim": "cos",
        }
    }
    overrides = {
        "loss.targ": "mp",
    }

    out = apply_overrides(base, overrides)

    assert out["loss"]["targ"] == "mp"
    assert out["loss"]["sim"] == "cos"


def test_apply_overrides_dot_path_navigates_nested_dict() -> None:
    base = {
        "loss": {
            "targ": "sp",
            "sim": "cos",
        }
    }
    overrides = {
        "loss.targ": "phylo",
    }

    out = apply_overrides(base, overrides)

    assert out["loss"]["targ"] == "phylo"
    assert out["loss"]["sim"] == "cos"


def test_apply_overrides_dot_path_preserves_sibling_keys() -> None:
    base = {
        "opt": {
            "lr": {
                "init": 1.0e-5,
                "decay_factor": 1.0e-3,
                "warmup": 100,
            }
        }
    }
    overrides = {
        "opt.lr.decay_factor": 1.0e-2,
    }

    out = apply_overrides(base, overrides)

    assert out["opt"]["lr"]["init"] == 1.0e-5
    assert out["opt"]["lr"]["warmup"] == 100
    assert out["opt"]["lr"]["decay_factor"] == 1.0e-2


def test_apply_overrides_rejects_undeclared_leaf() -> None:
    base = {"opt": {"wd": 0.2, "lr": {"init": 1.0e-5}}}

    with pytest.raises(ValueError, match=r"opt\.l2_reg"):
        apply_overrides(base, {"opt.l2_reg": 0.0})


def test_apply_overrides_rejects_undeclared_section() -> None:
    base = {"loss": {"targ": "mp", "crit": "infonce"}}

    with pytest.raises(ValueError, match=r"loss\.infonce"):
        apply_overrides(base, {"loss.infonce.targ_mass_preservation": True})


def test_apply_overrides_rejects_path_through_scalar() -> None:
    base = {"opt": {"lr": {"init": 1.0e-5}}}

    with pytest.raises(ValueError, match=r"opt\.lr\.init\.foo"):
        apply_overrides(base, {"opt.lr.init.foo": 1})


def test_apply_overrides_allows_declared_null_field() -> None:
    base = {"opt": {"wd": None, "beta2": None}}

    out = apply_overrides(base, {"opt.wd": 0.1})

    assert out["opt"]["wd"] == 0.1
    assert out["opt"]["beta2"] is None


def test_model_specific_opt_defaults_resolve_siglip_nulls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "utils.config.load_model_specific_config_dict",
        lambda: {
            "siglip": {"wd": 0.0, "beta2": 0.95},
            "clip": {"wd": 0.2, "beta2": 0.98},
        },
    )

    cfg_in = make_train_config_dummy(
        arch={"model_type": "siglip_vitb16", "clip": {"non_causal": False}},
        opt={"lr": {"decay_factor": 1.0e-3}, "wd": None, "beta1": 0.9, "beta2": None, "eps": 1.0e-6},
    )

    out = apply_model_specific_opt_defaults(cfg_in)

    assert out["opt"]["wd"] == 0.0
    assert out["opt"]["beta2"] == 0.95


def test_model_specific_opt_defaults_preserve_explicit_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "utils.config.load_model_specific_config_dict",
        lambda: {
            "siglip": {"wd": 0.0, "beta2": 0.95},
            "clip": {"wd": 0.2, "beta2": 0.98},
        },
    )

    cfg_in = make_train_config_dummy(
        arch={"model_type": "clip_vitb16", "clip": {"non_causal": False}},
        opt={"lr": {"decay_factor": 1.0e-3}, "wd": 0.11, "beta1": 0.9, "beta2": 0.77, "eps": 1.0e-6},
    )

    out = apply_model_specific_opt_defaults(cfg_in)

    assert out["opt"]["wd"] == 0.11
    assert out["opt"]["beta2"] == 0.77


def test_model_specific_opt_defaults_resolve_partial_null(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "utils.config.load_model_specific_config_dict",
        lambda: {
            "siglip": {"wd": 0.0, "beta2": 0.95},
            "clip": {"wd": 0.2, "beta2": 0.98},
        },
    )

    cfg_in = make_train_config_dummy(
        arch={"model_type": "clip_vitb16", "clip": {"non_causal": False}},
        opt={"lr": {"decay_factor": 1.0e-3}, "wd": None, "beta1": 0.9, "beta2": 0.7, "eps": 1.0e-6},
    )

    out = apply_model_specific_opt_defaults(cfg_in)

    assert out["opt"]["wd"] == 0.2
    assert out["opt"]["beta2"] == 0.7


def test_model_specific_opt_defaults_unknown_model_type_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "utils.config.load_model_specific_config_dict",
        lambda: {
            "siglip": {"wd": 0.0, "beta2": 0.95},
            "clip": {"wd": 0.2, "beta2": 0.98},
        },
    )

    cfg_in = make_train_config_dummy(
        arch={"model_type": "mystery_model", "clip": {"non_causal": False}},
        opt={"lr": {"decay_factor": 1.0e-3}, "wd": None, "beta1": 0.9, "beta2": None, "eps": 1.0e-6},
    )

    with pytest.raises(ValueError, match="Could not resolve model family"):
        apply_model_specific_opt_defaults(cfg_in)


def test_model_specific_opt_defaults_use_passed_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    # a campaign trial passes the frozen snapshot; the live model_specific.yaml must not be read
    def _boom():
        raise AssertionError("model_specific.yaml must not be read when a snapshot is passed")
    monkeypatch.setattr("utils.config.load_model_specific_config_dict", _boom)

    cfg_in = make_train_config_dummy(
        arch={"model_type": "clip_vitb16", "clip": {"non_causal": False}},
        opt={"lr": {"decay_factor": 1.0e-3}, "wd": None, "beta1": 0.9, "beta2": None, "eps": 1.0e-6},
    )

    snapshot = {"siglip": {"wd": 0.0, "beta2": 0.95}, "clip": {"wd": 0.2, "beta2": 0.98}}
    out = apply_model_specific_opt_defaults(cfg_in, snapshot)

    assert out["opt"]["wd"] == 0.2
    assert out["opt"]["beta2"] == 0.98


# cub D10 train split has 4_944 samples (the dummy's dataset/split)
def test_train_config_chain_floor_null_disables_chaining(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    cfg = TrainConfig(**make_train_config_dummy())

    assert cfg.chain_perms is None
    assert cfg.sample_volume == 4_944  # n_epochs 1 x train set size
    assert cfg.samps_per_pass == 4_944  # divisible by batch_size 8, no truncation
    assert cfg.epochs_per_pass == 1
    assert cfg.n_passes == 1


def test_train_config_chain_floor_below_train_set_disables_chaining(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    cfg = TrainConfig(**make_train_config_dummy(chain_floor=1_000))

    assert cfg.chain_perms is None


def test_train_config_chain_floor_chains_permutations(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    cfg = TrainConfig(**make_train_config_dummy(chain_floor=100_000, n_epochs=202))

    assert cfg.chain_perms == 21  # E_chain_nom = ceil(100_000 / 4_944)
    assert cfg.samps_per_pass == 103_824  # X_chain = 21 x 4_944 (divisible by batch_size 8)
    assert cfg.epochs_per_pass == 21  # E_chain: no truncation, all 21 permutations fully covered
    assert cfg.sample_volume == 998_688  # 202 x 4_944
    assert cfg.n_passes == 10  # ceil(998_688 / 103_824)


def test_train_config_chaining_credits_epochs_touched_by_truncated_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    # batch_size > train set size: drop_last trims more than a full permutation off the nominal
    # chain (103_824 -> 98_304 consumed, 5_520 trimmed > 4_944), so the pass credits only the
    # permutations it actually touches: ceil(98_304 / 4_944) = 20 < chain_perms 21
    patch_hw(monkeypatch)

    cfg = TrainConfig(**make_train_config_dummy(chain_floor=100_000, batch_size=8_192))

    assert cfg.chain_perms == 21
    assert cfg.samps_per_pass == 98_304  # 12 batches of 8_192
    assert cfg.epochs_per_pass == 20


def test_train_config_rejects_batch_size_above_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    with pytest.raises(ValueError, match="exceeds epoch size"):
        TrainConfig(**make_train_config_dummy(batch_size=8_192))

    with pytest.raises(ValueError, match="exceeds epoch size"):
        TrainConfig(**make_train_config_dummy(chain_floor=5_000, batch_size=16_384))  # 2 perms = 9_888


def test_train_config_rejects_nonpositive_chain_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    with pytest.raises(ValueError, match="chain_floor must be greater than 0"):
        TrainConfig(**make_train_config_dummy(chain_floor=0))


def test_train_config_rejects_batch_size_indivisible_by_loss_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    cfg_dict = make_train_config_dummy(batch_size=24)
    cfg_dict["hw"]["loss_chunk_size"] = 16  # 24 % (1 * 16) != 0; ragged band unsupported

    with pytest.raises(ValueError, match="must be an exact multiple of world_size"):
        TrainConfig(**cfg_dict)


def test_train_config_rejects_batch_size_indivisible_by_world_size_x_loss_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    # divisible by the chunk alone but not by world_size x chunk: the BxB rows must split into
    # world_size equal bands of whole chunks (world_size = the alloc's GPU count, one rank per GPU)
    monkeypatch.setattr(
        "utils.config.compute_dataloader_workers_prefetch",
        lambda *args, **kwargs: (2, 2, {"n_gpus": 2, "n_cpus": 4, "ram": 32}),
    )

    cfg_dict = make_train_config_dummy(batch_size=16)
    cfg_dict["hw"]["loss_chunk_size"] = 16  # 16 % (2 * 16) != 0

    with pytest.raises(ValueError, match="must be an exact multiple of world_size"):
        TrainConfig(**cfg_dict)


def test_train_config_accepts_batch_size_divisible_by_loss_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    cfg_dict = make_train_config_dummy(batch_size=32)
    cfg_dict["hw"]["loss_chunk_size"] = 16  # 32 % 16 == 0

    cfg = TrainConfig(**cfg_dict)
    assert cfg.hw.loss_chunk_size == 16


def test_train_config_infonce_makes_chunking_inert(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    cfg_dict = make_train_config_dummy()  # batch_size 8
    cfg_dict["loss"] = {"crit": "infonce", "sim": "cos", "targ": "mp", "wting": {"focal": {"gamma": 0.0}},
                        "logits": {"scalar_lr_factor": 1.0, "temp": {"init": None}, "bce": {"center": None, "bias": {"init": None}}}}
    cfg_dict["hw"]["loss_chunk_size"] = 8  # ignored with InfoNCE: nulled out, no error

    cfg = TrainConfig(**cfg_dict)
    assert cfg.hw.loss_chunk_size is None


def test_train_config_bif_bce_keeps_chunking(monkeypatch: pytest.MonkeyPatch) -> None:
    # bif_bce is BCE-family: the tiled loss supports it, so the chunk size survives config
    patch_hw(monkeypatch)

    cfg_dict = make_train_config_dummy()  # batch_size 8
    cfg_dict["loss"]["crit"] = "bif_bce"
    cfg_dict["hw"]["loss_chunk_size"] = 8

    cfg = TrainConfig(**cfg_dict)
    assert cfg.hw.loss_chunk_size == 8


def test_train_config_rejects_unknown_center(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    cfg_dict = make_train_config_dummy()
    cfg_dict["loss"]["logits"]["bce"]["center"] = "grad_proje"
    with pytest.raises(ValueError, match="Unknown Loss 1 logits.bce.center"):
        TrainConfig(**cfg_dict)


def test_train_config_rejects_sim_center_with_geo_under_chunking(monkeypatch: pytest.MonkeyPatch) -> None:
    # center: sim under the tiled loss needs the cos mean factorization for an exact global sim mean
    patch_hw(monkeypatch)

    cfg_dict = make_train_config_dummy()  # batch_size 8
    cfg_dict["loss"] = {"crit": "bce", "sim": "geo1", "targ": "mp", "wting": {"focal": {"gamma": 0.0}},
                        "logits": {"scalar_lr_factor": 1.0, "temp": {"init": None}, "bce": {"center": "sim", "bias": {"init": None}}}}
    cfg_dict["hw"]["loss_chunk_size"] = 8

    with pytest.raises(ValueError, match="center: sim requires loss.sim: cos"):
        TrainConfig(**cfg_dict)


def test_train_config_rejects_sim_center_with_geo_under_chunking_bif(monkeypatch: pytest.MonkeyPatch) -> None:
    # bif_bce configs now reach the chunking validations (chunking no longer nulled for them)
    patch_hw(monkeypatch)

    cfg_dict = make_train_config_dummy()  # batch_size 8
    cfg_dict["loss"] = {"crit": "bif_bce", "sim": "geo1", "targ": "mp", "wting": {"focal": {"gamma": 0.0}},
                        "logits": {"scalar_lr_factor": 1.0, "temp": {"init": None}, "bce": {"center": "sim", "bias": {"init": None}}}}
    cfg_dict["hw"]["loss_chunk_size"] = 8

    with pytest.raises(ValueError, match="center: sim requires loss.sim: cos"):
        TrainConfig(**cfg_dict)


def _make_stats_config_dummy(**overrides):
    config = {
        "spread_type": "std",
        "bold_high": True,
        "ordered": True,
        "heatmap": None,
        "prim_scores": False,
        "baseline_overrides": False,
        "hw_perf": False,
    }
    config.update(overrides)
    return config


def test_stats_config_rejects_invalid_spread_type() -> None:
    with pytest.raises(ValueError, match="spread_type"):
        StatsConfig(**_make_stats_config_dummy(spread_type="var"))
