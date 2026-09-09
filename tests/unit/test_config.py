import pytest

from utils.config import CampaignConfig, GenSplitConfig, ManifoldVizConfig, StatsConfig, TrainConfig
from utils.config import apply_overrides
from utils.config import apply_model_specific_opt_defaults
from utils.config import apply_dataset_specific_defaults


def make_train_config_dummy(**overrides):
    config = {
        "campaign": "campaign",
        "phase": "_screen",
        "arm": "exp",
        "coord": "base",
        "seed": 7,
        "dataset": "cub",
        "split": "D10",
        "train_pt": "train",
        "n_epochs": 1,
        "n_chkpts": 10,
        "batch_size": 8,
        "chain_floor": None,
        "dv_batching": False,
        "htarg": {"kernel": "laplace", "exp": {"beta": 1.0}, "shuffle": False},
        "dev": {"logging": False, "plot_every": "trial"},
        "arch": {"model_type": "clip_vitb16", "clip": {"non_causal": False}, "siglip": {"vis_proj_head": None}},
        "dropout": {"patch_dropout": 0.0, "siglip": {"proj_head": 0.0, "stoch_depth": None}},
        "loss": {"mix": 0.0, "unitless": False},
        "loss1": {"crit": "bce", "sim": "cos", "targ": "sp", "wting": {"focal": {"gamma": 0.0}}, "logits": {"scalar_lr_factor": 1.0, "temp": {"init": None}, "bce": {"center": None, "bias": {"init": None}}}},
        "loss2": {"crit": "bce", "sim": "cos", "targ": "sp", "wting": {"focal": {"gamma": 0.0}}, "logits": {"scalar_lr_factor": 1.0, "temp": {"init": None}, "bce": {"center": None, "bias": {"init": None}}}},
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

    with pytest.raises(ValueError, match="loss.mix out of bounds"):
        TrainConfig(**make_train_config_dummy(loss={"mix": 1.5, "unitless": False}))


def test_train_config_rejects_non_int_n_epochs(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    with pytest.raises(ValueError, match="n_epochs must be an int"):
        TrainConfig(**make_train_config_dummy(n_epochs=2.5))

    # a null that skipped dataset-specific resolution (TrainConfig built without get_config_train)
    with pytest.raises(ValueError, match="n_epochs must be an int"):
        TrainConfig(**make_train_config_dummy(n_epochs=None))


def test_train_config_rejects_non_int_n_chkpts(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    with pytest.raises(ValueError, match="n_chkpts must be an int"):
        TrainConfig(**make_train_config_dummy(n_chkpts=2.5))

    # a null that skipped dataset-specific resolution (TrainConfig built without get_config_train)
    with pytest.raises(ValueError, match="n_chkpts must be an int"):
        TrainConfig(**make_train_config_dummy(n_chkpts=None))


def make_manif_viz_config_dummy(**overrides):
    config = {
        "n_seeds": 1,
        "n_seeds_offset": 0,
        "eval_duration": 1500,
        "bg_color": None,
        "plot_2panel": True,
        "plot_7panel": True,
        "plot_8panel": True,
        "pooled": {"enabled": True, "budget": 1.0, "pca_bounds": None},
        "umap": {"n_neighbors": 15, "min_dist": 0.1, "n_iter": None, "n_iter_sphere": 100},
        "orient": {"ema_tau": 0.5},
    }
    config.update(overrides)
    return config


def test_manif_viz_config_rejects_negative_n_seeds() -> None:
    with pytest.raises(ValueError, match="n_seeds must be >= 0"):
        ManifoldVizConfig(**make_manif_viz_config_dummy(n_seeds=-1))


def test_manif_viz_config_rejects_negative_n_seeds_offset() -> None:
    with pytest.raises(ValueError, match="n_seeds_offset must be >= 0"):
        ManifoldVizConfig(**make_manif_viz_config_dummy(n_seeds_offset=-1))


def test_manif_viz_config_rejects_nonpositive_pooled_budget() -> None:
    with pytest.raises(ValueError, match="pooled.budget must be > 0"):
        ManifoldVizConfig(**make_manif_viz_config_dummy(
            pooled={"enabled": True, "budget": 0.0, "pca_bounds": None}))


def test_manif_viz_config_rejects_invalid_pca_bounds() -> None:
    with pytest.raises(ValueError, match="pooled.pca_bounds must be null or 'final'"):
        ManifoldVizConfig(**make_manif_viz_config_dummy(
            pooled={"enabled": True, "budget": 1.0, "pca_bounds": "first"}))


def test_manif_viz_config_rejects_too_few_umap_neighbors() -> None:
    with pytest.raises(ValueError, match="umap.n_neighbors must be >= 2"):
        ManifoldVizConfig(**make_manif_viz_config_dummy(umap={"n_neighbors": 1, "min_dist": 0.1, "n_iter": None, "n_iter_sphere": 100}))


def test_manif_viz_config_rejects_out_of_range_umap_min_dist() -> None:
    with pytest.raises(ValueError, match=r"umap.min_dist must be in \[0.0, 1.0\)"):
        ManifoldVizConfig(**make_manif_viz_config_dummy(umap={"n_neighbors": 15, "min_dist": 1.0, "n_iter": None, "n_iter_sphere": 100}))


def test_manif_viz_config_rejects_out_of_range_ema_tau() -> None:
    with pytest.raises(ValueError, match=r"orient.ema_tau must be in \(0.0, 1.0\]"):
        ManifoldVizConfig(**make_manif_viz_config_dummy(orient={"ema_tau": 0.0}))


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


def test_train_config_rejects_htarg_shuffle_without_phylo_target(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    with pytest.raises(ValueError, match="requires an active phylo target"):
        TrainConfig(**make_train_config_dummy(htarg={"kernel": "laplace", "exp": {"beta": 1.0}, "shuffle": True}))


def test_train_config_accepts_htarg_shuffle_with_secondary_phylo(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    cfg = TrainConfig(**make_train_config_dummy(
        htarg={"kernel": "laplace", "exp": {"beta": 1.0}, "shuffle": True},
        loss={"mix": 0.3, "unitless": False},
        loss2={"crit": "bce", "sim": "cos", "targ": "phylo", "wting": {"focal": {"gamma": 0.0}}, "logits": {"scalar_lr_factor": 1.0, "temp": {"init": None}, "bce": {"center": None, "bias": {"init": None}}}},
    ))

    assert cfg.htarg["shuffle"] is True


def test_train_config_rejects_htarg_shuffle_with_null_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    with pytest.raises(ValueError, match="requires a non-null seed"):
        TrainConfig(**make_train_config_dummy(
            htarg={"kernel": "laplace", "exp": {"beta": 1.0}, "shuffle": True},
            seed=None,
            loss1={"crit": "bce", "sim": "cos", "targ": "phylo", "logits": {"scalar_lr_factor": 1.0, "temp": {"init": None}, "bce": {"center": None, "bias": {"init": None}}}},
        ))


def test_train_config_rejects_unknown_htarg_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    with pytest.raises(ValueError, match=r"Unknown htarg\.kernel"):
        TrainConfig(**make_train_config_dummy(htarg={"kernel": "rbf", "exp": {"beta": 1.0}, "shuffle": False}))


def test_train_config_rejects_nonpositive_htarg_beta(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    with pytest.raises(ValueError, match=r"htarg\.exp\.beta must be a positive number"):
        TrainConfig(**make_train_config_dummy(htarg={"kernel": "laplace", "exp": {"beta": 0.0}, "shuffle": False}))


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
        "loss1": {
            "targ": "sp",
            "sim": "cos",
        }
    }
    overrides = {
        "loss1.targ": "mp",
    }

    out = apply_overrides(base, overrides)

    assert out["loss1"]["targ"] == "mp"
    assert out["loss1"]["sim"] == "cos"


def test_apply_overrides_dot_path_navigates_nested_dict() -> None:
    base = {
        "loss1": {
            "targ": "sp",
            "sim": "cos",
        }
    }
    overrides = {
        "loss1.targ": "phylo",
    }

    out = apply_overrides(base, overrides)

    assert out["loss1"]["targ"] == "phylo"
    assert out["loss1"]["sim"] == "cos"


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
    base = {"loss1": {"targ": "mp", "crit": "infonce"}}

    with pytest.raises(ValueError, match=r"loss1\.infonce"):
        apply_overrides(base, {"loss1.infonce.targ_mass_preservation": True})


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


def test_dataset_specific_defaults_resolve_null_n_epochs_and_n_chkpts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "utils.config.load_dataset_specific_config_dict",
        lambda: {"cub": {"n_epochs": 100, "n_chkpts": 50}, "lepid": {"n_epochs": 20, "n_chkpts": 10}},
    )

    out = apply_dataset_specific_defaults(make_train_config_dummy(n_epochs=None, n_chkpts=None, dataset="cub"))
    assert (out["n_epochs"], out["n_chkpts"]) == (100, 50)
    out = apply_dataset_specific_defaults(make_train_config_dummy(n_epochs=None, n_chkpts=None, dataset="lepid"))
    assert (out["n_epochs"], out["n_chkpts"]) == (20, 10)


def test_dataset_specific_defaults_resolve_each_key_independently(monkeypatch: pytest.MonkeyPatch) -> None:
    # only the null key is filled; a set key keeps its value even though the yaml is read for the other
    monkeypatch.setattr(
        "utils.config.load_dataset_specific_config_dict",
        lambda: {"cub": {"n_epochs": 100, "n_chkpts": 50}},
    )

    out = apply_dataset_specific_defaults(make_train_config_dummy(n_epochs=7, n_chkpts=None, dataset="cub"))
    assert (out["n_epochs"], out["n_chkpts"]) == (7, 50)
    out = apply_dataset_specific_defaults(make_train_config_dummy(n_epochs=None, n_chkpts=3, dataset="cub"))
    assert (out["n_epochs"], out["n_chkpts"]) == (100, 3)


def test_dataset_specific_defaults_leave_set_keys_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    # non-null n_epochs and n_chkpts: the dataset-specific defaults are not consulted (the yaml isn't even read)
    def _boom():
        raise AssertionError("dataset_specific.yaml must not be read when n_epochs and n_chkpts are set")
    monkeypatch.setattr("utils.config.load_dataset_specific_config_dict", _boom)

    out = apply_dataset_specific_defaults(make_train_config_dummy(n_epochs=7, n_chkpts=3))
    assert (out["n_epochs"], out["n_chkpts"]) == (7, 3)


def test_dataset_specific_defaults_use_passed_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    # a campaign trial passes the frozen snapshot; the live dataset_specific.yaml must not be read
    def _boom():
        raise AssertionError("dataset_specific.yaml must not be read when a snapshot is passed")
    monkeypatch.setattr("utils.config.load_dataset_specific_config_dict", _boom)

    snapshot = {"cub": {"n_epochs": 100, "n_chkpts": 50}}
    out = apply_dataset_specific_defaults(make_train_config_dummy(n_epochs=None, n_chkpts=None, dataset="cub"), snapshot)

    assert (out["n_epochs"], out["n_chkpts"]) == (100, 50)


# cub D10 train split has 4_935 samples (the dummy's dataset/split)
def test_train_config_chain_floor_null_disables_chaining(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    cfg = TrainConfig(**make_train_config_dummy())

    assert cfg.chain_perms is None
    assert cfg.samps_per_pass == 4_928  # 616 batches of 8 (drop_last trims the 7-sample remainder)
    assert cfg.samps_per_epoch == 4_928  # no chaining: an epoch is one batch-truncated pass
    assert cfg.sample_volume == 4_928  # n_epochs 1 x samps_per_epoch
    assert cfg.epochs_per_pass == 1
    assert cfg.n_passes == 1


def test_train_config_truncated_volume_prevents_extra_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    # batch_size 32 doesn't divide the 4_935-sample train set: drop_last trims each pass to 4_928,
    # and dropped samples don't count toward sample_volume, so 5 epochs = exactly 5 passes
    patch_hw(monkeypatch)

    cfg = TrainConfig(**make_train_config_dummy(n_epochs=5, batch_size=32))

    assert cfg.samps_per_pass == 4_928  # 154 batches of 32
    assert cfg.samps_per_epoch == 4_928
    assert cfg.sample_volume == 24_640  # 5 x 4_928
    assert cfg.n_passes == 5  # no bleed into a 6th pass


def test_train_config_chain_floor_below_train_set_disables_chaining(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    cfg = TrainConfig(**make_train_config_dummy(chain_floor=1_000))

    assert cfg.chain_perms is None


def test_train_config_chain_floor_chains_permutations(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    cfg = TrainConfig(**make_train_config_dummy(chain_floor=100_000, n_epochs=202))

    assert cfg.chain_perms == 21  # E_chain_nom = ceil(100_000 / 4_935)
    assert cfg.samps_per_pass == 103_632  # X_chain = 21 x 4_935 batch-aligned (the 3-sample tail carries into the next pass)
    assert cfg.epochs_per_pass == 21  # E_chain: the 3-sample carry (< 4_935) doesn't shift a permutation out of the window
    assert cfg.samps_per_epoch == 4_935  # chained: epochs stay nominal train-set permutations
    assert cfg.sample_volume == 996_870  # 202 x 4_935
    assert cfg.n_passes == 10  # ceil(996_870 / 103_632)


def test_train_config_chaining_credits_epochs_touched_by_truncated_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    # batch_size > train set size: batch alignment cuts more than a full permutation off the
    # nominal window (103_635 -> 98_304 consumed; the 5_331-sample tail > 4_935 leads the next
    # pass), so the pass credits only the permutations' worth it actually consumes:
    # ceil(98_304 / 4_935) = 20 < chain_perms 21
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
        TrainConfig(**make_train_config_dummy(chain_floor=5_000, batch_size=16_384))  # 2 perms = 9_870


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
    cfg_dict["loss1"] = {"crit": "infonce", "sim": "cos", "targ": "mp", "wting": {"focal": {"gamma": 0.0}},
                        "logits": {"scalar_lr_factor": 1.0, "temp": {"init": None}, "bce": {"center": None, "bias": {"init": None}}}}
    cfg_dict["hw"]["loss_chunk_size"] = 8  # ignored with InfoNCE: nulled out, no error

    cfg = TrainConfig(**cfg_dict)
    assert cfg.hw.loss_chunk_size is None


def test_train_config_bif_bce_keeps_chunking(monkeypatch: pytest.MonkeyPatch) -> None:
    # bif_bce is BCE-family: the tiled loss supports it, so the chunk size survives config
    patch_hw(monkeypatch)

    cfg_dict = make_train_config_dummy()  # batch_size 8
    cfg_dict["loss1"]["crit"] = "bif_bce"
    cfg_dict["hw"]["loss_chunk_size"] = 8

    cfg = TrainConfig(**cfg_dict)
    assert cfg.hw.loss_chunk_size == 8


def test_train_config_rejects_unknown_center(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    cfg_dict = make_train_config_dummy()
    cfg_dict["loss1"]["logits"]["bce"]["center"] = "grad_proje"
    with pytest.raises(ValueError, match="Unknown Loss 1 logits.bce.center"):
        TrainConfig(**cfg_dict)


def test_train_config_accepts_pos_prevalence_bias_init(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    cfg_dict = make_train_config_dummy()
    cfg_dict["loss1"]["logits"]["bce"]["bias"]["init"] = "pos_prevalence"
    cfg_dict["loss2"]["crit"] = "bif_bce"
    cfg_dict["loss2"]["logits"]["bce"]["bias"]["init"] = "pos_prevalence"

    cfg = TrainConfig(**cfg_dict)
    assert cfg.loss1["logits"]["bce"]["bias"]["init"] == "pos_prevalence"
    assert cfg.loss2["logits"]["bce"]["bias"]["init"] == "pos_prevalence"


def test_train_config_rejects_pos_prevalence_bias_init_with_infonce(monkeypatch: pytest.MonkeyPatch) -> None:
    # the prevalence is defined by the BCE-family criterion's weighting; the bias is inert under InfoNCE anyway
    patch_hw(monkeypatch)

    cfg_dict = make_train_config_dummy()
    cfg_dict["loss2"]["crit"] = "infonce"
    cfg_dict["loss2"]["logits"]["bce"]["bias"]["init"] = "pos_prevalence"
    with pytest.raises(ValueError, match="Loss 2 logits.bce.bias.init: pos_prevalence requires a BCE-family crit"):
        TrainConfig(**cfg_dict)


def test_train_config_rejects_unknown_bias_init(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    cfg_dict = make_train_config_dummy()
    cfg_dict["loss1"]["logits"]["bce"]["bias"]["init"] = "pos_prevalance"
    with pytest.raises(ValueError, match="Unknown Loss 1 logits.bce.bias.init"):
        TrainConfig(**cfg_dict)


def test_train_config_rejects_non_bool_unitless(monkeypatch: pytest.MonkeyPatch) -> None:
    # a string (e.g. a stale mode name) would otherwise be truthy and silently run unitless
    patch_hw(monkeypatch)

    cfg_dict = make_train_config_dummy()
    cfg_dict["loss"]["unitless"] = "unscaled"
    with pytest.raises(ValueError, match="loss.unitless must be a bool"):
        TrainConfig(**cfg_dict)


def test_train_config_rejects_sim_center_with_geo_under_chunking(monkeypatch: pytest.MonkeyPatch) -> None:
    # center: sim under the tiled loss needs the cos mean factorization for an exact global sim mean
    patch_hw(monkeypatch)

    cfg_dict = make_train_config_dummy()  # batch_size 8
    cfg_dict["loss1"] = {"crit": "bce", "sim": "geo1", "targ": "mp", "wting": {"focal": {"gamma": 0.0}},
                        "logits": {"scalar_lr_factor": 1.0, "temp": {"init": None}, "bce": {"center": "sim", "bias": {"init": None}}}}
    cfg_dict["hw"]["loss_chunk_size"] = 8

    with pytest.raises(ValueError, match="center: sim requires loss1.sim: cos"):
        TrainConfig(**cfg_dict)


def test_train_config_rejects_sim_center_with_geo_under_chunking_bif(monkeypatch: pytest.MonkeyPatch) -> None:
    # bif_bce configs now reach the chunking validations (chunking no longer nulled for them)
    patch_hw(monkeypatch)

    cfg_dict = make_train_config_dummy()  # batch_size 8
    cfg_dict["loss1"] = {"crit": "bif_bce", "sim": "geo1", "targ": "mp", "wting": {"focal": {"gamma": 0.0}},
                        "logits": {"scalar_lr_factor": 1.0, "temp": {"init": None}, "bce": {"center": "sim", "bias": {"init": None}}}}
    cfg_dict["hw"]["loss_chunk_size"] = 8

    with pytest.raises(ValueError, match="center: sim requires loss1.sim: cos"):
        TrainConfig(**cfg_dict)


def _make_stats_config_dummy(**overrides):
    config = {
        "spread_type": "std",
        "bold_high": True,
        "ordered": True,
        "heatmap": False,
        "supp_scores": {"primitive": False, "n_shot": False},
        "overrides": False,
    }
    config.update(overrides)
    return config


def test_stats_config_rejects_invalid_spread_type() -> None:
    with pytest.raises(ValueError, match="spread_type"):
        StatsConfig(**_make_stats_config_dummy(spread_type="var"))


def test_stats_config_rejects_unknown_supp_scores_keys() -> None:
    with pytest.raises(ValueError, match="supp_scores"):
        StatsConfig(**_make_stats_config_dummy(supp_scores={"primitive": False, "nshot": False}))


def _make_campaign_config_dummy(**overrides):
    config = {
        "n_trials_screen": 1,
        "n_trials_qual": 5,
        "trainval": False,
        "datasets": ["cub"],
        "ablation_arms": [[{"loss1.targ": "sp", "name": "sp"}]],
        "hpo_coords": [[{"name": "base"}]],
        "suffix": None,
    }
    config.update(overrides)
    return config


def test_campaign_config_rejects_screen_exceeding_qual() -> None:
    # the qual phase tops each pick up FROM its screening trials TO n_trials_qual, so it can't be fewer
    with pytest.raises(ValueError, match="n_trials_screen"):
        CampaignConfig(**_make_campaign_config_dummy(n_trials_screen=3, n_trials_qual=2))


def test_campaign_config_accepts_equal_or_null_qual() -> None:
    assert CampaignConfig(**_make_campaign_config_dummy(n_trials_screen=3, n_trials_qual=3)).n_trials_qual == 3
    assert CampaignConfig(**_make_campaign_config_dummy(n_trials_screen=3, n_trials_qual=None)).n_trials_qual is None


def test_campaign_config_trainval_requires_qual() -> None:
    # the trainval phase trains the qual picks up to their qual-selected checkpoints: nothing to train without qual
    with pytest.raises(ValueError, match="trainval"):
        CampaignConfig(**_make_campaign_config_dummy(trainval=True, n_trials_qual=None))
    assert CampaignConfig(**_make_campaign_config_dummy(trainval=True, n_trials_qual=5)).trainval is True


def test_train_config_rejects_chkpt_stop_out_of_range(monkeypatch: pytest.MonkeyPatch) -> None:
    # chkpt_stop is a checkpoint index, 1..n_chkpts (the dummy's n_chkpts is 10); null runs to sample_volume
    patch_hw(monkeypatch)

    for chkpt_stop in (0, 11):
        with pytest.raises(ValueError, match="chkpt_stop"):
            TrainConfig(**make_train_config_dummy(chkpt_stop=chkpt_stop))
    assert TrainConfig(**make_train_config_dummy(chkpt_stop=10)).chkpt_stop == 10
    assert TrainConfig(**make_train_config_dummy()).chkpt_stop is None
