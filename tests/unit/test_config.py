import pytest

from utils.config import CampaignConfig, GenSplitConfig, ManifoldVizConfig, StatsConfig, TrainConfig
from utils.config import apply_overrides
from utils.config import get_config_train, inert_params, _check_overrides_live
from utils.config import apply_model_specific_defaults
from utils.config import apply_dataset_specific_defaults


def _loss_cfg(lambda_=0.0, **overrides):
    """The dummy's minimal `loss` block (train.yaml's loss schema, the sections __post_init__ reads)."""
    cfg = {
        "crit": "bce", "sim": "cos", "blend": {"lambda": lambda_, "type": "targ"}, "unitless": False,
        "wting": {"focal": {"gamma": 0.0}},
        "logits": {"shared": True, "scalar_lr_factor": 1.0, "scale": {"init": None}, "bce": {"center": None, "bias": {"init": None}}},
    }
    cfg.update(overrides)
    return cfg


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
        "dev": False,
        "diagnostics": {"logging": False, "plot_every": "trial"},
        "kill_thresh": None,
        "del_base_eval_cache": None,
        "arch": {"model_type": "clip_vitb16", "clip": {"non_causal": False}, "siglip": {"vis_proj_head": None}},
        "dropout": {"patch_dropout": 0.0, "siglip": {"proj_head": 0.0, "stoch_depth": None}},
        "loss": _loss_cfg(),
        "lr": {"init": 1.0e-5, "decay_factor": 1.0e-3, "warmup": 0.02},
        "opt": {
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
    # loss1 / loss2 are blocks of `loss` now, but stay kwargs here so callers read unchanged
    targs = {k: overrides.pop(k) for k in ("loss1", "loss2") if k in overrides}
    config.update(overrides)
    config["loss"] = {**config["loss"], "loss1": {"targ": "sp"}, "loss2": {"targ": "sp"}, **targs}
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


def test_train_config_accepts_freezing_both_encoders(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    cfg = TrainConfig(**make_train_config_dummy(freeze={"text": True, "image": True}))
    assert cfg.freeze == {"text": True, "image": True}


def test_train_config_rejects_invalid_secondary_lambda(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    with pytest.raises(ValueError, match="loss.blend.lambda out of bounds"):
        TrainConfig(**make_train_config_dummy(loss=_loss_cfg(lambda_=1.5)))


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


def make_manifold_viz_config_dummy(**overrides):
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


def test_manifold_viz_config_rejects_negative_n_seeds() -> None:
    with pytest.raises(ValueError, match="n_seeds must be >= 0"):
        ManifoldVizConfig(**make_manifold_viz_config_dummy(n_seeds=-1))


def test_manifold_viz_config_rejects_negative_n_seeds_offset() -> None:
    with pytest.raises(ValueError, match="n_seeds_offset must be >= 0"):
        ManifoldVizConfig(**make_manifold_viz_config_dummy(n_seeds_offset=-1))


def test_manifold_viz_config_rejects_nonpositive_pooled_budget() -> None:
    with pytest.raises(ValueError, match="pooled.budget must be > 0"):
        ManifoldVizConfig(**make_manifold_viz_config_dummy(
            pooled={"enabled": True, "budget": 0.0, "pca_bounds": None}))


def test_manifold_viz_config_rejects_invalid_pca_bounds() -> None:
    with pytest.raises(ValueError, match="pooled.pca_bounds must be null or 'final'"):
        ManifoldVizConfig(**make_manifold_viz_config_dummy(
            pooled={"enabled": True, "budget": 1.0, "pca_bounds": "first"}))


def test_manifold_viz_config_rejects_too_few_umap_neighbors() -> None:
    with pytest.raises(ValueError, match="umap.n_neighbors must be >= 2"):
        ManifoldVizConfig(**make_manifold_viz_config_dummy(umap={"n_neighbors": 1, "min_dist": 0.1, "n_iter": None, "n_iter_sphere": 100}))


def test_manifold_viz_config_rejects_out_of_range_umap_min_dist() -> None:
    with pytest.raises(ValueError, match=r"umap.min_dist must be in \[0.0, 1.0\)"):
        ManifoldVizConfig(**make_manifold_viz_config_dummy(umap={"n_neighbors": 15, "min_dist": 1.0, "n_iter": None, "n_iter_sphere": 100}))


def test_manifold_viz_config_rejects_out_of_range_ema_tau() -> None:
    with pytest.raises(ValueError, match=r"orient.ema_tau must be in \(0.0, 1.0\]"):
        ManifoldVizConfig(**make_manifold_viz_config_dummy(orient={"ema_tau": 0.0}))


def test_train_config_rejects_yaml_string_scientific_notation(monkeypatch: pytest.MonkeyPatch) -> None:
    # YAML parses "1e-6" (no decimal point in the mantissa) as a STRING; a swept lr.init like
    # that must fail at config time, not deep inside AdamW
    patch_hw(monkeypatch)

    with pytest.raises(ValueError, match="lr.init must be numeric"):
        TrainConfig(**make_train_config_dummy(lr={"init": "1e-6", "decay_factor": 1.0e-3, "warmup": 0.02}))


def test_train_config_rejects_warmup_out_of_range(monkeypatch: pytest.MonkeyPatch) -> None:
    # lr.warmup is a fraction of sample_volume -- a stale absolute sample count must fail loudly
    patch_hw(monkeypatch)

    with pytest.raises(ValueError, match="lr.warmup must be a fraction of sample_volume"):
        TrainConfig(**make_train_config_dummy(lr={"init": 1.0e-5, "decay_factor": 1.0e-3, "warmup": 200_000}))


def test_train_config_rejects_unknown_plot_every(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    cfg_dict = make_train_config_dummy()
    cfg_dict["diagnostics"]["plot_every"] = "epoch"
    with pytest.raises(ValueError, match="diagnostics.plot_every must be 'trial' or 'chkpt'"):
        TrainConfig(**cfg_dict)


def test_train_config_rejects_unknown_del_base_eval_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    cfg_dict = make_train_config_dummy()
    cfg_dict["del_base_eval_cache"] = "always"
    with pytest.raises(ValueError, match="del_base_eval_cache must be null, 'campaign' or 'trial'"):
        TrainConfig(**cfg_dict)


def test_train_config_rejects_kill_thresh_out_of_range(monkeypatch: pytest.MonkeyPatch) -> None:
    # kill_thresh is a fraction of the run strictly inside (0, 1); null turns the kill check off
    patch_hw(monkeypatch)

    for kill_thresh in (0.0, 1.0, -0.1, 1.5):
        cfg_dict = make_train_config_dummy()
        cfg_dict["kill_thresh"] = kill_thresh
        with pytest.raises(ValueError, match="kill_thresh must be null or a fraction in"):
            TrainConfig(**cfg_dict)
    cfg_dict = make_train_config_dummy()
    cfg_dict["kill_thresh"] = 0.25
    assert TrainConfig(**cfg_dict).kill_thresh == 0.25


def test_train_config_rejects_htarg_shuffle_without_phylo_target(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    with pytest.raises(ValueError, match="requires a live phylo target"):
        TrainConfig(**make_train_config_dummy(htarg={"kernel": "laplace", "exp": {"beta": 1.0}, "shuffle": True}))


def test_train_config_accepts_htarg_shuffle_with_secondary_phylo(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    cfg = TrainConfig(**make_train_config_dummy(
        htarg={"kernel": "laplace", "exp": {"beta": 1.0}, "shuffle": True},
        loss=_loss_cfg(lambda_=0.3),
        loss2={"targ": "phylo"},
    ))

    assert cfg.htarg["shuffle"] is True


def test_train_config_rejects_htarg_shuffle_with_null_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    with pytest.raises(ValueError, match="requires a non-null seed"):
        TrainConfig(**make_train_config_dummy(
            htarg={"kernel": "laplace", "exp": {"beta": 1.0}, "shuffle": True},
            seed=None,
            loss1={"targ": "phylo"},
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


def test_model_specific_defaults_resolve_siglip_nulls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "utils.config.load_model_specific_config_dict",
        lambda: {
            "siglip": {"opt.wd": 0.0, "opt.beta2": 0.95},
            "clip": {"opt.wd": 0.2, "opt.beta2": 0.98},
        },
    )

    cfg_in = make_train_config_dummy(
        arch={"model_type": "siglip_vitb16", "clip": {"non_causal": False}},
        opt={"wd": None, "beta1": 0.9, "beta2": None, "eps": 1.0e-6},
    )

    out = apply_model_specific_defaults(cfg_in)

    assert out["opt"]["wd"] == 0.0
    assert out["opt"]["beta2"] == 0.95


def test_model_specific_defaults_preserve_explicit_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "utils.config.load_model_specific_config_dict",
        lambda: {
            "siglip": {"opt.wd": 0.0, "opt.beta2": 0.95},
            "clip": {"opt.wd": 0.2, "opt.beta2": 0.98},
        },
    )

    cfg_in = make_train_config_dummy(
        arch={"model_type": "clip_vitb16", "clip": {"non_causal": False}},
        opt={"wd": 0.11, "beta1": 0.9, "beta2": 0.77, "eps": 1.0e-6},
    )

    out = apply_model_specific_defaults(cfg_in)

    assert out["opt"]["wd"] == 0.11
    assert out["opt"]["beta2"] == 0.77


def test_model_specific_defaults_resolve_partial_null(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "utils.config.load_model_specific_config_dict",
        lambda: {
            "siglip": {"opt.wd": 0.0, "opt.beta2": 0.95},
            "clip": {"opt.wd": 0.2, "opt.beta2": 0.98},
        },
    )

    cfg_in = make_train_config_dummy(
        arch={"model_type": "clip_vitb16", "clip": {"non_causal": False}},
        opt={"wd": None, "beta1": 0.9, "beta2": 0.7, "eps": 1.0e-6},
    )

    out = apply_model_specific_defaults(cfg_in)

    assert out["opt"]["wd"] == 0.2
    assert out["opt"]["beta2"] == 0.7


def test_model_specific_defaults_unknown_model_type_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "utils.config.load_model_specific_config_dict",
        lambda: {
            "siglip": {"opt.wd": 0.0, "opt.beta2": 0.95},
            "clip": {"opt.wd": 0.2, "opt.beta2": 0.98},
        },
    )

    cfg_in = make_train_config_dummy(
        arch={"model_type": "mystery_model", "clip": {"non_causal": False}},
        opt={"wd": None, "beta1": 0.9, "beta2": None, "eps": 1.0e-6},
    )

    with pytest.raises(ValueError, match="Could not resolve model family"):
        apply_model_specific_defaults(cfg_in)


def test_model_specific_defaults_use_passed_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    # a campaign trial passes the frozen snapshot; the live model_specific.yaml must not be read
    def _boom():
        raise AssertionError("model_specific.yaml must not be read when a snapshot is passed")
    monkeypatch.setattr("utils.config.load_model_specific_config_dict", _boom)

    cfg_in = make_train_config_dummy(
        arch={"model_type": "clip_vitb16", "clip": {"non_causal": False}},
        opt={"wd": None, "beta1": 0.9, "beta2": None, "eps": 1.0e-6},
    )

    snapshot = {"siglip": {"opt.wd": 0.0, "opt.beta2": 0.95}, "clip": {"opt.wd": 0.2, "opt.beta2": 0.98}}
    out = apply_model_specific_defaults(cfg_in, snapshot)

    assert out["opt"]["wd"] == 0.2
    assert out["opt"]["beta2"] == 0.98


def test_model_specific_defaults_fill_any_dot_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # the defaults file keys are dot-paths into the config, so a family can default anything declared
    # there -- loss.crit among them (clip -> infonce, siglip -> bce)
    monkeypatch.setattr(
        "utils.config.load_model_specific_config_dict",
        lambda: {"siglip": {"loss.crit": "bce"}, "clip": {"loss.crit": "infonce"}},
    )
    for model_type, crit in (("clip_vitb16", "infonce"), ("siglip_vitb16", "bce")):
        cfg_in = make_train_config_dummy(arch={"model_type": model_type, "clip": {"non_causal": False}})
        cfg_in["loss"]["crit"] = None
        assert apply_model_specific_defaults(cfg_in)["loss"]["crit"] == crit
    # a crit set in the config proper wins over the family default
    cfg_in = make_train_config_dummy(arch={"model_type": "clip_vitb16", "clip": {"non_causal": False}})
    cfg_in["loss"]["crit"] = "bif_bce"
    assert apply_model_specific_defaults(cfg_in)["loss"]["crit"] == "bif_bce"
    # a stale key in the defaults file fails loudly rather than landing in a field nothing reads
    monkeypatch.setattr("utils.config.load_model_specific_config_dict",
                        lambda: {"clip": {"loss.nonesuch": 1}, "siglip": {}})
    with pytest.raises(ValueError, match="Unknown config key 'loss.nonesuch'"):
        apply_model_specific_defaults(make_train_config_dummy(
            arch={"model_type": "clip_vitb16", "clip": {"non_causal": False}}))


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
    cfg_dict["loss"]["crit"] = "infonce"
    cfg_dict["loss"]["loss1"]["targ"] = "mp"
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
    with pytest.raises(ValueError, match="Unknown loss.logits.bce.center"):
        TrainConfig(**cfg_dict)


def test_train_config_accepts_pos_prevalence_bias_init(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    for crit in ("bce", "bif_bce"):
        cfg_dict = make_train_config_dummy()
        cfg_dict["loss"]["crit"] = crit
        cfg_dict["loss"]["logits"]["bce"]["bias"]["init"] = "pos_prevalence"

        cfg = TrainConfig(**cfg_dict)
        assert cfg.loss["logits"]["bce"]["bias"]["init"] == "pos_prevalence"


def test_train_config_rejects_pos_prevalence_bias_init_with_infonce(monkeypatch: pytest.MonkeyPatch) -> None:
    # the prevalence is defined by the BCE-family criterion's weighting; the bias is inert under InfoNCE anyway
    patch_hw(monkeypatch)

    cfg_dict = make_train_config_dummy()
    cfg_dict["loss"]["crit"] = "infonce"
    cfg_dict["loss"]["logits"]["bce"]["bias"]["init"] = "pos_prevalence"
    with pytest.raises(ValueError, match="loss.logits.bce.bias.init: pos_prevalence requires a BCE-family crit"):
        TrainConfig(**cfg_dict)


def test_train_config_rejects_unknown_bias_init(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    cfg_dict = make_train_config_dummy()
    cfg_dict["loss"]["logits"]["bce"]["bias"]["init"] = "pos_prevalance"
    with pytest.raises(ValueError, match="Unknown loss.logits.bce.bias.init"):
        TrainConfig(**cfg_dict)


def test_train_config_rejects_identical_target_distributions_under_a_live_blend(monkeypatch: pytest.MonkeyPatch) -> None:
    # a blend of two identical target distributions is that distribution: loss.blend.lambda would do nothing
    patch_hw(monkeypatch)

    with pytest.raises(ValueError, match="same target distribution"):
        TrainConfig(**make_train_config_dummy(loss=_loss_cfg(lambda_=0.3), loss1={"targ": "mp"}, loss2={"targ": "mp"}))
    # under InfoNCE the same targ type still blends two distributions when the specs' tsm differ
    cfg = TrainConfig(**make_train_config_dummy(
        loss=_loss_cfg(crit="infonce", lambda_=0.3),
        loss1={"targ": "mp", "infonce": {"tsm": {"type": "linear", "sm_scale": "pinned"}}},
        loss2={"targ": "mp", "infonce": {"tsm": {"type": "softmax", "sm_scale": "pinned"}}},
    ))
    assert cfg.loss["blend"]["lambda"] == 0.3
    with pytest.raises(ValueError, match="same target distribution"):
        TrainConfig(**make_train_config_dummy(
            loss=_loss_cfg(crit="infonce", lambda_=0.3),
            loss1={"targ": "mp", "infonce": {"tsm": {"type": "linear", "sm_scale": "pinned"}}},
            loss2={"targ": "mp", "infonce": {"tsm": {"type": "linear", "sm_scale": "pinned"}}},
        ))
    # the endpoints have one live target: nothing to blend, no error
    assert TrainConfig(**make_train_config_dummy(loss=_loss_cfg(lambda_=1.0), loss1={"targ": "mp"}, loss2={"targ": "mp"})).loss["blend"]["lambda"] == 1.0


def test_train_config_rejects_invalid_blend_type_and_unitless(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    loss = _loss_cfg()
    loss["blend"]["type"] = "grad"
    with pytest.raises(ValueError, match="Unknown loss.blend.type"):
        TrainConfig(**make_train_config_dummy(loss=loss))
    with pytest.raises(ValueError, match="loss.unitless must be a bool"):
        TrainConfig(**make_train_config_dummy(loss=_loss_cfg(unitless="yes")))
    loss = _loss_cfg()
    loss["logits"]["shared"] = "no"
    with pytest.raises(ValueError, match="loss.logits.shared must be a bool"):
        TrainConfig(**make_train_config_dummy(loss=loss))
    loss = _loss_cfg(unitless=True)
    loss["blend"]["type"] = "loss"
    cfg = TrainConfig(**make_train_config_dummy(loss=loss))
    assert (cfg.loss["blend"]["type"], cfg.loss["unitless"]) == ("loss", True)


def test_train_config_rejects_sim_center_with_geo_under_chunking(monkeypatch: pytest.MonkeyPatch) -> None:
    # center: sim under the tiled loss needs the cos mean factorization for an exact global sim mean
    patch_hw(monkeypatch)

    cfg_dict = make_train_config_dummy(loss=_loss_cfg(  # batch_size 8
        sim="geo1", logits={"shared": True, "scalar_lr_factor": 1.0, "scale": {"init": None}, "bce": {"center": "sim", "bias": {"init": None}}}))
    cfg_dict["hw"]["loss_chunk_size"] = 8

    with pytest.raises(ValueError, match="center: sim requires loss.sim: cos"):
        TrainConfig(**cfg_dict)


def test_train_config_rejects_sim_center_with_geo_under_chunking_bif(monkeypatch: pytest.MonkeyPatch) -> None:
    # bif_bce configs now reach the chunking validations (chunking no longer nulled for them)
    patch_hw(monkeypatch)

    cfg_dict = make_train_config_dummy(loss=_loss_cfg(  # batch_size 8
        crit="bif_bce", sim="geo1", logits={"shared": True, "scalar_lr_factor": 1.0, "scale": {"init": None}, "bce": {"center": "sim", "bias": {"init": None}}}))
    cfg_dict["hw"]["loss_chunk_size"] = 8

    with pytest.raises(ValueError, match="center: sim requires loss.sim: cos"):
        TrainConfig(**cfg_dict)


def _full_loss_cfg(crit="bce", cls_imb_type=None, lambda_=0.0, unitless=False):
    # the train.yaml loss schema in full: the dummy's minimal block lacks the sections the inert rules read
    return {
        "crit": crit, "sim": "cos", "blend": {"lambda": lambda_, "type": "targ"}, "unitless": unitless,
        "bce": {"targ_mass_neut": False},
        "wting": {
            "cls_imb": {"type": cls_imb_type, "inv_freq": {"gamma": 0.5}, "class_bal": {"beta": 0.9999}, "norm": False},
            "focal": {"gamma": 0.0},
            "bce": {"dsmr": False},
        },
        "logits": {"shared": True, "scalar_lr_factor": 1.0, "scale": {"init": None, "freeze": False, "clamp": False},
                   "bce": {"center": None, "bias": {"init": None, "freeze": False}}},
    }


def _full_targ_cfg(targ="mp", tsm_type="softmax"):
    return {"targ": targ, "infonce": {"tsm": {"type": tsm_type, "sm_scale": "pinned"}}}


def test_get_config_train_rejects_inert_override(monkeypatch: pytest.MonkeyPatch) -> None:
    # loss.loss1.targ mp under loss.blend.lambda 0.0 leaves no live phylo target, so an htarg.kernel override is never read:
    # refused even when it restates the baseline's own value -- the value is beside the point -- while the
    # live override alongside it (loss.loss1.targ) goes unmentioned
    patch_hw(monkeypatch)
    cfg_dict = make_train_config_dummy(loss=_full_loss_cfg(), loss1=_full_targ_cfg("mp"), loss2=_full_targ_cfg("phylo"))
    cfg_dict["_overrides"] = {"loss.loss1.targ": "mp", "htarg.kernel": cfg_dict["htarg"]["kernel"]}

    with pytest.raises(ValueError, match=r"inert override\(s\).*htarg\.kernel \(no live target is phylo\)") as excinfo:
        get_config_train(cfg_dict)
    assert "loss.loss1.targ" not in str(excinfo.value)


@pytest.mark.parametrize("live", [{"loss.loss1.targ": "phylo"}, {"loss.blend.lambda": 0.3, "loss.loss2.targ": "phylo"}])
def test_get_config_train_accepts_override_a_live_phylo_target_reads(monkeypatch: pytest.MonkeyPatch, live) -> None:
    patch_hw(monkeypatch)
    cfg_dict = make_train_config_dummy(loss=_full_loss_cfg(), loss1=_full_targ_cfg("mp"), loss2=_full_targ_cfg("phylo"))
    cfg_dict["_overrides"] = {**live, "htarg.kernel": "bm"}

    assert get_config_train(cfg_dict).htarg["kernel"] == "bm"


def test_get_config_train_inert_override_names_outermost_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    # loss.loss2.infonce.tsm.type is inert both through loss.crit (bce) and through loss.blend.lambda 0.0 (all of loss2):
    # the enclosing cause is the one reported; every inert key is listed
    patch_hw(monkeypatch)
    cfg_dict = make_train_config_dummy(loss=_full_loss_cfg(), loss1=_full_targ_cfg(), loss2=_full_targ_cfg())
    cfg_dict["_overrides"] = {"loss.loss2.infonce.tsm.type": "linear", "loss.loss2.targ": "phylo"}

    with pytest.raises(ValueError) as excinfo:
        get_config_train(cfg_dict)
    msg = str(excinfo.value)
    assert "loss.loss2.infonce.tsm.type (loss.blend.lambda is 0.0)" in msg
    assert "loss.loss2.targ (loss.blend.lambda is 0.0)" in msg


def test_inert_params_clip_lone_bce_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)
    cfg = TrainConfig(**make_train_config_dummy(loss=_full_loss_cfg(crit="bce"), loss1=_full_targ_cfg("mp"), loss2=_full_targ_cfg()))

    inert = inert_params(cfg)
    assert {
        "arch.siglip", "dropout.siglip", "htarg", "loss.loss2",
        "loss.loss1.infonce", "loss.bce", "loss.wting.cls_imb.inv_freq", "loss.wting.cls_imb.class_bal",
        "loss.wting.cls_imb.norm", "loss.logits.bce.bias.freeze",  # CLIP's fixed 0.0 bias buffer
    } <= inert.keys()
    assert inert["loss.bce"] == "loss.crit is bce"
    assert "loss.loss1" not in inert
    # live: the family's own block, the toggles themselves, the BCE-path logit params under a BCE crit
    _check_overrides_live(cfg, {
        "arch.clip.non_causal": True, "dropout.patch_dropout": 0.1, "loss.blend.lambda": 0.0, "loss.loss1.targ": "sp",
        "loss.wting.cls_imb.type": None, "loss.wting.focal.gamma": 0.0, "loss.wting.bce.dsmr": False,
        "loss.logits.bce.center": None, "loss.logits.bce.bias.init": None, "aug": "custom",
    })


def test_inert_params_siglip_infonce_blend(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)
    cfg = TrainConfig(**make_train_config_dummy(
        arch={"model_type": "siglip_vitb16", "clip": {"non_causal": False}, "siglip": {"vis_proj_head": None}},
        htarg={"kernel": "bm", "exp": {"beta": 1.0}, "shuffle": False},
        loss=_full_loss_cfg(crit="infonce", cls_imb_type="inv_freq", lambda_=0.3),
        loss1=_full_targ_cfg("phylo", tsm_type="softmax"),
        loss2=_full_targ_cfg("mp", tsm_type="linear"),
    ))

    inert = inert_params(cfg)
    assert {
        "arch.clip", "dropout.siglip.proj_head", "htarg.exp",
        "loss.bce", "loss.wting.bce", "loss.logits.bce", "loss.wting.cls_imb.class_bal", "loss.loss2.infonce.tsm.sm_scale",
    } <= inert.keys()
    assert inert["loss.loss2.infonce.tsm.sm_scale"] == "loss.loss2.infonce.tsm.type is linear"
    assert not {"loss1", "loss2", "loss.loss1.infonce", "loss.loss2.infonce", "loss.wting.cls_imb.norm", "htarg"} & inert.keys()
    _check_overrides_live(cfg, {
        "dropout.siglip.stoch_depth": 0.1, "htarg.kernel": "bm", "htarg.shuffle": False, "loss.blend.lambda": 0.5,
        "loss.loss1.infonce.tsm.sm_scale": "pinned1", "loss.wting.cls_imb.inv_freq.gamma": 1.0, "loss.wting.cls_imb.norm": True,
        "loss.loss2.targ": "mp", "loss.loss2.infonce.tsm.type": "softmax", "loss.wting.cls_imb.type": None,
    })


@pytest.mark.parametrize("targ, reason", [("sp", "every live target is sp (row mass already 1)"), ("mp", None)])
def test_inert_params_targ_mass_neut_under_bif_bce(monkeypatch: pytest.MonkeyPatch, targ, reason) -> None:
    patch_hw(monkeypatch)
    cfg = TrainConfig(**make_train_config_dummy(loss=_full_loss_cfg(crit="bif_bce"), loss1=_full_targ_cfg(targ), loss2=_full_targ_cfg()))

    assert inert_params(cfg).get("loss.bce") == reason


def test_inert_params_targ_mass_neut_live_when_any_live_target_has_mass(monkeypatch: pytest.MonkeyPatch) -> None:
    # sp + mp blended: the blend's rows carry mass > 1 wherever mp does, so neutralization reads
    patch_hw(monkeypatch)
    cfg = TrainConfig(**make_train_config_dummy(loss=_full_loss_cfg(crit="bif_bce", lambda_=0.3), loss1=_full_targ_cfg("sp"), loss2=_full_targ_cfg("mp")))

    assert "loss.bce" not in inert_params(cfg)


def test_inert_params_linear_tsm_makes_sm_scale_inert(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)
    cfg = TrainConfig(**make_train_config_dummy(loss=_full_loss_cfg(crit="infonce"), loss1=_full_targ_cfg(tsm_type="linear"), loss2=_full_targ_cfg()))

    inert = inert_params(cfg)
    assert inert["loss.loss1.infonce.tsm.sm_scale"] == "loss.loss1.infonce.tsm.type is linear"
    assert "loss.loss1.infonce" not in inert


def test_inert_params_blend_type_needs_two_targets_and_a_target_dependent_factor(monkeypatch: pytest.MonkeyPatch) -> None:
    # the loss is affine in the target: a loss blend differs from the target blend only through a loss factor
    # that reads the target -- unitless, focal, DSMR, targ_mass_neut -- and only with two live targets
    patch_hw(monkeypatch)
    targs = {"loss1": _full_targ_cfg("mp"), "loss2": _full_targ_cfg("phylo")}

    lone = TrainConfig(**make_train_config_dummy(loss=_full_loss_cfg(unitless=True), **targs))
    assert inert_params(lone)["loss.blend.type"] == "loss.blend.lambda is 0.0 (a lone target)"

    plain = TrainConfig(**make_train_config_dummy(loss=_full_loss_cfg(lambda_=0.3), **targs))
    assert inert_params(plain)["loss.blend.type"].startswith("no target-dependent loss factor is live")

    unit, focal, dsmr = _full_loss_cfg(lambda_=0.3, unitless=True), _full_loss_cfg(lambda_=0.3), _full_loss_cfg(lambda_=0.3)
    focal["wting"]["focal"]["gamma"] = 2.0
    dsmr["wting"]["bce"]["dsmr"] = True
    for loss in (unit, focal, dsmr):
        assert "loss.blend.type" not in inert_params(TrainConfig(**make_train_config_dummy(loss=loss, **targs)))
    # DSMR is BCE-only and targ_mass_neut bif_bce-only: neither counts where the criterion never reads it
    loss = _full_loss_cfg(crit="infonce", lambda_=0.3)
    loss["wting"]["bce"]["dsmr"] = loss["bce"]["targ_mass_neut"] = True
    assert "loss.blend.type" in inert_params(TrainConfig(**make_train_config_dummy(loss=loss, **targs)))
    loss = _full_loss_cfg(crit="bif_bce", lambda_=0.3)
    loss["bce"]["targ_mass_neut"] = True
    assert "loss.blend.type" not in inert_params(TrainConfig(**make_train_config_dummy(loss=loss, **targs)))


def test_inert_params_separate_logit_scalars_need_a_live_loss_blend(monkeypatch: pytest.MonkeyPatch) -> None:
    # loss.logits.shared separates the scalars of a loss blend's two terms: a target blend or a lone target is one
    # loss on one set of logits. Separate scalars in turn set the blend types apart with no target-dependent factor on
    patch_hw(monkeypatch)
    targs = {"loss1": _full_targ_cfg("mp"), "loss2": _full_targ_cfg("phylo")}

    def inert(lambda_, blend_type, shared):
        loss = _full_loss_cfg(lambda_=lambda_)
        loss["blend"]["type"], loss["logits"]["shared"] = blend_type, shared
        return inert_params(TrainConfig(**make_train_config_dummy(loss=loss, **targs)))

    assert inert(0.0, "loss", False)["loss.logits.shared"] == "loss.blend.lambda is 0.0 (a lone target)"
    assert inert(0.3, "targ", False)["loss.logits.shared"] == "loss.blend.type is targ (one loss on one set of logits)"
    live = inert(0.3, "loss", False)
    assert "loss.logits.shared" not in live and "loss.blend.type" not in live
    assert "loss.logits.shared" not in inert(0.3, "loss", True)  # live, though the blend type it needs is not:
    assert "loss.blend.type" in inert(0.3, "loss", True)         # shared scalars, no target-dependent factor


def test_inert_params_unitless_cancels_cls_imb_norm(monkeypatch: pytest.MonkeyPatch) -> None:
    # L / L.detach() cancels any per-batch scalar on a loss term, cls_imb.norm's weight-mean division included
    patch_hw(monkeypatch)
    cfg = TrainConfig(**make_train_config_dummy(
        loss=_full_loss_cfg(cls_imb_type="inv_freq", unitless=True), loss1=_full_targ_cfg("mp"), loss2=_full_targ_cfg()))

    assert inert_params(cfg)["loss.wting.cls_imb.norm"].startswith("loss.unitless is true")


def test_inert_params_loss1_inert_at_lambda_one(monkeypatch: pytest.MonkeyPatch) -> None:
    # lambda 1.0 leaves loss2's target alone: the primary spec is never read, and only its target counts for htarg
    patch_hw(monkeypatch)
    cfg = TrainConfig(**make_train_config_dummy(loss=_full_loss_cfg(lambda_=1.0), loss1=_full_targ_cfg("phylo"), loss2=_full_targ_cfg("mp")))

    inert = inert_params(cfg)
    assert inert["loss.loss1"] == "loss.blend.lambda is 1.0"
    assert inert["htarg"] == "no live target is phylo"
    assert "loss.loss2" not in inert


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
        "baseline_overrides": {},
        "ablation_arms": [[{"loss.loss1.targ": "sp", "name": "sp"}]],
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


def test_campaign_config_baseline_overrides_take_scalars_only() -> None:
    # baseline_overrides applies to every trial, so it has nothing to vary over: a dict value would be a
    # combo group and a list value a combo list
    for bad in ({"loss": {"crit": "bce"}}, {"batch_size": [512, 1_024]}):
        with pytest.raises(ValueError, match="baseline_overrides takes scalar values"):
            CampaignConfig(**_make_campaign_config_dummy(baseline_overrides=bad))
    with pytest.raises(ValueError, match="baseline_overrides must be a mapping"):
        CampaignConfig(**_make_campaign_config_dummy(baseline_overrides=[[{"batch_size": 1_024}]]))
    assert CampaignConfig(**_make_campaign_config_dummy(
        baseline_overrides={"batch_size": 1_024, "aug": "custom"})).baseline_overrides["aug"] == "custom"


def test_train_config_rejects_chkpt_stop_out_of_range(monkeypatch: pytest.MonkeyPatch) -> None:
    # chkpt_stop is a checkpoint index, 1..n_chkpts (the dummy's n_chkpts is 10); null runs to sample_volume
    patch_hw(monkeypatch)

    for chkpt_stop in (0, 11):
        with pytest.raises(ValueError, match="chkpt_stop"):
            TrainConfig(**make_train_config_dummy(chkpt_stop=chkpt_stop))
    assert TrainConfig(**make_train_config_dummy(chkpt_stop=10)).chkpt_stop == 10
    assert TrainConfig(**make_train_config_dummy()).chkpt_stop is None
