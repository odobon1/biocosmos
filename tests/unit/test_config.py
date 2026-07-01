import pytest

from utils.config import GenSplitConfig, TrainConfig
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
        "sample_volume": 1_000,
        "chkpt_every": 100,
        "batch_size": 8,
        "dv_batching": False,
        "dev": {"logging": False, "manifold_viz": {"n_trials": 1}},
        "arch": {"model_type": "clip_vitb16", "non_causal": False},
        "img_norm": "dataset",
        "loss": {"type": "bce", "sim": "cos", "targ": "aligned", "logits": {"scale_init": None, "bias_init": None}},
        "loss2": {"type": "bce", "sim": "cos", "targ": "aligned", "mix": 0.0, "logits": {"scale_init": None, "bias_init": None}},
        "opt": {
            "lr": {"decay_factor": 1.0e-3},
            "l2reg": 0.0,
            "beta1": 0.9,
            "beta2": 0.95,
            "eps": 1.0e-6,
        },
        "freeze": {"text": False, "image": True},
        "text_template": {"train": "train", "eval": "sci"},
        "hw": {
            "mixed_prec": True,
            "act_chkpt": False,
            "prefetch_factor": 4,
            "max_n_workers_gpu": None,
            "persistent_workers_train": True,
            "persistent_workers_eval": True,
            "chunk_size": {"map_img2img": 512, "map_cross_modal": 512},
        },
    }
    config.update(overrides)
    return config


def patch_hw(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "utils.config.compute_dataloader_workers_prefetch",
        lambda *args, **kwargs: (2, 2, {"n_gpus": 1, "n_cpus": 4, "ram": 32}),
    )


def test_train_config_rejects_freezing_both_encoders(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    with pytest.raises(ValueError, match="both set to frozen"):
        TrainConfig(**make_train_config_dummy(freeze={"text": True, "image": True}))


def test_train_config_rejects_invalid_secondary_mix(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    with pytest.raises(ValueError, match="Secondary loss mix out of bounds"):
        TrainConfig(**make_train_config_dummy(loss2={"type": "bce", "sim": "cos", "targ": "aligned", "mix": 1.5}))


def test_train_config_rejects_negative_viz_n_trials(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    with pytest.raises(ValueError, match="dev.manifold_viz.n_trials must be >= 0"):
        TrainConfig(**make_train_config_dummy(dev={"logging": False, "manifold_viz": {"n_trials": -1}}))


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
        "prefetch_factor": 8,
        "max_n_workers_gpu": 3,
        "persistent_workers_train": False,
        "persistent_workers_eval": False,
        "chunk_size": {"map_img2img": 1024, "map_cross_modal": 1024},
    }))

    assert cfg.hw.mixed_prec is False
    assert cfg.hw.act_chkpt is True
    assert cfg.hw.prefetch_factor == 8
    assert cfg.hw.max_n_workers_gpu == 3


def test_apply_overrides_dot_path_sets_single_nested_field() -> None:
    base = {
        "loss": {
            "targ": "aligned",
            "sim": "cos",
        }
    }
    overrides = {
        "loss.targ": "multipos",
    }

    out = apply_overrides(base, overrides)

    assert out["loss"]["targ"] == "multipos"
    assert out["loss"]["sim"] == "cos"


def test_apply_overrides_dot_path_navigates_nested_dict() -> None:
    base = {
        "loss": {
            "targ": "aligned",
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


def test_model_specific_opt_defaults_resolve_siglip_nulls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "utils.config.load_model_specific_config_dict",
        lambda: {
            "siglip": {"l2reg": 0.0, "beta2": 0.95},
            "clip": {"l2reg": 0.2, "beta2": 0.98},
        },
    )

    cfg_in = make_train_config_dummy(
        arch={"model_type": "siglip_vitb16", "non_causal": False},
        opt={"lr": {"decay_factor": 1.0e-3}, "l2reg": None, "beta1": 0.9, "beta2": None, "eps": 1.0e-6},
    )

    out = apply_model_specific_opt_defaults(cfg_in)

    assert out["opt"]["l2reg"] == 0.0
    assert out["opt"]["beta2"] == 0.95


def test_model_specific_opt_defaults_preserve_explicit_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "utils.config.load_model_specific_config_dict",
        lambda: {
            "siglip": {"l2reg": 0.0, "beta2": 0.95},
            "clip": {"l2reg": 0.2, "beta2": 0.98},
        },
    )

    cfg_in = make_train_config_dummy(
        arch={"model_type": "clip_vitb16", "non_causal": False},
        opt={"lr": {"decay_factor": 1.0e-3}, "l2reg": 0.11, "beta1": 0.9, "beta2": 0.77, "eps": 1.0e-6},
    )

    out = apply_model_specific_opt_defaults(cfg_in)

    assert out["opt"]["l2reg"] == 0.11
    assert out["opt"]["beta2"] == 0.77


def test_model_specific_opt_defaults_resolve_partial_null(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "utils.config.load_model_specific_config_dict",
        lambda: {
            "siglip": {"l2reg": 0.0, "beta2": 0.95},
            "clip": {"l2reg": 0.2, "beta2": 0.98},
        },
    )

    cfg_in = make_train_config_dummy(
        arch={"model_type": "clip_vitb16", "non_causal": False},
        opt={"lr": {"decay_factor": 1.0e-3}, "l2reg": None, "beta1": 0.9, "beta2": 0.7, "eps": 1.0e-6},
    )

    out = apply_model_specific_opt_defaults(cfg_in)

    assert out["opt"]["l2reg"] == 0.2
    assert out["opt"]["beta2"] == 0.7


def test_model_specific_opt_defaults_unknown_model_type_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "utils.config.load_model_specific_config_dict",
        lambda: {
            "siglip": {"l2reg": 0.0, "beta2": 0.95},
            "clip": {"l2reg": 0.2, "beta2": 0.98},
        },
    )

    cfg_in = make_train_config_dummy(
        arch={"model_type": "mystery_model", "non_causal": False},
        opt={"lr": {"decay_factor": 1.0e-3}, "l2reg": None, "beta1": 0.9, "beta2": None, "eps": 1.0e-6},
    )

    with pytest.raises(ValueError, match="Could not resolve model family"):
        apply_model_specific_opt_defaults(cfg_in)


def test_model_specific_opt_defaults_use_passed_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    # a campaign trial passes the frozen snapshot; the live model_specific.yaml must not be read
    def _boom():
        raise AssertionError("model_specific.yaml must not be read when a snapshot is passed")
    monkeypatch.setattr("utils.config.load_model_specific_config_dict", _boom)

    cfg_in = make_train_config_dummy(
        arch={"model_type": "clip_vitb16", "non_causal": False},
        opt={"lr": {"decay_factor": 1.0e-3}, "l2reg": None, "beta1": 0.9, "beta2": None, "eps": 1.0e-6},
    )

    snapshot = {"siglip": {"l2reg": 0.0, "beta2": 0.95}, "clip": {"l2reg": 0.2, "beta2": 0.98}}
    out = apply_model_specific_opt_defaults(cfg_in, snapshot)

    assert out["opt"]["l2reg"] == 0.2
    assert out["opt"]["beta2"] == 0.98
