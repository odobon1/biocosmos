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
        "eval_every": 100,
        "batch_size": 8,
        "dv_batching": False,
        "dev": {"logging": False},
        "arch": {"model_type": "clip_vitb16", "non_causal": False},
        "img_norm": "dataset",
        "loss": {"type": "bce", "sim": "cos", "targ": "aligned"},
        "loss2": {"type": "bce", "sim": "cos", "targ": "aligned", "mix": 0.0},
        "opt": {
            "lr": {"decay_factor": 1.0e-3},
            "l2reg": 0.0,
            "beta1": 0.9,
            "beta2": 0.95,
            "eps": 1.0e-6,
        },
        "freeze": {"text": False, "image": True},
        "text_template": {"train": "train", "eval": "sci"}
    }
    config.update(overrides)
    return config


def patch_hw(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "utils.config.compute_dataloader_workers_prefetch",
        lambda *args, **kwargs: (2, 2, {"n_gpus": 1, "n_cpus": 4, "ram": 32}),
    )


def test_train_config_rejects_invalid_train_pt(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    with pytest.raises(ValueError, match="Unknown train_pt"):
        TrainConfig(**make_train_config_dummy(train_pt="invalid"))


def test_train_config_rejects_freezing_both_encoders(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    with pytest.raises(ValueError, match="both set to frozen"):
        TrainConfig(**make_train_config_dummy(freeze={"text": True, "image": True}))


def test_train_config_rejects_invalid_secondary_mix(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    with pytest.raises(ValueError, match="Secondary loss mix out of bounds"):
        TrainConfig(**make_train_config_dummy(loss2={"type": "bce", "sim": "cos", "targ": "aligned", "mix": 1.5}))


def test_train_config_populates_runtime_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    cfg = TrainConfig(**make_train_config_dummy())

    assert cfg.n_workers == 2
    assert cfg.prefetch_factor == 2
    assert cfg.n_gpus == 1
    assert str(cfg.device) == "cuda"


def test_splits_config_accepts_optional_pos_filter() -> None:
    cfg = GenSplitConfig(
        seed=7,
        split="split_a",
        pct_partition=0.1,
        pct_ood_tol=0.01,
        size_dev=128,
        pos_filter="dorsal",
        nst_names=["1-2", "3+"],
        nst_seps=[2],
    )

    assert cfg.pos_filter == "dorsal"
    assert cfg.size_dev == 128


def test_splits_config_accepts_optional_ood_family_name() -> None:
    cfg = GenSplitConfig(
        seed=7,
        split="split_a",
        pct_partition=0.1,
        pct_ood_tol=0.01,
        size_dev=128,
        pos_filter=None,
        ood_family_name="nymphalidae",
        nst_names=["1-2", "3+"],
        nst_seps=[2],
    )

    assert cfg.ood_family_name == "nymphalidae"


def test_splits_config_rejects_unknown_pos_filter() -> None:
    with pytest.raises(ValueError, match="Unknown pos_filter"):
        GenSplitConfig(
            seed=7,
            split="split_a",
            pct_partition=0.1,
            pct_ood_tol=0.01,
            size_dev=128,
            pos_filter="sideways",
            nst_names=["1-2", "3+"],
            nst_seps=[2],
        )


def test_splits_config_rejects_non_positive_size_dev() -> None:
    with pytest.raises(ValueError, match="size_dev must be greater than 0"):
        GenSplitConfig(
            seed=7,
            split="split_a",
            pct_partition=0.1,
            pct_ood_tol=0.01,
            size_dev=0,
            pos_filter=None,
            nst_names=["1-2", "3+"],
            nst_seps=[2],
        )


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
