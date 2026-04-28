import pytest  # type: ignore[import]

from utils.config import GenSplitConfig, TrainConfig


def make_train_config(**overrides):
    config = {
        "study_name": "study",
        "experiment_name": "exp",
        "seed": 7,
        "dataset": "nymph",
        "split_name": "split_a",
        "n_epochs": 3,
        "chkpt_every": 1,
        "batch_size": 8,
        "dv_batching": False,
        "dev": {},
        "arch": {"model_type": "clip_vitb16", "non_causal": False},
        "loss": {"type": "bce", "sim": "cos", "targ": "aligned"},
        "loss2": {"type": "bce", "sim": "cos", "targ": "aligned", "mix": 0.0},
        "opt": {"lr": {"sched": "cos"}},
        "freeze": {"text": False, "image": True},
        "text_template": {"train": "train", "valid": "bioclip_sci"},
        "logging": False,
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
        TrainConfig(**make_train_config(freeze={"text": True, "image": True}))


def test_train_config_rejects_unknown_scheduler(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    with pytest.raises(ValueError, match="Unknown LR scheduler type"):
        TrainConfig(**make_train_config(opt={"lr": {"sched": "bad_sched"}}))


def test_train_config_rejects_invalid_secondary_mix(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    with pytest.raises(ValueError, match="Secondary loss mix out of bounds"):
        TrainConfig(**make_train_config(loss2={"type": "bce", "sim": "cos", "targ": "aligned", "mix": 1.5}))


def test_train_config_populates_runtime_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_hw(monkeypatch)

    cfg = TrainConfig(**make_train_config())

    assert cfg.n_workers == 2
    assert cfg.prefetch_factor == 2
    assert cfg.n_gpus == 1
    assert cfg.rdpath_trial == "artifacts/study/exp/7"
    assert str(cfg.device) == "cuda"


def test_splits_config_accepts_optional_pos_filter() -> None:
    cfg = GenSplitConfig(
        seed=7,
        split_name="split_a",
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
        split_name="split_a",
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
            split_name="split_a",
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
            split_name="split_a",
            pct_partition=0.1,
            pct_ood_tol=0.01,
            size_dev=0,
            pos_filter=None,
            nst_names=["1-2", "3+"],
            nst_seps=[2],
        )