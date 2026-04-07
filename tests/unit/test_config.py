from __future__ import annotations

import pytest

from utils.config import TrainConfig


def make_train_config(**overrides):
    config = {
        "study_name": "study",
        "experiment_name": "exp",
        "seed": 7,
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
        lambda: (2, 2, {"n_gpus": 1, "n_cpus": 4, "ram": 32}),
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
