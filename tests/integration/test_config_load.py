import pytest

from utils.config import get_config_eval, get_config_hardware, get_config_train, load_train_config_dict


@pytest.mark.integration
def test_get_config_hardware_loads_yaml() -> None:
    cfg = get_config_hardware()

    assert hasattr(cfg, "mixed_prec")
    assert hasattr(cfg, "act_chkpt")


@pytest.mark.integration
def test_get_config_train_loads_repo_configs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "utils.config.compute_dataloader_workers_prefetch",
        lambda *args, **kwargs: (2, 2, {"n_gpus": 1, "n_cpus": 4, "ram": 32}),
    )

    cfg_dict = load_train_config_dict()
    cfg_dict.update({"campaign": "test", "setting": "test", "seed": 42, "dataset": "bryo"})

    cfg = get_config_train(cfg_dict)

    assert cfg.hw
    assert cfg.opt["l2reg"] == 0.0
    assert cfg.opt["beta2"] == 0.95


@pytest.mark.integration
def test_get_config_train_resolves_clip_model_opt_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "utils.config.compute_dataloader_workers_prefetch",
        lambda *args, **kwargs: (2, 2, {"n_gpus": 1, "n_cpus": 4, "ram": 32}),
    )

    cfg_dict = load_train_config_dict()
    cfg_dict.update({"campaign": "test", "setting": "test", "seed": 42, "dataset": "bryo"})
    cfg_dict["arch"]["model_type"] = "clip_vitb16"
    cfg_dict["opt"]["l2reg"] = None
    cfg_dict["opt"]["beta2"] = None

    cfg = get_config_train(cfg_dict=cfg_dict)

    assert cfg.opt["l2reg"] == 0.2
    assert cfg.opt["beta2"] == 0.98


@pytest.mark.integration
def test_get_config_eval_loads_repo_configs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "utils.config.compute_dataloader_workers_prefetch",
        lambda *args, **kwargs: (2, 2, {"n_gpus": 1, "n_cpus": 4, "ram": 32}),
    )

    cfg = get_config_eval(verbose=False)

    assert cfg.hw
    assert str(cfg.device) == "cuda"