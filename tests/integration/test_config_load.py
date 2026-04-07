import pytest  # type: ignore[import]

from utils.config import get_config_eval, get_config_hardware, get_config_train


@pytest.mark.integration
def test_get_config_hardware_loads_yaml() -> None:
    cfg = get_config_hardware()

    assert hasattr(cfg, "cached_imgs")
    assert hasattr(cfg, "mixed_prec")
    assert hasattr(cfg, "act_chkpt")


@pytest.mark.integration
def test_get_config_train_loads_repo_configs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "utils.config.compute_dataloader_workers_prefetch",
        lambda: (2, 2, {"n_gpus": 1, "n_cpus": 4, "ram": 32}),
    )

    cfg = get_config_train()

    assert cfg.loss["cfg"]
    assert cfg.lr_sched_params
    assert cfg.hw


@pytest.mark.integration
def test_get_config_eval_loads_repo_configs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "utils.config.compute_dataloader_workers_prefetch",
        lambda: (2, 2, {"n_gpus": 1, "n_cpus": 4, "ram": 32}),
    )

    cfg = get_config_eval(verbose=False)

    assert cfg.loss["cfg"]
    assert cfg.hw
    assert str(cfg.device) == "cuda"