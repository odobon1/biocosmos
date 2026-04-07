import importlib
import sys
import types
import pytest  # type: ignore[import]
import torch  # type: ignore[import]


def import_train_module():
    models_stub = types.ModuleType("models")
    models_stub.VLMWrapper = object
    sys.modules["models"] = models_stub

    eval_stub = types.ModuleType("utils.eval")
    eval_stub.ValidationPipeline = object
    sys.modules["utils.eval"] = eval_stub

    train_stub = types.ModuleType("utils.train")
    train_stub.ArtifactManager = object
    train_stub.plot_metrics = lambda *args, **kwargs: None
    sys.modules["utils.train"] = train_stub

    ddp_stub = types.ModuleType("utils.ddp")
    ddp_stub.setup_ddp = lambda: None
    ddp_stub.cleanup_ddp = lambda: None
    sys.modules["utils.ddp"] = ddp_stub

    sys.modules.pop("train", None)
    return importlib.import_module("train")


def test_cosine_exponential_lr_respects_nominal_floor() -> None:
    train_mod = import_train_module()
    param = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([param], lr=1.0)

    sched = train_mod.CosineExponentialLR(
        optimizer,
        gamma=0.1,
        period=2,
        peak_ratio=2.0,
        lr_nom_min=0.2,
    )

    values = [sched._lr_lambda(epoch) for epoch in range(5)]

    assert values[0] == pytest.approx(1.0)
    assert min(values) >= 0.1
    assert values[-1] >= 0.1


def test_cosine_wr_exponential_lr_restarts_each_period() -> None:
    train_mod = import_train_module()
    param = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([param], lr=1.0)

    sched = train_mod.CosineWRExponentialLR(
        optimizer,
        gamma=1.0,
        period=3,
        peak_ratio=4.0,
        lr_nom_min=0.05,
    )

    assert sched._lr_lambda(0) == pytest.approx(1.0)
    assert sched._lr_lambda(3) == pytest.approx(1.0)
    assert sched._lr_lambda(1) < sched._lr_lambda(0)