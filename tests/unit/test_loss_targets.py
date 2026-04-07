import importlib
import sys
import types

import torch  # type: ignore[import]


def import_loss_module():
    fake_phylo = types.ModuleType("utils.phylo")

    class DummyPhyloVCV:
        def __init__(self, dataset: str) -> None:
            self.dataset = dataset

        def get_targs_batch(self, targ_data_b):
            size = len(targ_data_b)
            return torch.full((size, size), 0.25)

    fake_phylo.PhyloVCV = DummyPhyloVCV
    sys.modules["utils.phylo"] = fake_phylo
    sys.modules.pop("utils.loss", None)
    return importlib.import_module("utils.loss")


def test_compute_targs_aligned_is_identity() -> None:
    loss_mod = import_loss_module()

    targs = loss_mod.compute_targs_aligned(3)

    assert torch.equal(targs, torch.eye(3))


def test_compute_targs_multipos_marks_matching_classes() -> None:
    loss_mod = import_loss_module()

    targs = loss_mod.compute_targs_multipos(torch.tensor([0, 1, 0]))

    expected = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ]
    )
    assert torch.equal(targs, expected)


def test_compute_targs_tax_uses_rank_distances() -> None:
    loss_mod = import_loss_module()
    targ_data = [
        {"rank_keys": [10, 100]},
        {"rank_keys": [10, 200]},
        {"rank_keys": [20, 300]},
    ]

    targs = loss_mod.compute_targs_tax(targ_data)

    expected = torch.tensor(
        [
            [1.0, 0.5, 0.0],
            [0.5, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert torch.equal(targs, expected)


def test_compute_targs_phylo_delegates_to_phylo_matrix() -> None:
    loss_mod = import_loss_module()

    targs = loss_mod.compute_targs_phylo([{"sid": "a"}, {"sid": "b"}])

    assert torch.equal(targs, torch.full((2, 2), 0.25))