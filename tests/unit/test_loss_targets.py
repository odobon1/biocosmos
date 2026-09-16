import importlib
import sys
import types

import torch


def import_loss_module():
    fake_phylo = types.ModuleType("utils.phylo")

    class DummyPhyloVCV:
        def __init__(self, dataset: str, split: str, train_pt: str, batch_size: int, kernel: str, beta: float,
                     shuffle: bool = False, seed: int | None = None) -> None:
            self.dataset = dataset

        def get_targs_batch(self, targ_data_b):
            size = len(targ_data_b)
            return torch.full((size, size), 0.25)

    fake_phylo.PhyloVCV = DummyPhyloVCV
    sys.modules["utils.phylo"] = fake_phylo
    sys.modules.pop("utils.loss", None)
    return importlib.import_module("utils.loss")


def test_compute_targs_iw_is_identity() -> None:
    loss_mod = import_loss_module()

    targs = loss_mod.compute_targs_iw(3)

    assert torch.equal(targs, torch.eye(3))


def test_compute_targs_sw_marks_matching_classes() -> None:
    loss_mod = import_loss_module()

    targs = loss_mod.compute_targs_sw(torch.tensor([0, 1, 0]))

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
        {"rank_encs": [10, 100]},
        {"rank_encs": [10, 200]},
        {"rank_encs": [20, 300]},
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


def test_compute_targs_tax_normalizes_by_tree_depth() -> None:
    loss_mod = import_loss_module()
    targ_data = [
        {"rank_encs": [1, 10, 100, 1000]},
        {"rank_encs": [1, 10, 100, 2000]},  # diverges only at the deepest rank -> rank_dist 1
        {"rank_encs": [2, 20, 200, 3000]},  # diverges at the root -> rank_dist R (== 4)
    ]

    targs = loss_mod.compute_targs_tax(targ_data)

    expected = torch.tensor(
        [
            [1.00, 0.75, 0.0],
            [0.75, 1.00, 0.0],
            [0.00, 0.00, 1.0],
        ]
    )
    assert torch.equal(targs, expected)


def test_compute_targs_phylo_delegates_to_phylo_matrix() -> None:
    loss_mod = import_loss_module()
    loss_mod.configure_phylo_targs(split="D10", train_pt="train", batch_size=4, kernel="laplace", beta=1.0, shuffle=False, seed=None)

    targs = loss_mod.compute_targs_phylo([{"cid": "a", "dataset": "cub"}, {"cid": "b", "dataset": "cub"}])

    assert torch.equal(targs, torch.full((2, 2), 0.25))


def _infonce(loss_mod, tsm_type, sm_scale, clamp=True):
    crit = loss_mod.InfoNCECriterion.__new__(loss_mod.InfoNCECriterion)  # bypass build_wting (no dataset needed)
    crit.cfg = {"infonce": {"tsm": {"type": tsm_type, "sm_scale": sm_scale}}, "logits": {"scale": {"clamp": clamp}}}
    return crit


def test_targ_dist_is_q_for_bce_and_the_tsm_row_transform_for_infonce() -> None:
    # the target distribution Y a criterion trains against: the BCE family scores each pair against
    # its own target, InfoNCE against Q row-normalized (linear tsm) or row-softmaxed (softmax tsm) --
    # the pinned scales at the (clamped) live logit scale, a numeric sm_scale as a constant
    loss_mod = import_loss_module()
    Q = torch.tensor([[1.0, 0.5, 0.0], [0.5, 1.0, 0.25], [0.0, 0.25, 1.0]])
    logit_scale = torch.tensor(5.0)  # exp(5) > 100, so the clamped pinned scale reads 100

    bce = loss_mod.BCECriterion.__new__(loss_mod.BCECriterion)
    assert bce.targ_dist(Q, logit_scale) is Q

    Y = _infonce(loss_mod, "linear", "pinned").targ_dist(Q, logit_scale)
    assert torch.allclose(Y, Q / Q.sum(dim=1, keepdim=True))  # rows sum to 1, zeros stay zero
    Y = _infonce(loss_mod, "softmax", "pinned").targ_dist(Q, logit_scale)
    assert torch.allclose(Y, torch.softmax(2 * Q * 100.0, dim=1))
    Y = _infonce(loss_mod, "softmax", "pinned1", clamp=False).targ_dist(Q, logit_scale)
    assert torch.allclose(Y, torch.softmax(Q * torch.exp(logit_scale), dim=1))
    Y = _infonce(loss_mod, "softmax", 3.0).targ_dist(Q, logit_scale)
    assert torch.allclose(Y, torch.softmax(2 * Q * 3.0, dim=1))
