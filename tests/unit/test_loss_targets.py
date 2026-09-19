import importlib
import math
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


def _spec(tsm_type, sm_scale):
    return {"targ": "phylo", "infonce": {"tsm": {"type": tsm_type, "sm_scale": sm_scale}}}


def _infonce(loss_mod, specs, clamp=True):
    """An InfoNCE criterion over explicit (weight, target spec) pairs (Criterion.targ_specs)."""
    crit = loss_mod.InfoNCECriterion.__new__(loss_mod.InfoNCECriterion)  # bypass build_wting (no dataset needed)
    crit.cfg = {"logits": {"scale": {"clamp": clamp}}}
    crit.targ_specs = specs
    return crit


def test_targ_dist_is_q_for_bce_and_the_tsm_row_transform_for_infonce() -> None:
    # the target distribution Y a criterion trains against: the BCE family scores each pair against
    # its own target, InfoNCE against Q row-normalized (linear tsm) or row-softmaxed (softmax tsm) --
    # the pinned scales at the (clamped) live logit scale, a numeric sm_scale as a constant
    loss_mod = import_loss_module()
    Q = torch.tensor([[1.0, 0.5, 0.0], [0.5, 1.0, 0.25], [0.0, 0.25, 1.0]])
    logit_scale = torch.tensor(5.0)  # exp(5) > 100, so the clamped pinned scale reads 100

    bce = loss_mod.BCECriterion.__new__(loss_mod.BCECriterion)
    bce.targ_specs = [(1.0, {"targ": "phylo"})]
    assert torch.equal(bce.targ_dist([Q], [logit_scale]), Q)

    Qd = Q.double()  # InfoNCE's tsm solves in float64 (InfoNCECriterion._tsm), so Y comes back there
    Y = _infonce(loss_mod, [(1.0, _spec("linear", "pinned"))]).targ_dist([Q], [logit_scale])
    assert torch.allclose(Y, Qd / Qd.sum(dim=1, keepdim=True))  # rows sum to 1, zeros stay zero
    Y = _infonce(loss_mod, [(1.0, _spec("softmax", "pinned"))]).targ_dist([Q], [logit_scale])
    assert torch.allclose(Y, torch.softmax(2 * Qd * 100.0, dim=1))
    Y = _infonce(loss_mod, [(1.0, _spec("softmax", "pinned1"))], clamp=False).targ_dist([Q], [logit_scale])
    assert torch.allclose(Y, torch.softmax(Qd * torch.exp(logit_scale), dim=1))
    Y = _infonce(loss_mod, [(1.0, _spec("softmax", 3.0))]).targ_dist([Q], [logit_scale])
    assert torch.allclose(Y, torch.softmax(2 * Qd * 3.0, dim=1))


def test_softmax_tsm_keeps_full_support_at_the_base_model_scales() -> None:
    # the softmax tsm's negatives reach exp(-2 alpha), which flushes to exact zero in float32 from
    # alpha ~ 50.6 -- past the scales a run starts from under scale.init null (clip_vitb16 ~100,
    # siglip_vitb16 ~117.3). Solving in float64 keeps them, which is what holds the row-wise
    # target-implied scale bound finite: alpha_req = 0.5 log(max Y / min Y) = alpha (max_j Q_ij -
    # min_j Q_ij), the pinned target being exactly reachable at the scale that built it
    loss_mod = import_loss_module()
    enc = torch.arange(6) // 2
    Q = (enc[:, None] == enc[None, :]).float()  # mp-like 0/1, so the log range is the full 2 alpha
    for alpha in (30.0, 100.0, 117.3):
        logit_scale = torch.tensor(math.log(alpha))
        Y = _infonce(loss_mod, [(1.0, _spec("softmax", "pinned"))], clamp=False).targ_dist([Q], [logit_scale])
        assert Y.dtype == torch.float64
        assert (Y > 0).all(), alpha  # float32 flushes every negative to exactly zero above ~50.6
        alpha_req = 0.5 * torch.log(Y.amax(dim=1) / Y.amin(dim=1))
        assert torch.isfinite(alpha_req).all(), alpha
        expected = torch.exp(logit_scale).double() * (Q.double().amax(dim=1) - Q.double().amin(dim=1))
        torch.testing.assert_close(alpha_req, expected)


def test_targ_dist_blends_each_spec_under_its_own_tsm() -> None:
    # a blend maps every spec's Q through ITS tsm first, then mixes the distributions: Y = sum_k w_k tsm_k(Q_k);
    # the blended membership matrix (targ_memb) mixes the raw Q's instead
    loss_mod = import_loss_module()
    Q1 = torch.tensor([[1.0, 0.5, 0.0], [0.5, 1.0, 0.25], [0.0, 0.25, 1.0]])
    Q2 = torch.eye(3)
    logit_scale = torch.tensor(1.0)
    specs = loss_mod.targ_specs(0.3, _spec("linear", "pinned"), _spec("softmax", 3.0))
    assert [w for w, _ in specs] == [0.7, 0.3]
    crit = _infonce(loss_mod, specs)
    Y = crit.targ_dist([Q1, Q2], [logit_scale] * 2)
    Q1d, Q2d = Q1.double(), Q2.double()  # the tsm's float64 (InfoNCECriterion._tsm); targ_memb keeps Q's dtype
    assert torch.allclose(Y, 0.7 * Q1d / Q1d.sum(dim=1, keepdim=True) + 0.3 * torch.softmax(2 * Q2d * 3.0, dim=1))
    assert torch.allclose(crit.targ_memb([Q1, Q2]), 0.7 * Q1 + 0.3 * Q2)
    # a lone spec is at full weight, so its blend is the matrix itself -- returned as-is, not copied
    assert _infonce(loss_mod, [(1.0, _spec("linear", "pinned"))]).targ_memb([Q1]) is Q1


def test_targ_specs_drop_zero_weight_specs() -> None:
    loss_mod = import_loss_module()
    s1, s2 = _spec("linear", "pinned"), _spec("softmax", "pinned")
    assert loss_mod.targ_specs(0.0, s1, s2) == [(1.0, s1)]
    assert loss_mod.targ_specs(1.0, s1, s2) == [(1.0, s2)]
    assert loss_mod.targ_specs(0.25, s1, s2) == [(0.75, s1), (0.25, s2)]
