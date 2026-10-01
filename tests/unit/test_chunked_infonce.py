"""
Equivalence tests for the tiled/chunked global-batch InfoNCE loss (hardware.loss_chunk_size.infonce).

chunked_infonce_loss_backward must reproduce the full-batch path (InfoNCECriterion.__call__ via
_global_batch_loss) up to floating-point summation order: the loss and raw loss, the gradients wrt the
image/text embeddings and the logit scale(s), sum(dL/dsim), loss.infonce.block_residuals' record
(crit.block_stats), and every batch stat the full-batch path reports -- sim_targ_batch_stats',
infonce_batch_stats' and sim_grad_entropies_actual's, the last two including the pair-level entropies
that fold the two anchor directions per cell and so come off the tiled path's fold pass. Across
InfoNCECriterion's config space: sp/mp/tax/phylo targets under the linear and softmax tsm, their blend
under either loss.blend.type, loss.unitless, separate logit scalars, class-imbalance + focal weighting,
block_residuals alpha/full over every residual path (hard and soft two-level closed forms, the active
sets, a target blend's spec-less rows), geo sims and the scale clamp.
"""
import math

import pytest
import torch

import utils.loss as L
from models import sim_targ_batch_stats
from utils.head import compute_sim


HIST_BINS = 20  # the chunked calls' reporting.learning_curves.hist_bins
KAPPAS = (0.0, 3.0, 10_000.0)  # incl. one far past where a column-streamed T2I margin's weights underflow


class _DummyPhyloVCV:
    """A graded class-pair target in place of the tree-derived matrix, shaped like its kernels: symmetric, 1.0 on
    same-class pairs, the rest spread over (0.0025, 1) -- wide enough that rows leave the reachable band at the
    tests' scales. Block builder agrees with the full matrix by construction."""
    def __init__(self):
        g = torch.Generator().manual_seed(5)
        U = torch.rand(64, 64, generator=g)
        self.targs = torch.exp(-6.0 * 0.5 * (U + U.T))
        self.targs.fill_diagonal_(1.0)

    @staticmethod
    def _idxs(targ_data_b):
        return torch.tensor([int(td["cid"][1:]) for td in targ_data_b])  # cid "c<class enc>" (_make_targ_data)

    def get_targs_batch(self, targ_data_b):
        idxs = self._idxs(targ_data_b)
        return self.targs[idxs][:, idxs]

    def make_targ_block_fn(self, targ_data_b, device):
        idxs = self._idxs(targ_data_b)
        return lambda rs, re: self.targs[idxs[rs:re]][:, idxs].to(device)


@pytest.fixture(autouse=True)
def _phylo_targets_without_a_tree(monkeypatch):
    monkeypatch.setattr(L, "get_phylo_vcv", lambda dataset: _DummyPhyloVCV())


def _make_targ_data(B, K, R, class_encs_b):
    """targ_data carrying every field any targ_type needs (rank_encs for tax, cid/dataset for phylo)."""
    g = torch.Generator().manual_seed(99)
    rank_encs = torch.randint(0, 3, (B, R), generator=g).tolist()
    return [{"rank_encs": rank_encs[i], "cid": f"c{int(class_encs_b[i])}", "dataset": "cub"} for i in range(B)]


def _cfg(lambda_=0.0, blend_type="targ", unitless=False, shared=True, clamp=False, block=None, focal_gamma=0.0,
         cls_imb=None, norm=False, sim="cos"):
    """The loss-level config (train.yaml's `loss` block, as InfoNCECriterion reads it)."""
    return {
        "crit": "infonce", "sim": sim, "blend": {"lambda": lambda_, "type": blend_type}, "unitless": unitless,
        "infonce": {"block_residuals": block},
        "wting": {
            "cls_imb": {"type": cls_imb, "inv_freq": {"gamma": 0.5}, "class_bal": {"beta": 0.9999}, "norm": norm},
            **({"focal": {"gamma": focal_gamma}} if focal_gamma > 0.0 else {}),  # config load prunes the block when gamma = 0.0
        },
        "logits": {"shared": shared, "scale": {"clamp": clamp}, "bce": {"center": None, "bias": {}}},
    }


def _spec(targ, tsm_type="linear", sm_scale="pinned"):
    """A target spec (train.yaml's loss1 / loss2 block)."""
    return {"targ": targ, "infonce": {"tsm": {"type": tsm_type, "sm_scale": sm_scale}}}


def _make_crit(cfg, spec1, spec2, K, B):
    crit = L.InfoNCECriterion.__new__(L.InfoNCECriterion)  # bypass build_wting (no dataset needed)
    crit.cfg = cfg
    crit.targ_specs = L.targ_specs(cfg["blend"]["lambda"], spec1, spec2)
    crit.device = torch.device("cpu")
    crit.batch_size = B
    g = torch.Generator().manual_seed(K)
    crit.counts = torch.randint(1, 1000, (K,), generator=g).to(torch.float64)
    crit.wt_mean = 1.0
    return crit


def _params(log_scale, freeze=False, dtype=torch.float32):
    # loss2's term's own scale (separate logit scalars) sits apart from the primary; unused -- no grad -- otherwise.
    # freeze: the log-scale parameters take no gradient (loss.logits.scale.freeze)
    return {"scale": torch.tensor(log_scale, dtype=dtype, requires_grad=not freeze),
            "scale2": torch.tensor(log_scale + 0.4, dtype=dtype, requires_grad=not freeze)}


def _compute_logits_fn(p):
    """Stub mirroring VLMWrapper.compute_logits under InfoNCE: no bias, no centering."""
    def compute_logits(sim, clamp, center=None, center_global=None, half_live=False, secondary=False):
        s = p["scale2"] if secondary else p["scale"]
        if clamp:
            s = s.clamp(max=math.log(100))
        return sim.float() * s.exp()  # the head runs in float32 whatever the sims came in as
    return compute_logits


def _logit_scale(crit, p):
    return (p["scale"], p["scale2"]) if crit.sep_scalars else p["scale"]


def _full_reference(crit, img, txt, class_encs_b, targ_data_b, p):
    """The full-batch loss, backpropagated, with everything the production path reads off it: (loss, loss_raw,
    the retained dL/dsim, batch stats as VLMWrapper._batch_stats + TrainPipeline._step_train assemble them)."""
    clogits = _compute_logits_fn(p)
    clamp = crit.cfg["logits"]["scale"]["clamp"]
    sim = compute_sim(img, txt, crit.cfg["sim"])
    sim.retain_grad()
    logits = [clogits(sim, clamp, secondary=sec) for sec in ((False, True) if crit.sep_scalars else (False,))]
    loss, loss_raw, Q, y = crit(logits if crit.sep_scalars else logits[0], class_encs_b, targ_data_b, True, _logit_scale(crit, p), sim)
    stats = sim_targ_batch_stats(sim, Q, KAPPAS, HIST_BINS)
    stats.update(L.infonce_batch_stats(sim, Q, y, logits[0], p["scale"].detach(), clamp))
    if crit.lambda_eff is not None:
        stats["lambda_eff"] = crit.lambda_eff.item()
    loss.backward()
    stats.update(L.sim_grad_entropies_actual(sim.grad))
    return loss.detach(), loss_raw.detach(), sim.grad.double(), stats


def _assert_stat_close(key, a, b, rel=2e-4, abs_=1e-6):
    if isinstance(b, list):
        assert len(a) == len(b), key
        for a_i, b_i in zip(a, b):
            _assert_stat_close(key, a_i, b_i, rel, abs_)
    elif math.isnan(b) or math.isinf(b):
        assert (math.isnan(a) and math.isnan(b)) or a == b, f"{key}: chunked {a} != full {b}"
    else:
        assert a == pytest.approx(b, rel=rel, abs=abs_), f"{key}: chunked {a} != full {b}"


def _batch(seed=0):
    """One global batch: (B, K, normalized image / text embeddings, class encodings, targ_data)."""
    B, K, D, R = 48, 8, 16, 4
    g = torch.Generator().manual_seed(seed)
    img0 = torch.nn.functional.normalize(torch.randn(B, D, generator=g), dim=1)
    txt0 = torch.nn.functional.normalize(torch.randn(B, D, generator=g), dim=1)
    class_encs_b = torch.randint(0, K, (B,), generator=g)
    return B, K, img0, txt0, class_encs_b, _make_targ_data(B, K, R, class_encs_b)


def _run(cfg, spec1, spec2, C, log_scale=0.5, seed=0, freeze_scale=False):
    """Full-batch reference vs the chunked path on one batch; returns both criteria's records for the caller."""
    device = torch.device("cpu")
    B, K, img0, txt0, class_encs_b, targ_data_b = _batch(seed)

    crit = _make_crit(cfg, spec1, spec2, K, B)
    img, txt, p = img0.clone().requires_grad_(True), txt0.clone().requires_grad_(True), _params(log_scale, freeze_scale)
    loss_ref, loss_raw_ref, G_ref, stats_ref = _full_reference(crit, img, txt, class_encs_b, targ_data_b, p)
    gsum_ref, gabs_ref = G_ref.sum().item(), G_ref.abs().sum().item()
    block_ref, correction_ref = crit.block_stats, crit.dlogscale_correction

    crit_c = _make_crit(cfg, spec1, spec2, K, B)
    imgc, txtc, pc = img0.clone().requires_grad_(True), txt0.clone().requires_grad_(True), _params(log_scale, freeze_scale)
    loss_c, loss_raw_c, stats_c, gsum_c = L.chunked_infonce_loss_backward(
        imgc, txtc, class_encs_b, targ_data_b, crit_c, _compute_logits_fn(pc), _logit_scale(crit_c, pc), C, False, device,
        rank=0, world_size=1, hpsm_kappas=KAPPAS, hist_bins=HIST_BINS,
    )

    def assert_grad_close(grad_c, grad_ref):
        # float32 rounding scales with the gradient's own magnitude (alpha's), which an absolute floor does not
        torch.testing.assert_close(grad_c, grad_ref, rtol=1e-4, atol=1e-6 * max(1.0, grad_ref.abs().max().item()))

    torch.testing.assert_close(loss_c, loss_ref, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(loss_raw_c, loss_raw_ref, rtol=1e-4, atol=1e-6)
    assert_grad_close(imgc.grad, img.grad)
    assert_grad_close(txtc.grad, txt.grad)
    # the second scale is live under separate scalars only, and a frozen scale takes no gradient on either path
    assert (p["scale2"].grad is not None) == (crit.sep_scalars and not freeze_scale)
    for key in ("scale", "scale2") if crit.sep_scalars else ("scale",):
        if freeze_scale:
            assert pc[key].grad is None and p[key].grad is None
        else:
            assert_grad_close(pc[key].grad, p[key].grad)
    # sum(dL/dsim) is zero up to rounding under InfoNCE -- a row softmax is shift-invariant, so every anchor's
    # gradient sums to zero in its own direction, and so does a residual row -- hence it is held against the
    # gradient's magnitude, which scales with alpha
    assert abs(gsum_c - gsum_ref) <= 1e-5 * gabs_ref
    assert abs(gsum_ref) <= 1e-5 * gabs_ref

    # the batch stats: the full-batch path's keys, all of them. The medians are subsampled per tile on the
    # chunked path (_SimTargStatsAccum), so they are checked for presence only
    assert set(stats_c) == set(stats_ref)
    assert "p_hist" not in stats_c  # sigmoid(logits) is no pair probability under InfoNCE
    skip = {"sim_median", "targ_median"}
    if stats_ref["dscale_sum_abs_res"][0] < 1e-12:
        # a residual at rounding level (rows on the band's edge, e.g. a pinned softmax tsm: mathematically
        # feasible, an ulp past it as stored) is the subtraction's noise on both paths, and the entropy of
        # noise is not a quantity to agree on: the fold pass rebuilds the column anchors' targets to an ulp
        skip.add("sim_grad_entropy_pair_res")
    for key, val in stats_ref.items():
        if key not in skip:
            _assert_stat_close(key, stats_c[key], val)

    # loss.infonce.block_residuals' record, as the full-batch criterion leaves it
    block_c = crit_c.block_stats
    assert (block_c is None) == (block_ref is None) == (cfg["infonce"]["block_residuals"] is None)
    if block_ref is not None:
        assert set(block_c) == set(block_ref)
        for key, val in block_ref.items():
            _assert_stat_close(key, block_c[key], val, rel=1e-4, abs_=1e-9)
        torch.testing.assert_close(crit_c.dlogscale_correction, correction_ref, rtol=1e-4, atol=1e-9)
    return stats_c, block_c


CASES = [
    # (name, cfg, spec1, spec2)
    ("mp", _cfg(), _spec("mp"), _spec("sp")),                                         # hard binary rows (FLYP / SupCon)
    ("sp", _cfg(), _spec("sp"), _spec("mp")),
    ("tax_linear", _cfg(), _spec("tax"), _spec("sp")),                                # graded rows holding zeros: the subtracted residual
    ("phylo_linear", _cfg(), _spec("phylo"), _spec("sp")),                            # graded, full-support rows past the band
    ("tax_softmax_pinned", _cfg(), _spec("tax", "softmax", "pinned"), _spec("sp")),   # inside the band: every row feasible
    ("tax_softmax_c", _cfg(), _spec("tax", "softmax", 3.0), _spec("sp")),             # a fixed scale past the band
    ("phylo_softmax_c", _cfg(), _spec("phylo", "softmax", 3.0), _spec("sp")),
    ("targ_blend_mp_phylo", _cfg(lambda_=0.3), _spec("mp"), _spec("phylo", "softmax", "pinned")),  # loss.yaml's default pair
    ("mp_softmax_pinned3", _cfg(), _spec("mp", "softmax", "pinned3"), _spec("sp")),   # soft two-level rows
    ("targ_blend", _cfg(lambda_=0.3), _spec("mp"), _spec("tax")),
    ("targ_blend_softmax", _cfg(lambda_=0.3), _spec("mp"), _spec("tax", "softmax", "pinned")),  # one normalizer per spec
    ("loss2_alone", _cfg(lambda_=1.0), _spec("mp"), _spec("tax")),
    ("focal", _cfg(focal_gamma=2.0), _spec("mp"), _spec("sp")),                       # the softmax coupling through the focal weights
    ("focal_cls_imb_norm", _cfg(focal_gamma=1.5, cls_imb="inv_freq", norm=True), _spec("tax"), _spec("sp")),
    ("loss_blend_focal", _cfg(lambda_=0.3, blend_type="loss", focal_gamma=2.0, cls_imb="inv_freq"), _spec("mp"), _spec("tax")),
    ("loss_blend_unitless", _cfg(lambda_=0.3, blend_type="loss", unitless=True), _spec("mp"), _spec("tax")),
    ("targ_blend_unitless", _cfg(lambda_=0.3, unitless=True), _spec("mp"), _spec("tax")),
    ("lone_unitless_focal", _cfg(unitless=True, focal_gamma=2.0), _spec("mp"), _spec("sp")),
    ("sep_scalars", _cfg(lambda_=0.3, blend_type="loss", shared=False), _spec("mp"), _spec("tax")),
    ("sep_scalars_unitless_focal", _cfg(lambda_=0.3, blend_type="loss", shared=False, unitless=True, focal_gamma=2.0),
     _spec("mp"), _spec("tax", "softmax", "pinned")),                                 # each spec's tsm pinned to its own scale
    ("geo_focal", _cfg(sim="geo", focal_gamma=2.0), _spec("mp"), _spec("sp")),
    ("clamp", _cfg(clamp=True), _spec("tax", "softmax", "pinned"), _spec("sp")),
    # block_residuals: the hard closed form, the active sets, a target blend's spec-less rows, the soft
    # two-level closed form, per-term coefficients (unitless) and per-term scales (separate logit scalars)
    ("block_alpha_mp", _cfg(block="alpha"), _spec("mp"), _spec("sp")),
    ("block_full_mp", _cfg(block="full"), _spec("mp"), _spec("sp")),
    ("block_full_tax", _cfg(block="full"), _spec("tax"), _spec("sp")),
    ("block_full_targ_blend", _cfg(lambda_=0.3, block="full"), _spec("mp"), _spec("tax")),
    ("block_alpha_pinned_control", _cfg(block="alpha"), _spec("mp", "softmax", "pinned"), _spec("sp")),
    ("block_full_softmax_c", _cfg(block="full"), _spec("mp", "softmax", 3.0), _spec("sp")),
    ("block_full_tax_pinned3", _cfg(block="full"), _spec("tax", "softmax", "pinned3"), _spec("sp")),
    ("block_full_phylo", _cfg(block="full"), _spec("phylo"), _spec("sp")),
    ("block_full_phylo_pinned3", _cfg(block="full"), _spec("phylo", "softmax", "pinned3"), _spec("sp")),
    ("block_full_loss_blend_unitless", _cfg(lambda_=0.3, blend_type="loss", block="full", unitless=True), _spec("mp"), _spec("tax")),
    ("block_full_sep_scalars", _cfg(lambda_=0.3, blend_type="loss", block="full", shared=False), _spec("mp"), _spec("tax")),
]


@pytest.mark.parametrize("C", [16, 48])  # 3 blocks, and single-block (== full)
@pytest.mark.parametrize("name,cfg,spec1,spec2", CASES, ids=[case[0] for case in CASES])
def test_chunked_matches_full(name, cfg, spec1, spec2, C):
    stats, block = _run(cfg, spec1, spec2, C)
    unitless_blend = cfg["unitless"] and cfg["blend"]["type"] == "loss" and 0.0 < cfg["blend"]["lambda"] < 1.0
    assert ("lambda_eff" in stats) == unitless_blend
    # the cases named for a residual path must reach it: a correction that is skipped or feasible everywhere
    # would pass the equivalence above without exercising the tiles' contraction
    if name in ("block_alpha_mp", "block_full_mp", "block_full_softmax_c"):
        assert block["block_coverage"][0][0] == 1.0  # every row a closed form
    if name in ("block_full_tax", "block_full_targ_blend", "block_full_tax_pinned3", "block_full_phylo", "block_full_phylo_pinned3"):
        assert block["block_coverage"][0][0] > 0.0  # some rows through the active sets
    if name == "block_alpha_pinned_control":
        assert block["block_coverage"][0] == [0.0, 1.0, 0.0] and block["dlogscale_correction"] == 0.0


@pytest.mark.parametrize("log_scale", [-0.5, 1.5, 3.0])
def test_chunked_matches_full_across_scales(log_scale):
    # the residual family is exp(-2 alpha)-small: the equivalence has to hold where it is large (alpha 0.6)
    # and where it has shrunk well under the structural terms (alpha 20)
    _run(_cfg(lambda_=0.3, block="full"), _spec("mp"), _spec("tax"), 16, log_scale=log_scale)
    _run(_cfg(focal_gamma=2.0), _spec("tax"), _spec("sp"), 16, log_scale=log_scale)


@pytest.mark.parametrize("block", ["alpha", "full"])
@pytest.mark.parametrize("targ", ["mp", "tax"])
def test_chunked_matches_full_where_the_correction_is_skipped(targ, block):
    # past ALPHA_SUPPORTED_MAX (alpha 403 here) no residual is trusted: every row past the band (these targets'
    # rows hold zeros) is SKIPPED, closed forms and active sets alike, and keeps its ordinary gradient -- on the
    # tiles as on the full batch. exp(-2 alpha) has underflowed float64 there, so the diagnostics' projection
    # floors at zero: the fold pass has to rebuild it from its cap
    stats, stats_block = _run(_cfg(block=block), _spec(targ), _spec("sp"), 16, log_scale=6.0)
    assert stats_block["block_coverage"] == [[0.0, 0.0, 1.0]] and stats_block["dlogscale_correction"] == 0.0
    assert not math.isnan(stats["sim_grad_entropy_pair_struct"])


@pytest.mark.parametrize("name", ["focal_cls_imb_norm", "loss_blend_focal", "sep_scalars_unitless_focal", "geo_focal",
                                  "targ_blend_mp_phylo", "block_alpha_mp", "block_full_tax", "block_full_phylo_pinned3",
                                  "block_full_loss_blend_unitless", "block_full_sep_scalars"])
def test_chunked_is_exact_in_float64(name, monkeypatch):
    """In float32 the two paths agree to ~1e-6 of the gradient's magnitude, their summation orders apart -- a
    tolerance a small missing term could hide under. In float64 that noise drops to ~1e-15, so both are run there:
    float64 embeddings and scales, and Tensor.float() -- by which both paths cast the sims, targets and weights
    down -- patched to keep the precision. What is left between them is the summation order alone."""
    monkeypatch.setattr(torch.Tensor, "float", lambda self: self.double())
    cfg, spec1, spec2 = next(case[1:] for case in CASES if case[0] == name)
    B, K, img0, txt0, class_encs_b, targ_data_b = _batch()
    img0 = torch.nn.functional.normalize(img0.double(), dim=1)
    txt0 = torch.nn.functional.normalize(txt0.double(), dim=1)
    clamp = cfg["logits"]["scale"]["clamp"]

    crit = _make_crit(cfg, spec1, spec2, K, B)
    img, txt, p = img0.clone().requires_grad_(True), txt0.clone().requires_grad_(True), _params(0.5, dtype=torch.float64)
    sim = compute_sim(img, txt, cfg["sim"])
    logits = [_compute_logits_fn(p)(sim, clamp, secondary=sec) for sec in ((False, True) if crit.sep_scalars else (False,))]
    loss, loss_raw, _, _ = crit(logits if crit.sep_scalars else logits[0], class_encs_b, targ_data_b, True, _logit_scale(crit, p), sim)
    loss.backward()

    crit_c = _make_crit(cfg, spec1, spec2, K, B)
    imgc, txtc, pc = img0.clone().requires_grad_(True), txt0.clone().requires_grad_(True), _params(0.5, dtype=torch.float64)
    loss_c, loss_raw_c, _, _ = L.chunked_infonce_loss_backward(
        imgc, txtc, class_encs_b, targ_data_b, crit_c, _compute_logits_fn(pc), _logit_scale(crit_c, pc), 16, False,
        torch.device("cpu"), rank=0, world_size=1, sim_grad_sums=False, sim_targ_stats=False, hist_bins=HIST_BINS,
    )

    assert loss.dtype == loss_c.dtype == img.grad.dtype == imgc.grad.dtype == torch.float64  # else nothing was tested
    torch.testing.assert_close(loss_c, loss.detach(), rtol=1e-12, atol=0)
    torch.testing.assert_close(loss_raw_c, loss_raw.detach(), rtol=1e-12, atol=0)
    for grad_c, grad_ref in ((imgc.grad, img.grad), (txtc.grad, txt.grad),
                             *((pc[key].grad, p[key].grad) for key in (("scale", "scale2") if crit.sep_scalars else ("scale",)))):
        torch.testing.assert_close(grad_c, grad_ref, rtol=1e-9, atol=1e-13 * max(1.0, grad_ref.abs().max().item()))


@pytest.mark.parametrize("name", ["focal_cls_imb_norm", "loss_blend_focal", "sep_scalars_unitless_focal", "geo_focal",
                                  "block_full_mp", "block_full_tax", "block_full_targ_blend", "block_full_softmax_c",
                                  "block_full_tax_pinned3", "block_full_phylo", "block_full_phylo_pinned3",
                                  "block_full_loss_blend_unitless", "block_full_sep_scalars"])
def test_fold_pass_rebuilds_the_full_batch_sim_gradient(name, monkeypatch):
    """The measured entropies are taken off dL/dS rebuilt cell by cell in the fold pass: each image anchor's row
    whole, each text anchor's pairs from its softmax normalizer, that normalizer's adjoint (which carries the
    focal weights' coupling) and, under block_residuals: full, its target row's residual parameters. An entropy
    is a forgiving summary of that, so the rebuilt gradient itself is held against the full-batch path's
    retained one."""
    cfg, spec1, spec2 = next(case[1:] for case in CASES if case[0] == name)
    B, K, img0, txt0, class_encs_b, targ_data_b = _batch()
    crit = _make_crit(cfg, spec1, spec2, K, B)
    img, txt, p = img0.clone().requires_grad_(True), txt0.clone().requires_grad_(True), _params(0.5)
    _, _, G_ref, _ = _full_reference(crit, img, txt, class_encs_b, targ_data_b, p)

    tiles = []
    update = L._SimGradEntropyAccum.update

    def capture(self, G):
        tiles.append(G.detach().double())
        update(self, G)

    monkeypatch.setattr(L._SimGradEntropyAccum, "update", capture)
    crit_c = _make_crit(cfg, spec1, spec2, K, B)
    imgc, txtc, pc = img0.clone().requires_grad_(True), txt0.clone().requires_grad_(True), _params(0.5)
    L.chunked_infonce_loss_backward(
        imgc, txtc, class_encs_b, targ_data_b, crit_c, _compute_logits_fn(pc), _logit_scale(crit_c, pc), 16, False,
        torch.device("cpu"), rank=0, world_size=1, hpsm_kappas=KAPPAS, hist_bins=HIST_BINS,
    )
    assert len(tiles) == B // 16
    torch.testing.assert_close(torch.cat(tiles), G_ref, rtol=1e-4, atol=1e-6 * G_ref.abs().max().item())


@pytest.mark.parametrize("block", [None, "alpha", "full"])
def test_chunked_matches_full_with_a_frozen_scale(block):
    # loss.logits.scale.freeze: the log-scale parameter takes no gradient, so the tiles' logits carry the towers'
    # alone, the scale's part of the block term has nothing to backpropagate into and its recorded correction is
    # zero -- while `full` still reaches the towers through the sims
    _, stats_block = _run(_cfg(lambda_=0.3, blend_type="loss", shared=False, block=block), _spec("mp"), _spec("tax"), 16,
                          freeze_scale=True)
    if block is not None:
        assert stats_block["dlogscale_correction"] == 0.0 and stats_block["dlogscale_correction2"] == 0.0


@pytest.mark.parametrize("block", [None, "alpha", "full"])
def test_chunked_matches_full_under_a_held_clamp(block):
    # loss.logits.scale.clamp with the raw parameter above ln(100): the logits run at alpha 100, the clamp's
    # backward blocks the parameter's gradient (the dlogscale family and the correction read zero) and a tsm
    # pinned to the scale is pinned to the capped one
    stats, stats_block = _run(_cfg(clamp=True, block=block), _spec("mp"), _spec("sp"), 16, log_scale=5.0)
    assert stats["dlogscale_sum_full"][0] == 0.0 and stats["dscale_sum_full"][0] != 0.0
    if block is not None:
        assert stats_block["dlogscale_correction"] == 0.0
    _run(_cfg(clamp=True), _spec("tax", "softmax", "pinned"), _spec("sp"), 16, log_scale=5.0)


def test_batch_diagnostics_off():
    """Disabling either diagnostics component must leave the loss, every gradient and the block_residuals
    record identical (the hooks, the stats and the fold pass only observe): sim_grad_sums=False returns
    grad_sum_sim None and drops the measured entropies, sim_targ_stats=False returns batch_stats None."""
    B, C, K, D = 48, 16, 8, 16
    cfg = _cfg(lambda_=0.3, blend_type="loss", block="full", unitless=True)
    g = torch.Generator().manual_seed(3)
    img0 = torch.nn.functional.normalize(torch.randn(B, D, generator=g), dim=1)
    txt0 = torch.nn.functional.normalize(torch.randn(B, D, generator=g), dim=1)
    class_encs_b = torch.randint(0, K, (B,), generator=g)
    targ_data_b = _make_targ_data(B, K, 4, class_encs_b)

    def run(sim_grad_sums, sim_targ_stats):
        crit = _make_crit(cfg, _spec("mp"), _spec("tax"), K, B)
        img, txt, p = img0.clone().requires_grad_(True), txt0.clone().requires_grad_(True), _params(0.5)
        loss, loss_raw, stats, gsum = L.chunked_infonce_loss_backward(
            img, txt, class_encs_b, targ_data_b, crit, _compute_logits_fn(p), p["scale"], C, False, torch.device("cpu"),
            rank=0, world_size=1, sim_grad_sums=sim_grad_sums, sim_targ_stats=sim_targ_stats, hpsm_kappas=KAPPAS,
            hist_bins=HIST_BINS,
        )
        return loss, loss_raw, stats, gsum, img, txt, p, crit.block_stats

    loss_on, raw_on, stats_on, gsum_on, img_on, txt_on, p_on, block_on = run(True, True)
    actual_keys = {f"sim_grad_entropy_{level}_actual" for level in ("pair", "anchor", "active")}
    assert actual_keys <= set(stats_on) and gsum_on is not None
    for sim_grad_sums, sim_targ_stats in ((False, False), (True, False), (False, True)):
        loss_off, raw_off, stats_off, gsum_off, img_off, txt_off, p_off, block_off = run(sim_grad_sums, sim_targ_stats)
        if sim_targ_stats:  # the stats without the measured-gradient entropies, which need the sim-grad component
            assert stats_off == {key: val for key, val in stats_on.items() if key not in actual_keys}
        else:
            assert stats_off is None
        assert gsum_off == (gsum_on if sim_grad_sums else None)
        assert block_off == block_on
        torch.testing.assert_close(loss_off, loss_on, rtol=0, atol=0)
        torch.testing.assert_close(raw_off, raw_on, rtol=0, atol=0)
        torch.testing.assert_close(img_off.grad, img_on.grad, rtol=0, atol=0)
        torch.testing.assert_close(txt_off.grad, txt_on.grad, rtol=0, atol=0)
        torch.testing.assert_close(p_off["scale"].grad, p_on["scale"].grad, rtol=0, atol=0)


@pytest.mark.parametrize("cfg,spec", [
    (_cfg(block="full"), _spec("mp")),                           # hard two-level rows
    (_cfg(block="full"), _spec("mp", "softmax", 3.0)),           # soft two-level rows
    (_cfg(block="full"), _spec("tax")),                          # active-set rows, many entries tied per level
    (_cfg(block="full"), _spec("tax", "softmax", "pinned3")),
    (_cfg(block="full"), _spec("phylo")),                        # ... and full-support ones
    (_cfg(block="full"), _spec("phylo", "softmax", "pinned3")),
])
def test_resid_row_params_reproduce_the_row_functions(cfg, spec):
    """The fold pass rebuilds a residual from its anchor's row parameters; on the rows themselves that must be
    the row function's own output -- infonce_train_resid's (the correction's) and infonce_resid's (the
    diagnostics') -- so the rebuilt column-anchor half differs from it by the entry's rounding alone."""
    B, K = 48, 8
    g = torch.Generator().manual_seed(1)
    class_encs_b = torch.randint(0, K, (B,), generator=g)
    crit = _make_crit(cfg, spec, _spec("sp"), K, B)
    log_scale = torch.tensor(0.5)
    Q = crit._targets(B, class_encs_b, _make_targ_data(B, K, 4, class_encs_b))
    (Y,) = crit.targ_dists(Q, [log_scale])
    alpha = log_scale.exp().double()

    R, applied, _, _ = L._block_resid_rows(Y, alpha, spec)
    assert applied.any()
    prm = L._train_resid_params(Y, R, applied)
    torch.testing.assert_close(L._train_resid_from_params(Y, *prm.T.unsqueeze(2)), R, rtol=1e-12, atol=1e-18)

    R, P_opt, hard, feasible = L.infonce_resid(Y, alpha)
    prm = L._infonce_resid_params(Y, R, P_opt, hard, feasible)
    R_prm, P_opt_prm = L._infonce_resid_from_params(Y, alpha, *prm.T.unsqueeze(2))
    torch.testing.assert_close(R_prm, R, rtol=1e-9, atol=1e-15)
    torch.testing.assert_close(P_opt_prm, P_opt, rtol=1e-12, atol=0)


def test_chunked_matches_full_under_mixed_precision_with_the_real_head():
    # the production head (VLMWrapper.compute_logits, logit_bias None under InfoNCE) under bf16 autocast: the
    # sims come in bf16 and the head casts them to float32, on the main sweep's tiles and on the fold pass's
    # float32 sim leaf alike
    from types import SimpleNamespace
    from models import VLMWrapper

    device = torch.device("cpu")
    B, K, D, C = 32, 6, 16, 8
    cfg = _cfg(focal_gamma=2.0)
    g = torch.Generator().manual_seed(0)
    img0 = torch.nn.functional.normalize(torch.randn(B, D, generator=g), dim=1)
    txt0 = torch.nn.functional.normalize(torch.randn(B, D, generator=g), dim=1)
    class_encs_b = torch.randint(0, K, (B,), generator=g)

    def head():
        model = SimpleNamespace(logit_scale=torch.nn.Parameter(torch.tensor(1.5)), logit_bias=None)
        stub = SimpleNamespace(_unwrapped_model=model)
        return model, lambda *args, **kwargs: VLMWrapper.compute_logits(stub, *args, **kwargs)

    crit = _make_crit(cfg, _spec("mp"), _spec("sp"), K, B)
    img, txt = img0.clone().requires_grad_(True), txt0.clone().requires_grad_(True)
    model, compute_logits = head()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        sim = compute_sim(img, txt, "cos")
        assert sim.dtype == torch.bfloat16  # else the case under test is not being exercised
        sim.retain_grad()
        loss, *_ = crit(compute_logits(sim, False, None), class_encs_b, None, True, model.logit_scale, sim)
    loss.backward()

    crit_c = _make_crit(cfg, _spec("mp"), _spec("sp"), K, B)
    imgc, txtc = img0.clone().requires_grad_(True), txt0.clone().requires_grad_(True)
    modelc, compute_logits_c = head()
    loss_c, _, stats_c, gsum_c = L.chunked_infonce_loss_backward(
        imgc, txtc, class_encs_b, [None] * B, crit_c, compute_logits_c, modelc.logit_scale, C, True, device,
        rank=0, world_size=1, hist_bins=HIST_BINS,
    )

    torch.testing.assert_close(loss_c, loss.detach(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(modelc.logit_scale.grad, model.logit_scale.grad, rtol=1e-4, atol=1e-6)
    # each tile's embedding grads accumulate bf16 partial products in a different order than the full
    # matmul's backward, so they agree at the bf16-ulp level only (as on the BCE path)
    assert ((imgc.grad - img.grad).norm() / img.grad.norm()).item() < 1e-2
    assert ((txtc.grad - txt.grad).norm() / txt.grad.norm()).item() < 1e-2
    # the measured entropies are rebuilt from a float32 dL/dS where the full-batch path reads the bf16 one
    ref = L.sim_grad_entropies_actual(sim.grad)
    for key, val in ref.items():
        assert stats_c[key] == pytest.approx(val, abs=5e-3), key


def test_chunked_rejects_ragged_bands():
    B, K = 24, 5
    crit = _make_crit(_cfg(), _spec("mp"), _spec("sp"), K, B)
    g = torch.Generator().manual_seed(0)
    img = torch.nn.functional.normalize(torch.randn(B, 8, generator=g), dim=1).requires_grad_(True)
    txt = torch.nn.functional.normalize(torch.randn(B, 8, generator=g), dim=1).requires_grad_(True)
    class_encs_b = torch.randint(0, K, (B,), generator=g)
    p = _params(0.5)
    with pytest.raises(ValueError, match="equal row-bands"):
        L.chunked_infonce_loss_backward(img, txt, class_encs_b, [None] * B, crit, _compute_logits_fn(p), p["scale"], 16, False,
                                        torch.device("cpu"), rank=0, world_size=1, hist_bins=HIST_BINS)
