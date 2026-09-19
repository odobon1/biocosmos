import torch
import torch.nn.functional as F
import torch.distributed as dist
import abc
from contextlib import nullcontext
import math

from utils.rank_encs import compute_rank_dists, compute_rank_encs
from utils.phylo import PhyloVCV
from utils.imb import build_wting, compute_cls_imb_wts, _pair_prob_freqs, _compute_wts
from utils.head import compute_sim
from utils.utils import load_split

import pdb


_phylo_vcv_cache: dict[tuple, PhyloVCV] = {}
_phylo_params: dict | None = None


def configure_phylo_targs(split: str, train_pt: str, batch_size: int, kernel: str, beta: float,
                          shuffle: bool, seed: int | None) -> None:
    """Set the phylo-target params for this run; call once at setup before any loss is computed."""
    global _phylo_params
    _phylo_params = {
        "split": split,
        "train_pt": train_pt,
        "batch_size": batch_size,
        "kernel": kernel,
        "beta": beta,
        "shuffle": shuffle,
        "seed": seed,
    }

def get_phylo_vcv(dataset: str) -> PhyloVCV:
    key = (dataset, *_phylo_params.values())
    if key not in _phylo_vcv_cache:
        _phylo_vcv_cache[key] = PhyloVCV(dataset=dataset, **_phylo_params)
    return _phylo_vcv_cache[key]

def compute_targets(targ_type, batch_size, class_encs_b, targ_data_b, device):
    if targ_type == "sp":
        targs = compute_targs_iw(batch_size)
    elif targ_type == "mp":
        targs = compute_targs_sw(class_encs_b)
    elif targ_type == "tax":
        targs = compute_targs_tax(targ_data_b)
    elif targ_type == "phylo":
        targs = compute_targs_phylo(targ_data_b)
    targs = targs.to(device)  # pt[B, B]

    return targs

def compute_targs_iw(batch_size):
    targs = torch.eye(batch_size)
    return targs

def compute_targs_sw(class_encs_b):
    targs = (class_encs_b.unsqueeze(0) == class_encs_b.unsqueeze(1)).float()
    return targs

def compute_targs_tax(targ_data_b):
    rank_dists = compute_rank_dists(targ_data_b)
    R = len(targ_data_b[0]["rank_encs"])  # tree depth (max rank_dist)
    targs = 1 - rank_dists / R
    return targs

def compute_targs_phylo(targ_data_b):
    dataset = targ_data_b[0]["dataset"]
    targs = get_phylo_vcv(dataset).get_targs_batch(targ_data_b)
    return targs

def targ_specs(lambda_, cfg_loss1, cfg_loss2):
    """
    The live (weight, target cfg) pairs of the target blend Y = (1 - lambda) * Y1 + lambda * Y2 -- cfg_loss1 /
    cfg_loss2 the primary / secondary target params (targ type + InfoNCE tsm). A spec with zero weight
    (loss2 at lambda 0.0, loss1 at lambda 1.0) is dropped outright: its target is never computed.
    """
    return [(w, cfg_targ) for w, cfg_targ in ((1.0 - lambda_, cfg_loss1), (lambda_, cfg_loss2)) if w != 0.0]

def sep_logit_scalars(cfg_loss):
    """
    Whether each loss term runs on its own logit scalars (scale + bias; the model's logit_scale / logit_bias
    for loss1's term, logit_scale2 / logit_bias2 for loss2's): loss.logits.shared false under a loss blend
    (loss.blend.type loss) of two live targets -- a target blend or a lone target is one loss on one set of
    logits, which leaves nothing to separate. Both pairs follow the one loss.logits block (init, freeze,
    clamp, center, scalar_lr_factor).
    """
    return not cfg_loss["logits"]["shared"] and cfg_loss["blend"]["type"] == "loss" and 0.0 < cfg_loss["blend"]["lambda"] < 1.0

class Criterion(abc.ABC):
    """
    A loss over the blended target distribution Y = sum_k w_k Y_k -- w = (1 - lambda, lambda) over the primary /
    secondary target specs (targ_specs), each Y_k its target matrix Q_k under the criterion (targ_dists) --
    paired with the class-imbalance weighting it consumes. The weighting dimensionality is a property of
    the loss (`wting_dim`) -- 1D per-class weights for InfoNCE and bifurcated BCE (per-anchor), 2D
    per-class-pair weights for BCE.

    The training loss is a sum of terms (loss_terms) blended by their coefficients (term_coeffs): under
    loss.blend.type targ the one loss against the blended distribution, under loss.blend.type loss each
    spec's own loss L(Y_k) weighted w_k -- one criterion, similarity and weighting config either way. On
    shared logit scalars the loss is affine in the target, so the two coincide (value and gradients) unless
    a target-dependent factor is on: focal, DSMR and targ_mass_neut read the term's own target, and
    loss.unitless rescales each term to unit magnitude. Under separate logit scalars (sep_scalars) each
    term of a loss blend is scored on its own logits -- the one sim matrix under the term's own scale / bias.

    Only the class counts and the normalization scalar are held; batch weights are computed from
    them on the fly, so no n_classes (1D) / n_classes^2 (2D) weight buffer persists for the run.
    """

    wting_dim: int
    bifurcated = False  # True -> consumes the (i2t, t2i) branch logits pair (see BifurcatedBCECriterion)
    lambda_eff = None  # the last training batch's effective lambda (term_coeffs), or None

    def __init__(self, cfg_loss, cfg_loss1, cfg_loss2, dataset, split, train_pt, device, batch_size):
        self.cfg = cfg_loss
        self.targ_specs = targ_specs(cfg_loss["blend"]["lambda"], cfg_loss1, cfg_loss2)
        self.device = device
        self.batch_size = batch_size
        counts, self.wt_mean = build_wting(cfg_loss["wting"]["cls_imb"], dataset, split, train_pt, self.wting_dim, batch_size)
        self.counts = counts.to(device)

    @staticmethod
    def build(cfg_loss, cfg_loss1, cfg_loss2, dataset, split, train_pt, device, batch_size):
        crit_cls = {
            "infonce": InfoNCECriterion,
            "bce":     BCECriterion,
            "bif_bce": BifurcatedBCECriterion,
        }[cfg_loss["crit"]]

        return crit_cls(cfg_loss, cfg_loss1, cfg_loss2, dataset, split, train_pt, device, batch_size)

    def _targets(self, batch_size, class_encs_b, targ_data_b):
        """The live target specs' matrices Q_k, in targ_specs order; pt[B, B] each."""
        return [compute_targets(cfg_targ["targ"], batch_size, class_encs_b, targ_data_b, self.device) for _, cfg_targ in self.targ_specs]

    def targ_memb(self, Qs):
        """The blended target matrix Q = sum_k w_k Q_k from the specs' matrices Qs (_targets): the pair
        memberships in [0, 1] the batch stats read (target histogram, hard-pair margins, positive / negative
        mass masks). The distribution the loss trains against is targ_dist.

        A lone spec is always at full weight (targ_specs drops the zero-weight one), so its blend is the
        matrix itself and the result is returned as-is rather than as a fresh 1.0 * Q -- the copy it skips
        is a full BxB, twice per InfoNCE batch (the membership blend and the float64 distribution blend,
        48 MB between them at B 2048). The result then ALIASES Qs[0], so nothing may write into a blended
        target in place; nothing does (the only in-place ops in the loss path are scalar accumulators, and
        the collectives all reduce freshly built stat buffers, never a target)."""
        if len(Qs) == 1:
            return Qs[0]
        return sum(w * Q for (w, _), Q in zip(self.targ_specs, Qs))

    @property
    def sep_scalars(self):
        """Whether each loss term runs on its own logit scalars (sep_logit_scalars): __call__ then takes
        per-term logits and log-scales."""
        return sep_logit_scalars(self.cfg)

    def _per_term(self, fn, x, n):
        """fn(x) per loss term (or spec), `n` of them: fn over each term's own entry of `x` under separate
        logit scalars (x then holds one entry per term), else fn(x) once, shared by every term."""
        return [fn(x_k) for x_k in x] if self.sep_scalars else [fn(x)] * n

    def targ_dists(self, Qs, logit_scales):
        """Each live spec's target distribution Y_k from its matrix Q_k (_targets), in targ_specs order: for
        the BCE family each pair's own target probability, Q_k itself; InfoNCE row-normalizes / softmaxes
        Q_k under the spec's own tsm, at the spec's log logit scale (`logit_scales`, one per spec)."""
        return Qs

    def targ_dist(self, Qs, logit_scales):
        """The blended target distribution Y = sum_k w_k Y_k over the specs' distributions (targ_dists)."""
        return self.targ_memb(self.targ_dists(Qs, logit_scales))

    def loss_terms(self, Ys, Y):
        """The (weight, target distribution) terms the training loss sums over, from the specs'
        distributions Ys (targ_dists) and their blend Y (targ_dist): the loss against Y alone under
        loss.blend.type targ, each spec's own loss weighted w_k under loss.blend.type loss."""
        if self.cfg["blend"]["type"] == "targ":
            return [(1.0, Y)]
        return [(w, Y_k) for (w, _), Y_k in zip(self.targ_specs, Ys)]

    def term_coeffs(self, ws, losses):
        """The blend coefficients c_k of the terms' weighted losses L_k (loss = sum_k c_k L_k): the term
        weights `ws`, each divided under loss.unitless by its term's detached magnitude, so every term
        enters at unit magnitude (L_k / L_k.detach()) and its weight is its share of the loss reading (a
        lone term reads a constant 1.0). A bifurcated loss reads 2x its gradient scale (un-halved branch
        sum, 1x grads), so its normalizer is L_k / 2 -- the gradient-scale-equivalent value (a lone term
        reads 2.0).

        Records `lambda_eff` for the batch stats (the lambda_eff learning-curve strip): under loss.unitless
        over a loss blend's two terms, loss2's term's share of the blend coefficients, c_2 / (c_1 + c_2) =
        lambda L_1 / (lambda L_1 + (1 - lambda) L_2) -- with target-independent weights on shared logit
        scalars the lambda of the target blend the gradient follows; None otherwise (a lone term, or the
        static weights)."""
        if not self.cfg["unitless"]:
            self.lambda_eff = None
            return ws
        coeffs = [w / (L.detach() / (2.0 if self.bifurcated else 1.0)).clamp_min(1e-12) for w, L in zip(ws, losses)]
        self.lambda_eff = coeffs[1] / (coeffs[0] + coeffs[1]) if len(coeffs) > 1 else None
        return coeffs

    def _cls_imb_wts(self, class_encs_b):
        return compute_cls_imb_wts(self.cfg["wting"]["cls_imb"], self.counts, class_encs_b, self.wting_dim, self.wt_mean, self.batch_size)

    def _focal_2d(self, Z, Y):
        if "focal" not in self.cfg["wting"]:
            return torch.ones_like(Y)
        gamma = self.cfg["wting"]["focal"]["gamma"]
        return torch.abs(Y - self._preds(Z)).clamp_min(1e-12).pow(gamma)

    @abc.abstractmethod
    def _preds(self, Z):
        """Logits -> prediction probabilities under this criterion's link (for focal weighting)."""
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, logits, class_encs_b, targ_data_b, train, logit_scale):
        """
        Computes loss for a batch given logits and target data. `logits` is the full-batch logit
        matrix pt[B, B]; for a bifurcated criterion, the (i2t, t2i) branch pair, both
        [img-row, txt-col]. `logit_scale` is the learnable log logit scale param (model.logit_scale),
        raw (pre-clamp). Under separate logit scalars (sep_scalars) both are per loss term: a list of the
        terms' logits and of their log-scale params, in targ_specs order.

        Returns:
        - loss ------- Weighted scalar loss (== loss_raw when not training)
        - loss_raw --- Unweighted scalar loss: the terms' raw losses under their weights (the blended
                       target's raw loss on shared logits, the loss being affine in the target)
        - targs ------ Blended target matrix Q (targ_memb); pt[B, B]
        - y ---------- Target distribution Y trained against (targ_dist; a training InfoNCE loss returns
                       the distribution its blended gradient follows -- see InfoNCECriterion); pt[B, B].
                       float64 under InfoNCE (the batch stats are its only consumer and its tsm is
                       solved there -- see InfoNCECriterion._tsm), the loss's own dtype otherwise
        """
        raise NotImplementedError

class InfoNCECriterion(Criterion):
    """
    InfoNCE weighted by 1D per-class weights, applied to per-sample cross-entropy terms.

    Note: may need to be adjusted for multiple GPUs (wrt reduction)
    """

    wting_dim = 1

    def _preds(self, Z):
        return F.softmax(Z, dim=1)

    def _tsm(self, Q, cfg_tsm, logit_scale):
        """One target spec's simplex mapping: its Q row-normalized (linear) or row-softmaxed (softmax).

        Solved and returned in float64. The softmax's negatives reach exp(-2 alpha), which flushes to
        exact zero in float32 from alpha ~ 50.6 -- past the base-model scales the run starts from
        (clip_vitb16 ~100, siglip_vitb16 ~117) under sm_scale pinned. That costs the loss nothing (a
        flushed entry contributes ~1e-86 to its CE) but it destroys the row-wise target-implied scale
        bound the batch stats read off Y: alpha_req = 0.5 * log(max Y / min Y) goes to infinity where
        the true bound is alpha * (max_j Q_ij - min_j Q_ij), at most alpha itself -- pinned means the
        target is always exactly reachable at the current scale, which is the line worth seeing. The
        row sums land on 1 to ~1e-16 here rather than the ~1e-8 float32 manages, which is what
        infonce_p_opt's reachable-set solve needs (see infonce_batch_stats). __call__ takes float32
        copies for the loss and hands these on for the stats. `logit_scale` is left in its own dtype
        so alpha is bit-identical to the one infonce_batch_stats derives from the same parameter.
        """
        Q = Q.double()
        Q_mass = Q.sum(dim=1)

        if cfg_tsm["type"] == "linear":
            Y = Q / Q_mass[:, None]  # pt[B, B]; for MP + HCon (note: symmetrical for MP, non-symmetrical for HCon)
        elif cfg_tsm["type"] == "softmax":
            scale_Q = cfg_tsm["sm_scale"]
            if scale_Q == "pinned" or scale_Q == "pinned1":
                logit_scale = logit_scale.detach()
                if self.cfg["logits"]["scale"]["clamp"]:
                    logit_scale = logit_scale.clamp(max=math.log(100))
                if scale_Q == "pinned":
                    Y = F.softmax(2 * Q * torch.exp(logit_scale), dim=1)  # pt[B, B]; for HCon (note: symmetrical for MP, non-symmetrical for HCon)
                elif scale_Q == "pinned1":
                    Y = F.softmax(Q * torch.exp(logit_scale), dim=1)  # pt[B, B]; for HCon (note: symmetrical for MP, non-symmetrical for HCon)
            else:
                Y = F.softmax(2 * Q * scale_Q, dim=1)  # pt[B, B]; for MP + HCon (note: symmetrical for MP, non-symmetrical for HCon)
        return Y

    def targ_dists(self, Qs, logit_scales):
        return [self._tsm(Q, cfg_targ["infonce"]["tsm"], logit_scale) for (_, cfg_targ), Q, logit_scale in zip(self.targ_specs, Qs, logit_scales)]

    def __call__(self, logits, class_encs_b, targ_data_b, train, logit_scale):
        B = class_encs_b.size(0)
        
        Qs = self._targets(B, class_encs_b, targ_data_b)
        Q = self.targ_memb(Qs)  # pt[B, B]
        # the tsm runs in float64 (see _tsm): the batch stats read those, the loss takes float32 copies of
        # just the targets its terms actually score against
        Ys64 = self.targ_dists(Qs, self._per_term(lambda t: t, logit_scale, len(Qs)))  # pt[B, B] per live spec
        Y64 = self.targ_memb(Ys64)  # pt[B, B]
        terms64 = self.loss_terms(Ys64, Y64)
        terms = [(w, Y_k.float()) for w, Y_k in terms64]
        Zs = self._per_term(lambda Z: Z, logits, len(terms))  # pt[B, B] per term
        log_ps = self._per_term(lambda Z: (F.log_softmax(Z, dim=1), F.log_softmax(Z.T, dim=1)), logits, len(terms))
        losses_raw = [(-Y_k * log_p_i2t, -Y_k * log_p_t2i) for (_, Y_k), (log_p_i2t, log_p_t2i) in zip(terms, log_ps)]  # pt[B, B] pairs

        # per-anchor CE (row sum) averaged over anchors -- CLIP's /B, the scale the weighted loss carries
        loss_raw = sum(
            w * 0.5 * (loss_i2t_raw.sum(dim=1).mean() + loss_t2i_raw.sum(dim=1).mean())
            for (w, _), (loss_i2t_raw, loss_t2i_raw) in zip(terms, losses_raw)
        )

        if self.sep_scalars:
            # no one distribution is trained against across two sets of logits: the batch stats read the
            # primary term's, with its logits and scale (VLMWrapper._batch_stats)
            Y64 = terms64[0][1]

        if not train:
            return loss_raw, loss_raw, Q, Y64

        W_ci = self._cls_imb_wts(class_encs_b)  # class-imbalance weights; pt[B]
        if self.cfg["wting"]["cls_imb"]["norm"]:
            W_ci = W_ci / W_ci.mean()  # pt[B]

        losses = []
        for (_, Y_k), Z, (loss_i2t_raw, loss_t2i_raw) in zip(terms, Zs, losses_raw):
            # Note: 2D-focal is still used despite 1D class-imbalance weighting (reduces to standard focal loss in the SP setting)
            W_i2t = self._focal_2d(Z,   Y_k) * W_ci[:, None]  # pt[B, B]
            W_t2i = self._focal_2d(Z.T, Y_k) * W_ci[:, None]  # pt[B, B]

            loss_i2t = (W_i2t * loss_i2t_raw).sum(dim=1)
            loss_t2i = (W_t2i * loss_t2i_raw).sum(dim=1)

            losses.append(0.5 * (loss_i2t.mean() + loss_t2i.mean()))

        coeffs = self.term_coeffs([w for w, _ in terms], losses)
        loss = sum(c * L for c, L in zip(coeffs, losses))

        if self.cfg["unitless"] and not self.sep_scalars:
            # the distribution the blended gradient follows: the terms' targets under their normalized blend
            # coefficients (the loss is linear in the target) -- unitless reweights a loss blend's terms
            # away from (1 - lambda, lambda), so the batch stats read this Y rather than the static blend
            Y64 = sum(c * Y_k for c, (_, Y_k) in zip(coeffs, terms64)) / sum(coeffs)

        return loss, loss_raw, Q, Y64

class BCECriterion(Criterion):
    """
    Sigmoid BCE weighted by 2D per-class-pair weights.
    """

    wting_dim = 2

    def _preds(self, Z):
        return torch.sigmoid(Z)

    def __call__(self, logits, class_encs_b, targ_data_b, train, logit_scale):
        B = class_encs_b.size(0)

        Qs = self._targets(B, class_encs_b, targ_data_b)
        Y = self.targ_memb(Qs)  # pt[B, B]; the blended targets are the pair probabilities trained against
        terms = self.loss_terms(Qs, Y)

        # fp32: cos-path logits are bf16 under autocast, where sigmoid saturates to exactly 1.0 at
        # |logit| >~ 6 (zeroing focal weights on the easy set, quantizing the rest); upcast once and
        # reuse for both focal preds and the BCE loss. No-op when logits are already fp32 (geo sim).
        logits_f = self._per_term(lambda Z: Z.float(), logits, len(terms))  # pt[B, B] per term

        # Unweighted loss matrices
        losses_2d_raw = [F.binary_cross_entropy_with_logits(Z, Y_k, reduction="none") for (_, Y_k), Z in zip(terms, logits_f)]  # pt[B, B] each
        loss_raw = sum(w * loss_2d_raw.detach().sum() for (w, _), loss_2d_raw in zip(terms, losses_2d_raw)) / B

        if not train:
            return loss_raw, loss_raw, Y, Y

        W_ci = self._cls_imb_wts(class_encs_b)  # class-imbalance weights; pt[B, B]
        if self.cfg["wting"]["cls_imb"]["norm"]:
            W_ci = W_ci / W_ci.mean()

        losses = []
        for (_, Y_k), Z, loss_2d_raw in zip(terms, logits_f, losses_2d_raw):
            W = W_ci * self._focal_2d(Z, Y_k)  # pt[B, B]
            if self.cfg["wting"]["bce"]["dsmr"]:
                mass_pos = torch.sum(Y_k)
                mass_neg = torch.sum(1.0 - Y_k)
                W = W * _dsmr_weight(Y_k, mass_pos, mass_neg, B)

            losses.append((W * loss_2d_raw).sum() / B)

        coeffs = self.term_coeffs([w for w, _ in terms], losses)
        loss = sum(c * L for c, L in zip(coeffs, losses))

        return loss, loss_raw, Y, Y

class BifurcatedBCECriterion(Criterion):
    """
    Sigmoid BCE with a per-direction anchor branch, weighted by 1D per-class (per-anchor) weights.

    Bifurcated: `logits` is the branch pair (logits_bif_i2t, logits_bif_t2i), both [img-row,
    txt-col]; the t2i branch is consumed as its transposed [txt-row, img-col] view, so each
    direction's anchors are rows and per-anchor weighting/reduction is row-wise in both branches
    (pairing the transposed view against Y relies on Y being symmetric, which holds for every
    targ type and hence for their blend). The i2t branch backprops into the image tower only (txt detached upstream), t2i
    into the text tower only. The un-halved branch sum makes the loss value (and loss_raw) 2x the
    non-bifurcated reading, but with identical weighting on both branches every gradient matches
    non-bifurcated 1x: towers live in one branch each, and the logit scale/bias are half-live
    upstream (compute_logits) so their two branch contributions sum to 1x.
    """

    wting_dim = 1
    bifurcated = True

    def _preds(self, Z):
        return torch.sigmoid(Z)

    def __call__(self, logits, class_encs_b, targ_data_b, train, logit_scale):
        B = class_encs_b.size(0)

        Qs = self._targets(B, class_encs_b, targ_data_b)
        Y = self.targ_memb(Qs)  # pt[B, B]; the blended targets are the pair probabilities trained against
        terms = self.loss_terms(Qs, Y)

        # fp32: cos-path logits are bf16 under autocast, where sigmoid saturates to exactly 1.0 at
        # |logit| >~ 6 (zeroing focal weights on the easy set, quantizing the rest); upcast once and
        # reuse for both focal preds and the BCE loss. No-op when logits are already fp32 (geo sim).
        # Per term its (i2t, t2i) pair, the t2i branch transposed: anchors as rows
        logits_f = self._per_term(lambda Zs: (Zs[0].float(), Zs[1].T.float()), logits, len(terms))

        # Unweighted loss matrices
        losses_2d_raw = [
            tuple(F.binary_cross_entropy_with_logits(Z, Y_k, reduction="none") for Z in Zs)  # pt[B, B] each
            for (_, Y_k), Zs in zip(terms, logits_f)
        ]
        loss_raw = sum(
            w * (loss_i2t_2d_raw.detach().sum() / B + loss_t2i_2d_raw.detach().sum() / B)
            for (w, _), (loss_i2t_2d_raw, loss_t2i_2d_raw) in zip(terms, losses_2d_raw)
        )

        if not train:
            return loss_raw, loss_raw, Y, Y

        # per-anchor weights are row weights in both branches (rows = each direction's anchors;
        # paired image/text share the class, so w_ci indexes both directions)
        w_ci = self._cls_imb_wts(class_encs_b)  # class-imbalance weights; pt[B]
        if self.cfg["wting"]["cls_imb"]["norm"]:
            w_ci = w_ci / w_ci.mean()

        losses = []
        for (_, Y_k), (logits_i2t_f, logits_t2i_f), (loss_i2t_2d_raw, loss_t2i_2d_raw) in zip(terms, logits_f, losses_2d_raw):
            W_i2t = w_ci[:, None] * self._focal_2d(logits_i2t_f, Y_k)  # pt[B, B]
            W_t2i = w_ci[:, None] * self._focal_2d(logits_t2i_f, Y_k)  # pt[B, B]
            if self.cfg["wting"]["bce"]["dsmr"]:
                W_dsmr = _dsmr_weight_rows(Y_k, B)  # pt[B, B]
                W_i2t = W_i2t * W_dsmr
                W_t2i = W_t2i * W_dsmr

            # 2D --> 1D: per-anchor (row) reduction
            loss_i2t_1d = (W_i2t * loss_i2t_2d_raw).sum(dim=1)  # pt[B]
            loss_t2i_1d = (W_t2i * loss_t2i_2d_raw).sum(dim=1)  # pt[B]

            if self.cfg["bce"]["targ_mass_neut"]:
                Y_mass = Y_k.sum(dim=1)  # pt[B]
                loss_i2t_1d = loss_i2t_1d / Y_mass
                loss_t2i_1d = loss_t2i_1d / Y_mass

            # Batch-mean over anchors, branches summed un-halved
            losses.append(loss_i2t_1d.mean() + loss_t2i_1d.mean())

        coeffs = self.term_coeffs([w for w, _ in terms], losses)
        loss = sum(c * L for c, L in zip(coeffs, losses))

        return loss, loss_raw, Y, Y

def _dsmr_weight(targs, mass_pos, mass_neg, B):
    """
    DSMR (dynamic same-class mass reweighting) pair weight from the global pos/neg target mass. Shared
    by the full-batch (BCECriterion.__call__) and tiled (chunked) paths -- for a tile, `targs` is the
    [C, B] block and (mass_pos, mass_neg) are the full-BxB masses. mass_neg == 0 (all-positive batch)
    is masked to 1.0 (scale -> inf otherwise).
    """
    scale = B**2 / (2 * mass_pos * mass_neg)  # mass_neg == 0 --> inf here, but masked below (guard against div-by-zero for all-positive batch)
    wt_neg = scale * mass_pos
    wt_pos = scale * mass_neg
    W_dsmr = torch.where(
        mass_neg == 0, 
        torch.ones_like(targs), 
        targs * wt_pos + (1 - targs) * wt_neg
    )
    return W_dsmr

def _dsmr_weight_rows(targs, B):
    """
    Row-wise DSMR: _dsmr_weight applied independently per row, with each row's pos/neg target mass in
    place of the global masses (row weight mass sums to B, split evenly between pos and neg). Rows must
    span the full B columns (holds for tiles too -- a [C, B] block has complete rows). All-positive rows
    (mass_neg == 0) are masked to 1.0.
    """
    mass_pos = targs.sum(dim=1, keepdim=True)          # pt[B, 1]
    mass_neg = (1 - targs).sum(dim=1, keepdim=True)    # pt[B, 1]
    scale = B / (2 * mass_pos * mass_neg)
    wt_neg = scale * mass_pos
    wt_pos = scale * mass_neg
    W_dsmr = torch.where(  # guard against div-by-zero for all-positive rows
        mass_neg == 0,
        torch.ones_like(targs),
        targs * wt_pos + (1 - targs) * wt_neg
    )
    return W_dsmr

def pos_prevalence(cfg_loss, cfg_loss1, cfg_loss2, dataset, split, train_pt, batch_size):
    """
    Expected weighted positive prevalence p = sum(W * Y) / sum(W) of a BCE-family loss's B x B blended
    target matrix (diagonal included) under uniform batch sampling from the train partition -- the constant
    sigmoid probability minimizing the expected weighted BCE (for binary and soft targets alike), so
    logit(p) is the matched logit-bias init (logits.bce.bias.init: pos_prevalence). Computed once at the
    class level from the pair-probability matrix (_pair_prob_freqs, sums to B), each same-class cell
    split into the anchor's own slot (target 1 under every targ type) and its other same-class slots
    (target 0 under sp, 1 otherwise); the blend Y = sum_k w_k Y_k is taken over the live target specs
    (targ_specs) before the weights, which read it. W holds the loss's weights that are constant given
    the pair's classes -- class-imbalance (per-anchor / per-pair by the criterion's wting_dim), DSMR and
    targ_mass_neut with the expected (global / per-anchor-class) target masses in place of the batch's
    -- not focal (prediction-dependent). Under loss.blend.type loss the loss is a sum of per-spec terms
    (Criterion.loss_terms), each under its own target's weights W_k, and the minimizing constant is
    p = sum_k w_k sum(W_k * Y_k) / sum_k w_k sum(W_k) -- under the static term weights w_k (loss.unitless'
    coefficients follow the model's losses, unknown at init). Dataset-level constant, identical across
    DDP ranks.
    """
    B = batch_size
    split_obj = load_split(dataset, split)
    counts = torch.tensor(split_obj.class_counts[train_pt], dtype=torch.float64)
    encs = (~torch.isnan(counts)).nonzero(as_tuple=True)[0]  # classes present in the partition
    K = encs.numel()

    specs = targ_specs(cfg_loss["blend"]["lambda"], cfg_loss1, cfg_loss2)
    Y_rests = []  # per live spec, the other B-1 slots; under sp only the anchor's own slot is positive
    for _, cfg_targ in specs:
        if cfg_targ["targ"] == "sp":
            Y_rests.append(torch.zeros(K, K, dtype=torch.float64))
            continue
        cids = [split_obj.enc2cid[int(enc)] for enc in encs]
        rank_encs = compute_rank_encs(dataset, cids) if cfg_targ["targ"] == "tax" else [None] * K
        targ_data_cls = [{"cid": cid, "dataset": dataset, "rank_encs": re} for cid, re in zip(cids, rank_encs)]
        Y_rests.append(compute_targets(cfg_targ["targ"], K, encs, targ_data_cls, "cpu").double())  # [K, K]; unit diagonal (same-class slots positive)
    if cfg_loss["blend"]["type"] == "targ":
        terms = [(1.0, sum(w * Y_rest for (w, _), Y_rest in zip(specs, Y_rests)))]
    else:
        terms = [(w, Y_rest) for (w, _), Y_rest in zip(specs, Y_rests)]

    # [2, K, K]: (anchor's own slot -- diagonal, P(anchor class); the other B-1 slots) x class pair
    P_full = _pair_prob_freqs(counts, encs, B)
    P_self = torch.diag(counts[encs] / counts.nansum())
    P = torch.stack((P_self, P_full - P_self))

    wting_dim = {"bce": 2, "bif_bce": 1}[cfg_loss["crit"]]
    mass_WY = mass_W = 0.0
    for w_term, Y_rest in terms:
        Y = torch.stack((torch.ones(K, K, dtype=torch.float64), Y_rest))
        if wting_dim == 2:
            W = _compute_wts(cfg_loss["wting"]["cls_imb"], P_full)  # per-pair; [K, K]
            if cfg_loss["wting"]["bce"]["dsmr"]:
                mass_pos = B * (P * Y).sum()  # expected batch target mass: each of the B anchor slots contributes sum(P * Y)
                W = W * _dsmr_weight(Y, mass_pos, B**2 - mass_pos, B)
        else:
            W = _compute_wts(cfg_loss["wting"]["cls_imb"], counts[encs])[:, None]  # per-anchor (row); [K, 1]
            row_mass = (P * Y).sum(dim=(0, 2)) / P_self.diagonal()  # expected row target mass given the anchor's class; [K]
            if cfg_loss["wting"]["bce"]["dsmr"]:
                W = W * _dsmr_weight(Y, row_mass[:, None], B - row_mass[:, None], B)  # _dsmr_weight_rows' B vs B**2 scale: constant factor, cancels in the ratio
            if cfg_loss["bce"]["targ_mass_neut"]:
                W = W / row_mass[:, None]
        mass_WY += w_term * (W * P * Y).sum()
        mass_W += w_term * (W * P).sum()

    return (mass_WY / mass_W).item()

# ------------------------------------------------------------------------------------------------
# Tiled / chunked global-batch loss (hardware.loss_chunk_size)
#
# The full-batch contrastive loss materializes several BxB matrices (sim, logits, weights, loss) and
# their autograd graph -- O(B^2) VRAM, the wall that OOMs bs32k. The chunked path computes the exact
# same weighted-BCE loss and gradients while never holding the full BxB matrices, and no rank computes
# more than its share of them: the BxB rows are sharded across ranks into world_size equal row-bands of
# b = B/world_size (SigLIP-style decomposition -- the pairwise-independent BCE loss has no batch-global
# normalizer, so each rank sweeps only its own band instead of redundantly recomputing the full matrix),
# and each rank sums its band over b/C row-blocks of C x B (C = loss_chunk_size rows, b an exact multiple
# of C), backpropagating each block into the detached embedding leaves (GradCache-style representation
# gradients). Peak VRAM is O(C*B); per-rank loss compute is O(B^2/world_size). Cross-band couplings --
# the loss/raw totals, the batch stats, and the precomputed constants below -- are folded with small
# all-reduces so every rank returns identical full-batch values; the leaves' band-partial dL/dembs sum
# to the full gradient across ranks (completed by batch_step_chunked's grad all-reduce).
#
# Supports the full BCE-family config space (bce and bif_bce, incl. mp/sp/tax/phylo targets and their
# blend under either loss.blend.type, cls_imb.norm, loss.unitless) -- only InfoNCE is excluded
# (chunking_supported). The target tiles are the loss terms' (loss_term_spec_fns): the criterion's blended
# targets under blend.type targ (blend_targ_block_fn: sum_k w_k Q_k over the live target specs, Y = Q for
# the BCE family), each spec's own Q_k under blend.type loss -- all on the block's one logits tile, or
# under separate logit scalars (loss.logits.shared false) each term on its own scalar pair's logits tile
# off the block's one sim tile, grad_proj* then projecting per pair (_crit_center_grad_mean). The
# reductions that couple across the whole BxB matrix -- the cls_imb.norm weight-mean normalizers (a 2D
# band sweep for bce; bif_bce's 1D per-anchor vector is O(B) and built outright), bce's global DSMR mass
# per term, and under loss.unitless the terms' loss magnitudes (_term_loss_values) -- are all DETACHED
# constants, so they are precomputed (cheap embedding-free closed forms + no_grad band sweeps,
# all-reduced to rank-identical values) before the single grad-carrying backward sweep applies them as
# constants. See _precompute_crit_consts.
#
# A bifurcated criterion (bif_bce) runs each block as TWO branch tiles in the branches' own anchor
# frames -- i2t = (img rows, detached txt), t2i = (txt rows, detached img); Y is symmetric, so one
# targ block serves both -- with half-live logit scalars (compute_logits), reproducing the full-batch
# opposite-tower detach routing directly on the leaves (~2x that criterion's per-block compute/memory,
# same O(C*B) asymptotics). Its row-local weightings (row-wise DSMR, targ_mass_neut) need complete
# rows, which every [C, B] tile has.
#
# Centering (logits.bce.center) is reproduced exactly, never per-tile: "sim" recovers the full-batch
# sim mean in-graph per block via the cos bilinearity mean(sim) = mean(img) . mean(txt) (geo sims have
# no closed form -- TrainConfig rejects center: sim + chunking + geo, and this module asserts it);
# grad_proj/grad_proj2 precompute the full-batch mean of the incoming gradient at the projection node
# (_crit_center_grad_mean: a band sweep with tile-local autograd, all-reduced) and every tile subtracts
# that constant (_ZeroSumGradConst via compute_logits's center_global) in place of the full path's
# g.mean() -- per branch for a bifurcated criterion (each branch's projection sees only its own
# incoming grad, and the two frames' grad means differ under per-anchor row weighting).
# ------------------------------------------------------------------------------------------------

def chunking_supported(cfg_loss):
    """
    The tiled loss reproduces every BCE-family config (bce and bif_bce) but not InfoNCE (its
    row/column softmax couples along columns, which a row-block cannot tile). Config treats
    hardware.loss_chunk_size as inert (full BxB path) when this returns False.
    """
    return cfg_loss["crit"] in ("bce", "bif_bce")

def make_targ_block_fn(targ_type, class_encs_b, targ_data_b, B, device):
    """
    Build a closure (rs, re) -> [re-rs, B] target row-block (rows rs:re vs all B columns) matching the
    full-batch compute_targets for the given targ_type. Reusable per-tile inputs (tax rank vectors,
    phylo correlation lookups) are precomputed once here so the sweeps only slice per block.
    """
    if targ_type == "mp":
        return lambda rs, re: (class_encs_b[rs:re].unsqueeze(1) == class_encs_b.unsqueeze(0)).float()
    if targ_type == "sp":
        cols = torch.arange(B, device=device)
        return lambda rs, re: (torch.arange(rs, re, device=device).unsqueeze(1) == cols.unsqueeze(0)).float()
    if targ_type == "tax":
        tax_vecs = torch.tensor([td["rank_encs"] for td in targ_data_b], device=device)  # [B, R]
        R = tax_vecs.size(1)
        def targ_block(rs, re):
            neq = (tax_vecs[rs:re].unsqueeze(1) != tax_vecs.unsqueeze(0)).int()  # [C, B, R]
            # first differing rank (deterministic, no argmax tie-break); all-equal -> R (rank_dist 0)
            div = R - (neq.cumsum(dim=2) >= 1).sum(dim=2)  # divergence level; [C, B]
            rank_dists = R - div
            return (1.0 - rank_dists / R).float()
        return targ_block
    if targ_type == "phylo":
        return get_phylo_vcv(targ_data_b[0]["dataset"]).make_targ_block_fn(targ_data_b, device)

def make_targ_block_fns(crit, class_encs_b, targ_data_b, B, device):
    """
    Per live target spec of the criterion (targ_specs), its (w_k, targ_type, make_targ_block_fn closure)
    triple -- the pieces of the blended row-block Y[rs:re] = sum_k w_k Q_k[rs:re] (blend_targ_block_fn),
    kept apart for the reductions that read the specs individually (bce_dsmr_mass' closed forms).
    """
    return [(w, cfg_targ["targ"], make_targ_block_fn(cfg_targ["targ"], class_encs_b, targ_data_b, B, device))
            for w, cfg_targ in crit.targ_specs]

def blend_targ_block_fn(spec_fns):
    """(rs, re) -> the blended [re-rs, B] target row-block Y[rs:re] = sum_k w_k Q_k[rs:re] over `spec_fns`
    (make_targ_block_fns) -- the criterion's targ_memb / targ_dist tile for the BCE family (Y = Q)."""
    return lambda rs, re: sum(w * fn(rs, re) for w, _, fn in spec_fns)

def loss_term_spec_fns(crit, spec_fns):
    """
    The tiled analogue of Criterion.loss_terms: per loss term its (term weight, term spec_fns) pair, the
    term's target tile being blend_targ_block_fn over its spec_fns -- every live spec blended into one term
    under loss.blend.type targ, each spec alone (its unweighted Q_k) under loss.blend.type loss.
    """
    if crit.cfg["blend"]["type"] == "targ":
        return [(1.0, spec_fns)]
    return [(w, [(1.0, targ_type, fn)]) for w, targ_type, fn in spec_fns]

def bce_dsmr_mass(spec_fns, class_encs_b, B, chunk_size, lo, hi, world_size):
    """
    Global DSMR mass over the full BxB blended target matrix: mass_pos = sum(Y) = sum_k w_k sum(Q_k) (the
    blend is linear), mass_neg = B^2 - mass_pos (== sum(1 - Y) for targets in [0, 1]). `spec_fns` are the
    live specs' (w_k, targ_type, block closure) triples (make_targ_block_fns). A spec's sum(Q_k) is, for
    mp/sp (0/1 targets), the O(B) closed form sum_c count_c^2 (rank-identical, no collective); for soft
    tax/phylo targets it is summed over this rank's band [lo, hi) of target tiles (embedding-free) and
    all-reduced across bands. Matches torch.sum(Y) / torch.sum(1 - Y) in BCECriterion.__call__.
    """
    device = class_encs_b.device
    mass_pos = torch.zeros((), dtype=torch.float64, device=device)
    for w, targ_type, targ_block_fn in spec_fns:
        if targ_type == "mp":
            counts = torch.bincount(class_encs_b).to(torch.float64)
            mass_k = (counts * counts).sum()
        elif targ_type == "sp":
            mass_k = torch.tensor(float(B), dtype=torch.float64, device=device)
        else:  # tax, phylo -- soft targets: sum over this rank's band of tiles, fold across bands
            mass_k = torch.zeros((), dtype=torch.float64, device=device)
            for rs in range(lo, hi, chunk_size):
                mass_k += targ_block_fn(rs, rs + chunk_size).double().sum()
            if world_size > 1:
                dist.all_reduce(mass_k)
        mass_pos += w * mass_k
    mass_pos = mass_pos.to(dtype=torch.float32)
    mass_neg = torch.tensor(float(B) * float(B), dtype=torch.float32, device=device) - mass_pos
    return mass_pos, mass_neg


HIST_BINS = 20  # bins spanning [0, 1] for the target / predicted-probability histogram strips


def hard_pair_similarity_margin(S, Q, kappa):
    """
    Hardness-weighted continuous-Q hard-pair similarity margin, per row.

        w^+_ij ∝ q_ij     exp(-kappa * s_ij)
        w^-_ij ∝ (1-q_ij) exp(+kappa * s_ij)

        Δs_i(kappa) = sum_j w^+_ij s_ij - sum_j w^-_ij s_ij

    kappa = 0 reduces exactly to the continuous-Q mean-separation margin (the q-weighted mean
    similarity of a row's positives minus the (1-q)-weighted mean of its negatives). Increasing
    kappa concentrates the positive side on lower-similarity ("hard") positives and the negative
    side on higher-similarity ("hard") negatives. Generalizes over sp / mp / continuous targets.

    - S ---- [R, B] similarities (a full BxB matrix, or a row-block of one)
    - Q ---- [R, B] target memberships in [0, 1]
    - kappa  hardness weight; reporting.learning_curves.hpsm.kappas lists the values curved

    Returns Δs_i(kappa) per row, [R]; the batch statistic is its mean over all B rows. Rows are the
    anchors: on S (image rows, text columns) that is the I2T margin, on S.T with Q.T the T2I one; the
    batch stats report both and their mean, mirroring the bidirectional loss.
    """
    assert S.shape == Q.shape
    assert S.ndim == 2
    # log-weights; log(0) = -inf keeps an exact-zero membership at exactly zero weight
    pos_logits = torch.log(Q) - kappa * S
    neg_logits = torch.log1p(-Q) + kappa * S
    w_pos = torch.softmax(pos_logits, dim=1)
    w_neg = torch.softmax(neg_logits, dim=1)
    return (w_pos * S).sum(dim=1) - (w_neg * S).sum(dim=1)


def infonce_p_opt(Y, alpha, n_iter=60):
    """
    Row-wise reachable-set projection of the InfoNCE target distribution Y: the closest distribution
    a row softmax can realize under bounded-cosine logits. With s in [-1, 1] and z = alpha * s, a
    row's logits span at most 2 * alpha, so its softmax p can only realize distributions with
    max(p) / min(p) <= exp(2 * alpha). The one closest to a target row y in the cross-entropy the
    loss minimizes (-sum y log p; the I-projection) is

        p* = clamp(y, lam, exp(2 * alpha) * lam),  sum(p*) = 1

    -- entries already inside the band are kept, the rest pinned to its edges (a target of exactly
    0, an sp / mp negative, lands on the floor lam), lam set by the sum. lam is solved per row by
    bisection on eta = log(lam) over [-log(B) - 2 * alpha, -log(B)], where the clamped row mass is
    <= 1 and >= 1 respectively (monotone in eta between). 60 halvings exhausts float64 at any alpha:
    the bracket is 2 * alpha wide and eta itself is of order 2 * alpha, so the ulp it has to resolve
    is ~2 * alpha * 2^-52 and the scale cancels -- ~52 halvings reach one ulp, 60 leave margin, and
    more are wasted. Fewer are not: the shortfall is 2 * alpha * 2^-(n+1), which at n = 40 corrupts
    the residual p* - y by 3% at alpha 14 and by three orders past alpha 20. One p* serves both anchor
    directions:
    InfoNCECriterion reads Y's rows as each anchor's targets in the I2T and the T2I direction alike.

    - Y ------ [R, B] target distributions (rows summing to 1; Criterion.targ_dist under InfoNCE)
    - alpha -- the logit scale alpha = exp(logit_scale) the logits carry (post-clamp)

    Returns p* in Y's shape, float64 (the residual p* - y underflows float32 at moderate alpha).
    """
    Y = Y.detach().double()
    alpha = torch.as_tensor(alpha, dtype=torch.float64, device=Y.device).detach()
    log_Y = Y.log()  # log(0) = -inf: a zero target sits at the floor after the clamp
    hi = torch.full((Y.size(0), 1), -math.log(Y.size(1)), dtype=torch.float64, device=Y.device)  # row mass >= 1
    lo = hi - 2 * alpha  # row mass <= 1
    for _ in range(n_iter):
        eta = 0.5 * (lo + hi)
        under = log_Y.clamp(min=eta, max=eta + 2 * alpha).exp().sum(dim=1, keepdim=True) < 1.0
        lo = torch.where(under, eta, lo)
        hi = torch.where(under, hi, eta)
    eta = 0.5 * (lo + hi)
    return log_Y.clamp(min=eta, max=eta + 2 * alpha).exp()


def infonce_s_opt(P_opt, alpha):
    """
    The row geometry that realizes the reachable optimum p* = infonce_p_opt(Y, alpha) at that alpha:
    invert the row softmax (p* is full-support -- its floor lam > 0 -- so its log is finite) and
    recenter each row on its own midrange,

        s* = (log(p*) - (max_j log(p*_j) + min_j log(p*_j)) / 2) / alpha

    The softmax is shift-invariant per row, so the recentering leaves p* untouched and is the one
    choice symmetric about zero; p*'s log range is at most 2 * alpha by construction, so the
    recentered logits sit in [-alpha, alpha] and s* in [-1, 1] -- a bounded-cosine geometry. It is a
    row-wise construct, not a Gram matrix: nothing makes it symmetric or realizable by actual
    embeddings, it is just the similarity row that would put the model exactly at p*.

    - P_opt -- [R, B] infonce_p_opt(Y, alpha)
    - alpha -- the logit scale the logits carry (post-clamp), as passed to infonce_p_opt

    Returns s* in P_opt's shape, float64.
    """
    log_P_opt = P_opt.log()
    return (log_P_opt - 0.5 * (log_P_opt.amax(1, keepdim=True) + log_P_opt.amin(1, keepdim=True))) / alpha


def infonce_scale_grad_sums(S, Q, Y, P, P_opt, S_opt):
    """
    One anchor direction's per-pair InfoNCE logit-scale gradient terms, decomposed and aggregated:

        dL/dalpha_ij = (p_ij - y_ij) s_ij                                    (full)
                     = (p_ij - p*_ij) s_ij  +  (p*_ij - y_ij) s_ij            (struct + res)

    with p the row softmax of the logits and p* = infonce_p_opt(Y, alpha). 'struct' is the part the
    model could still remove at this alpha (its p is not the reachable optimum p*), 'res' the part
    no similarity geometry can (the target lies outside the reachable set: sp / mp zeros, or graded
    targets steeper than exp(2 * alpha) allows). The residual splits again along s* =
    infonce_s_opt(P_opt, alpha), the geometry that realizes p*:

        (p*_ij - y_ij) s_ij = (p*_ij - y_ij)(s_ij - s*_ij) + (p*_ij - y_ij) s*_ij    (sres + ires)

    -- 'sres' the part carried by the model's geometry standing off the optimal one (it vanishes as
    s approaches s*, however unreachable the target is), 'ires' what the optimal geometry itself
    still pushes on alpha, the pressure that survives at p = p*. Each term is also attributed to the
    positive and negative target mass by the soft masks q_ij and 1 - q_ij (generalizing the binary
    split over graded targets).

    - S, Q, Y, P, P_opt, S_opt -- [R, B]: sims, target memberships, target distributions, the
      predicted row distributions, the reachable optimum and its geometry, rows = anchors (image
      anchors on the batch's S / Q / Y / softmax(logits); text anchors on S.T / Q.T / Y /
      softmax(logits.T), Y -- and with it p* and s* -- serving both directions as in
      InfoNCECriterion)

    Returns [2, 5, 3]: (sum, sum of |.|) x (full, struct, res, sres, ires) x (all, positive,
    negative mass), each a per-anchor row sum averaged over the anchors -- so the full / all sum is
    exactly this direction's d(loss_raw)/d(alpha) (the per-anchor mean CE the loss carries).
    """
    R = P_opt - Y
    terms = ((P - Y) * S, (P - P_opt) * S, R * S, R * (S - S_opt), R * S_opt)
    masks = (None, Q, 1.0 - Q)
    sums = []
    for G in terms:
        for M in masks:
            GM = G if M is None else G * M
            sums.append(torch.stack([GM.sum(), GM.abs().sum()]))
    return torch.stack(sums).view(5, 3, 2).permute(2, 0, 1) / S.size(0)


def infonce_kl_terms(Y, log_P, P_opt):
    """
    One anchor direction's InfoNCE target-to-prediction divergence, decomposed through the reachable
    optimum p* = infonce_p_opt(Y, alpha):

        D_KL(y || p) = D_KL(y || p*) + D_KL(p* || p) + <y - p*, log(p* / p)>
                     =     E_ir      +      E_s      +        E_sr

    per anchor row, each averaged over the anchors. D_KL(y || p) is the direction's raw loss (the
    per-anchor CE) less the targets' entropy -- the part of the loss training can drive down. E_ir
    is the part no similarity geometry can remove at this alpha (the target lies outside the
    reachable set), E_s the part the model still could (its p is not p*), E_sr the cross term. All
    four are >= 0: the divergences by definition, E_sr because p is itself reachable -- with A the
    target mass the cap clips off, E_sr = A * (2 alpha - the mean log-ratio of p between the capped
    and the floored entries), and no two entries of p differ by more than 2 alpha in logit; it is
    zero iff p keeps every capped entry the full 2 alpha above every floored one, as p* does.

    - Y ------ [R, B] target distributions (rows = anchors; Y serves both directions, as in
               InfoNCECriterion)
    - log_P -- [R, B] the anchors' predicted log distributions (the row log-softmax of the
               direction's logits)
    - P_opt -- infonce_p_opt(Y, alpha)

    Returns [4]: the batch means of (D_KL(y || p), E_s, E_ir, E_sr).
    """
    log_P_opt = P_opt.log()
    neg_H = torch.xlogy(Y, Y).sum(dim=1)  # -H(y); 0 log 0 = 0
    kl = neg_H - (Y * log_P).sum(dim=1)
    E_s = (P_opt * (log_P_opt - log_P)).sum(dim=1)
    E_ir = neg_H - (Y * log_P_opt).sum(dim=1)
    E_sr = ((Y - P_opt) * (log_P_opt - log_P)).sum(dim=1)
    return torch.stack([kl.mean(), E_s.mean(), E_ir.mean(), E_sr.mean()])


def infonce_batch_stats(sim, targs, y, logits, logit_scale, clamp):
    """
    The InfoNCE loss's per-batch reachable-optimum (p* = infonce_p_opt) diagnostics, each taken over
    both anchor directions and averaged -- the loss is the mean of the two directions' CE:

    - the logit-scale gradient decomposition (the dalpha_* and dlogalpha_* learning-curve strips): infonce_scale_grad_sums, so the averaged full / all sum is d(loss_raw)/d(alpha), with
      the coherence C = |sum| / sum|.| taken on the averaged sums (the bidirectional gradient's own
      cancellation ratio, so C = |sum| / sum_abs holds across the reported series). alpha is the
      scale the logits carry, post-clamp, so the dalpha family is the loss's pressure on that
      effective scale whether or not the parameter can follow it. The dlogalpha family is the
      gradient of the log-scale parameter the model actually learns (logits.scale: the param is
      log(alpha_raw), and z = exp(min(log alpha_raw, ln 100)) * s under logits.scale.clamp, exp(log
      alpha_raw) * s without): d/d(log alpha_raw) = alpha * d/dalpha per pair while the clamp is off
      or slack, so its sums are alpha times the dalpha ones and its C, alpha cancelling in the ratio,
      equals the dalpha one (up to the ratio's epsilon); once the clamp holds (the raw parameter
      above ln 100, where clamp's backward blocks the gradient) the parameter's gradient is exactly
      zero however hard the loss pushes on the effective scale, and the whole family reads zero (sum,
      sum_abs, and C by the ratio's epsilon) while dalpha keeps reporting the pressure. The factor
      d(alpha)/d(log alpha_raw) is taken by autograd through the same clamp-then-exp path
      compute_logits applies, so exactly at the cap it goes whichever way the running torch's clamp
      backward breaks the tie (the gradient passes in 2.5 / 2.7, is blocked in 2.14).
    - the KL decomposition (the kl* strips): infonce_kl_terms, D_KL(y || p) = E_ir + E_s + E_sr
      per anchor, batch-meaned.
    - the row-wise target-implied scale bounds (the alpha panels' red lines): for row i, the smallest
      alpha whose logit range alpha * S over S in [-1, 1] spans Y_i as optimal logits log(Y_i) (up to
      a constant), 0.5 * log(max_j Y_ij / min_j Y_ij), reported as its min / mean / max over rows.
      Softmax feasibility is row-wise, so the batch's requirement is the max (a global max(Y) / min(Y)
      would pair extremes from different rows and overstate it); a row holding a zero sits at
      infinity. One statistic for both directions, which train against Y's rows alike. The
      log_alpha_req_* trio is the same three reductions taken over log(alpha_req) instead, for the
      logalpha panel, which plots the log parameter: the min and max are the logs of the alpha_req
      ones (log is monotone) but the mean is not -- it is the mean of the rows' logs, the log of
      their geometric mean.

    Y's rows are renormalized before any of it, the reachable-set solve needing sum(p*) = 1 to be the
    constraint it says it is. InfoNCECriterion._tsm already hands this path a float64 Y whose rows sum to
    1 to ~1e-16, so the correction is nil there; it is the guard for a float32 Y, whose rows sum to 1 only
    to ~1e-8. Left un-renormalized, that deficit is made up by lifting the floor lam, and the lift reports
    as residual -- sum|p* - y| comes out at the row-sum deficit itself, some 7 orders above the true
    residual at large alpha, where it decays like exp(-2 alpha). The correction sits below a float32 ulp,
    so it is inert for the loss itself, whose CE needs no normalized target.

    - sim, targs, y, logits -- the batch's BxB S, Q (the blended target matrix), Y (the blended target
      distribution) and logits (first branch), as passed to sim_targ_batch_stats
    - logit_scale ---------- the log logit-scale parameter, raw (pre-clamp) and detached
    - clamp ---------------- logits.scale.clamp: whether compute_logits caps the parameter at ln(100)

    Returns {{dalpha,dlogalpha}_{sum,sum_abs,C}_{full,struct,res,sres,ires}: [all, pos, neg]} plus
    {kl, kl_s, kl_ir, kl_sr: scalar} and {{alpha_req,log_alpha_req}_{min,mean,max}: scalar}; the reductions are stacked
    so the device->host transfer is a single .cpu() sync.
    """
    with torch.no_grad():
        S, Q, Y, Z = (t.detach().double() for t in (sim, targs, y, logits))
        Y = Y / Y.sum(dim=1, keepdim=True)  # rows to 1 in float64; see the renormalization note above
        with torch.enable_grad():
            # the scale the logits carry and its derivative in the raw parameter, d(alpha)/d(log alpha_raw),
            # by autograd through the clamp-then-exp path compute_logits takes: alpha while the clamp is
            # slack, zero once it holds (its backward blocks the gradient above the cap)
            log_alpha = torch.as_tensor(logit_scale, device=S.device).detach().requires_grad_(True)
            alpha = (log_alpha.clamp(max=math.log(100)) if clamp else log_alpha).exp()
            (dalpha_dlog,) = torch.autograd.grad(alpha, log_alpha)
        alpha, dalpha_dlog = alpha.detach().double(), dalpha_dlog.double()
        P_opt = infonce_p_opt(Y, alpha)
        S_opt = infonce_s_opt(P_opt, alpha)  # one geometry for both directions, as p* is
        log_P_i2t, log_P_t2i = torch.log_softmax(Z, dim=1), torch.log_softmax(Z.T, dim=1)
        sums = 0.5 * (infonce_scale_grad_sums(S, Q, Y, log_P_i2t.exp(), P_opt, S_opt)
                      + infonce_scale_grad_sums(S.T, Q.T, Y, log_P_t2i.exp(), P_opt, S_opt))
        kl = 0.5 * (infonce_kl_terms(Y, log_P_i2t, P_opt) + infonce_kl_terms(Y, log_P_t2i, P_opt))
        alpha_req = 0.5 * torch.log(Y.amax(1) / Y.amin(1))  # per row
        log_alpha_req = alpha_req.log()  # the logalpha panel's own bounds: the reductions over the logs
        bounds = torch.stack([alpha_req.min(), alpha_req.mean(), alpha_req.max(),
                              log_alpha_req.min(), log_alpha_req.mean(), log_alpha_req.max()])
        families = []
        for scaled in (sums, dalpha_dlog * sums):  # d/dalpha, then d/d(log alpha_raw)
            C = scaled[0].abs() / (scaled[1] + 1e-30)
            families.append(torch.cat([scaled, C[None]]))
        grad = torch.stack(families)  # [2, 3, 5, 3]
        packed = torch.cat([grad.flatten(), kl, bounds]).cpu()
        vals = packed[:grad.numel()].view_as(grad).tolist()
        kl_vals = packed[grad.numel():grad.numel() + kl.numel()].tolist()
        bound_vals = packed[grad.numel() + kl.numel():].tolist()
    return {
        **{
            f"{prefix}_{agg}_{comp}": vals[f][a][c]
            for f, prefix in enumerate(("dalpha", "dlogalpha"))
            for a, agg in enumerate(("sum", "sum_abs", "C"))
            for c, comp in enumerate(("full", "struct", "res", "sres", "ires"))
        },
        **dict(zip(("kl", "kl_s", "kl_ir", "kl_sr"), kl_vals)),
        **dict(zip(("alpha_req_min", "alpha_req_mean", "alpha_req_max",
                    "log_alpha_req_min", "log_alpha_req_mean", "log_alpha_req_max"), bound_vals)),
    }


class _SimTargStatsAccum:
    """
    Streams the batch's sim/target/probability distribution stats over the loss tiles so the chunked
    path can report the same batch_stats keys as sim_targ_batch_stats without holding
    the full BxB matrices. sim/targ min/max/mean are exact; their median is over a strided subsample
    of each tile (an exact BxB median would need the whole matrix). Targets and probabilities are
    also summarized as HIST_BINS histograms, which stream exactly -- counts just add across tiles
    and ranks. The mean hard-pair similarity margins (sim_margin*, one entry per hpsm_kappas
    value) are exact too. I2T (image anchors): every tile holds whole rows, so the per-row margins just
    sum across tiles and ranks. T2I (text anchors): a text's weights over the images span every tile
    and rank, so per column the four weighted sums behind its margin (positive / negative side, weight
    mass and weighted sim) stream in float64 with the exponents shifted to <= 0 (no overflow; the
    weights underflow only past kappa ~ 300) and the ratio is taken after the fold. The chunked path
    is BCE-family only, so p_hist is always reported here.
    """
    _FIELDS = ("sim", "targ")
    _HIST_FIELDS = ("targ", "p")

    def __init__(self, device, hpsm_kappas, B):
        self.mins = {f: torch.tensor(float("inf"), device=device) for f in self._FIELDS}
        self.maxs = {f: torch.tensor(float("-inf"), device=device) for f in self._FIELDS}
        self.sums = {f: torch.zeros((), dtype=torch.float64, device=device) for f in self._FIELDS}
        self.samps = {f: [] for f in self._FIELDS}
        self.hists = {f: torch.zeros(HIST_BINS, dtype=torch.float64, device=device) for f in self._HIST_FIELDS}
        self.count = 0
        self.kappas = hpsm_kappas
        self.B = B
        self.margin_sums = torch.zeros(len(hpsm_kappas), dtype=torch.float64, device=device)  # I2T per-row margins, summed
        # T2I per-column sums per kappa: [pos mass, pos weighted sim, neg mass, neg weighted sim] x B
        self.t2i_sums = torch.zeros(len(hpsm_kappas), 4, B, dtype=torch.float64, device=device)

    def update(self, sim_tile, targs_tile, logits_tile):
        tiles = {
            "sim": sim_tile.reshape(-1).float(),
            "targ": targs_tile.reshape(-1).float(),
            "p": logits_tile.reshape(-1).float().sigmoid(),
        }
        for field in self._FIELDS:
            vals = tiles[field]
            self.mins[field] = torch.minimum(self.mins[field], vals.min())
            self.maxs[field] = torch.maximum(self.maxs[field], vals.max())
            self.sums[field] += vals.double().sum()
            stride = max(1, vals.numel() // 4096)  # bound the median subsample per tile
            self.samps[field].append(vals[::stride])
        for field in self._HIST_FIELDS:
            self.hists[field] += torch.histc(tiles[field], bins=HIST_BINS, min=0.0, max=1.0).double()
        self.count += tiles["sim"].numel()
        S, Q = sim_tile.float(), targs_tile.float()
        self.margin_sums += torch.stack([hard_pair_similarity_margin(S, Q, kappa).double().sum() for kappa in self.kappas])
        S, Q = S.double(), Q.double()
        for idx_kappa, kappa in enumerate(self.kappas):
            W_pos = Q * torch.exp(-kappa * (S + 1.0))          # ∝ q exp(-kappa s), shifted by the s >= -1 bound
            W_neg = (1.0 - Q) * torch.exp(kappa * (S - 1.0))   # ∝ (1-q) exp(+kappa s), shifted by the s <= 1 bound
            self.t2i_sums[idx_kappa] += torch.stack([W_pos.sum(0), (W_pos * S).sum(0), W_neg.sum(0), (W_neg * S).sum(0)])

    def finalize(self, world_size):
        mins = dict(self.mins)
        maxs = dict(self.maxs)
        sums = dict(self.sums)
        samps = {f: torch.cat(self.samps[f]) for f in self._FIELDS}
        hists = dict(self.hists)
        count = self.count
        margin_sums = self.margin_sums
        t2i_sums = self.t2i_sums
        if world_size > 1:  # fold per-band partials; the bands partition the BxB rows exactly
            ext = torch.stack([*(-mins[f] for f in self._FIELDS), *(maxs[f] for f in self._FIELDS)])
            dist.all_reduce(ext, op=dist.ReduceOp.MAX)
            mins = {f: -ext[i] for i, f in enumerate(self._FIELDS)}
            maxs = {f: ext[len(self._FIELDS) + i] for i, f in enumerate(self._FIELDS)}
            # the scalar sums and margin accumulators ride along with every histogram's bins in one collective
            parts = [torch.stack([sums[f] for f in self._FIELDS]), margin_sums, t2i_sums.flatten(),
                     *(hists[f] for f in self._HIST_FIELDS)]
            packed = torch.cat(parts)
            dist.all_reduce(packed)
            sums_v, margin_sums, t2i_flat, *hists_v = torch.split(packed, [p.numel() for p in parts])
            sums = {f: sums_v[i] for i, f in enumerate(self._FIELDS)}
            t2i_sums = t2i_flat.view_as(t2i_sums)
            hists = dict(zip(self._HIST_FIELDS, hists_v))
            count *= world_size  # equal bands -> equal per-rank counts
            # median subsamples: equal bands + equal tile sizes -> equal lengths on every rank, so a
            # plain all_gather reassembles the exact same subsample pool a single full sweep produces
            samp = torch.stack([samps[f] for f in self._FIELDS])
            parts = [torch.empty_like(samp) for _ in range(world_size)]
            dist.all_gather(parts, samp)
            samps = {f: torch.cat([p[i] for p in parts]) for i, f in enumerate(self._FIELDS)}
        stats = {}
        for field in self._FIELDS:
            stats[f"{field}_min"] = mins[field].item()
            stats[f"{field}_max"] = maxs[field].item()
            stats[f"{field}_median"] = samps[field].median().item()
            stats[f"{field}_mean"] = (sums[field] / count).item()
        for field in self._HIST_FIELDS:
            stats[f"{field}_hist"] = (hists[field] / hists[field].sum()).tolist()
        i2t = margin_sums / self.B
        pos_mass, pos_sim, neg_mass, neg_sim = t2i_sums.unbind(1)  # [kappas, B] each
        t2i = (pos_sim / pos_mass - neg_sim / neg_mass).mean(1)
        stats["sim_margin_i2t"] = i2t.tolist()
        stats["sim_margin_t2i"] = t2i.tolist()
        stats["sim_margin"] = (0.5 * (i2t + t2i)).tolist()
        return stats


def _crit_block_weight_bce(crit, logits_f, targs, class_encs_rows, class_encs_cols, B, consts):
    """
    For one criterion and one [C, B] row-block: the aggregated per-pair weight W (differentiable via the
    focal factor) and the raw BCE matrix. `consts` carries the precomputed detached global scalars
    (cls_imb_mean, dsmr_mass); a None entry means that normalizer is off. Mirrors the
    train-mode weighting of BCECriterion.__call__ tile-by-tile via the shared _dsmr_weight.
    """
    cfg_wting = crit.cfg["wting"]

    W_ci = compute_cls_imb_wts(
        cfg_wting["cls_imb"], 
        crit.counts, 
        class_encs_rows, 
        crit.wting_dim,
        crit.wt_mean, 
        crit.batch_size, 
        class_encs_cols=class_encs_cols
    )
    if consts["cls_imb_mean"] is not None:
        W_ci = W_ci / consts["cls_imb_mean"]

    W_foc = crit._focal_2d(logits_f, targs)

    W = W_ci * W_foc
    if cfg_wting["bce"]["dsmr"]:
        W = W * _dsmr_weight(targs, *consts["dsmr_mass"], B)

    bce = F.binary_cross_entropy_with_logits(logits_f, targs, reduction="none")

    return W, bce

def _bif_branches(img, txt):
    """
    The two bifurcated branches as (rows_live, cols) frames: i2t = (img rows, detached txt), t2i =
    (txt rows, detached img). Each branch's tile grads flow into its live rows only -- the tiled
    analogue of the full-batch opposite-tower detaches (_loss_full_batch).
    """
    return ((img, txt.detach()), (txt, img.detach()))

def _bif_block_invariants(crit, targs, B):
    """
    Branch-invariant per-block weighting factors (Y symmetric, so both branch frames share them,
    as the full-batch path computes them once): the row-wise DSMR weights and the targ_mass_neut
    row masses; None when the corresponding toggle is off.
    """
    W_dsmr = _dsmr_weight_rows(targs, B) if crit.cfg["wting"]["bce"]["dsmr"] else None
    neut_mass = targs.sum(dim=1) if crit.cfg["bce"]["targ_mass_neut"] else None
    return W_dsmr, neut_mass

def _bif_block_num_raw(crit, logits_f, targs, w_rows, W_dsmr, neut_mass):
    """
    For one bifurcated-branch [C, B] row-block (rows = the branch's anchors): the branch's weighted
    per-anchor loss sum `num` and the raw BCE matrix. Mirrors the train-mode weighting of
    BifurcatedBCECriterion.__call__ tile-by-tile: 1D per-anchor row weights `w_rows` (the consts'
    precomputed normalized w_ci, sliced to this block), focal, and the precomputed block invariants
    (_bif_block_invariants).
    """
    W = w_rows[:, None] * crit._focal_2d(logits_f, targs)
    if W_dsmr is not None:
        W = W * W_dsmr
    bce = F.binary_cross_entropy_with_logits(logits_f, targs, reduction="none")
    num_rows = (W * bce).sum(dim=1)
    if neut_mass is not None:
        num_rows = num_rows / neut_mass
    return num_rows.sum(), bce

def n_scalar_pairs(crit):
    """The logit-scalar pairs the criterion's logits are built on: one per loss term under separate logit
    scalars (Criterion.sep_scalars; pair g is compute_logits' secondary=bool(g)), else the one shared."""
    return len(crit.targ_specs) if crit.sep_scalars else 1

def _crit_block_logits_f(crit, rows, cols, compute_logits, center, center_globals, half_live=False):
    """
    [C, B] similarity tile and its float32 logits tiles for the criterion (its sim_type + logit
    scale/bias) -- one logits tile per logit-scalar pair (n_scalar_pairs): the one shared pair's, or
    under separate logit scalars each loss term's own (compute_logits' secondary), all on the one sim
    tile. `rows`/`cols` are the tile's anchor rows and full column embeddings -- (img rows, txt)
    non-bifurcated; a bifurcated branch passes its own (rows_live, cols) frame (_bif_branches) with
    half_live=True so the un-halved branch sum carries 1x logit scale/bias grads.

    `center`/`center_globals` (one per scalar pair) are passed through to compute_logits: the chunked
    sweeps supply the full-batch quantity (in-graph global sim mean for "sim"; detached full-batch
    incoming-grad mean for grad_proj/grad_proj2) so tiles reproduce full-batch centering exactly, never
    the per-tile mean.
    """
    sim_block = compute_sim(rows, cols, crit.cfg["sim"])
    clamp = crit.cfg["logits"]["scale"]["clamp"]
    logits_blocks = [
        compute_logits(sim_block, clamp, center, center_global=center_global, half_live=half_live, secondary=bool(g)).float()
        for g, center_global in enumerate(center_globals)
    ]
    return sim_block, logits_blocks

def _precompute_crit_consts(crit, terms, class_encs_b, B, chunk_size, device, lo, hi, world_size):
    """
    Detached global constants for the criterion (see module header), one consts dict per loss term. For
    bce, cls_imb_mean (mean of W_ci, shared by the terms) and the term's dsmr_mass are embedding-free;
    for bif_bce the consts are just the full normalized 1D per-anchor weight vector (O(B), built outright
    -- its cls_imb.norm mean is over B values, and row-wise DSMR needs no global mass), so the consts
    dicts are structurally distinct and a bif crit can never reach the 2D _crit_block_weight_bce. The
    sweeps cover only this rank's row-band [lo, hi); the partial sums are all-reduced so every rank
    derives identical constants.
    `terms` are the loss terms' (weight, spec_fns) pairs (loss_term_spec_fns), a term's spec_fns read by
    its DSMR mass.
    Returns the terms' consts dicts for _crit_block_weight_bce / _bif_block_num_raw.
    """
    cfg_w = crit.cfg["wting"]

    if crit.bifurcated:
        # full [B] per-anchor weight vector, incl. the cls_imb.norm batch-mean division -- exactly
        # as BifurcatedBCECriterion.__call__ computes it (rank-identical, no collective)
        w_ci = compute_cls_imb_wts(cfg_w["cls_imb"], crit.counts, class_encs_b, crit.wting_dim, crit.wt_mean, crit.batch_size)
        if cfg_w["cls_imb"]["norm"]:
            w_ci = w_ci / w_ci.mean()
        return [{"w_ci": w_ci} for _ in terms]

    cls_imb_mean = None
    if cfg_w["cls_imb"]["norm"]:  # mean of W_ci over BxB -- embedding-free, tiled to stay O(C*B)
        s = torch.zeros((), dtype=torch.float64, device=device)
        for rs in range(lo, hi, chunk_size):
            W_ci = compute_cls_imb_wts(cfg_w["cls_imb"], crit.counts, class_encs_b[rs:rs + chunk_size],
                                       crit.wting_dim, crit.wt_mean, crit.batch_size, class_encs_cols=class_encs_b)
            s += W_ci.double().sum()
        if world_size > 1:
            dist.all_reduce(s)
        cls_imb_mean = (s / (B * B)).float()

    return [
        {"cls_imb_mean": cls_imb_mean,
         "dsmr_mass": bce_dsmr_mass(term_spec_fns, class_encs_b, B, chunk_size, lo, hi, world_size) if cfg_w["bce"]["dsmr"] else None}
        for _, term_spec_fns in terms
    ]

def _block_term_nums(crit, logits_fs, term_blocks, term_invs, consts_list, rs, re, class_encs_b, B):
    """
    For one block's [C, B] logits tiles over rows rs:re (_crit_block_logits_f; a bifurcated branch's tiles
    in its own anchor frame): each loss term's weighted loss numerator and detached raw BCE sum against
    the term's target tile -- on the term's own logits tile under separate logit scalars, else the one
    shared -- under the term's consts and -- bifurcated -- its block invariants (_bif_block_invariants).
    """
    nums, raws = [], []
    logits_terms = logits_fs if crit.sep_scalars else logits_fs * len(term_blocks)
    for logits_f, targs, invs, consts in zip(logits_terms, term_blocks, term_invs, consts_list):
        if crit.bifurcated:
            num, bce = _bif_block_num_raw(crit, logits_f, targs, consts["w_ci"][rs:re], *invs)
        else:
            W, bce = _crit_block_weight_bce(crit, logits_f, targs, class_encs_b[rs:re], class_encs_b, B, consts)
            num = (W * bce).sum()
        nums.append(num)
        raws.append(bce.sum().detach().double())
    return nums, raws

def _term_loss_values(crit, img, txt, term_fns, consts_list, class_encs_b, B, compute_logits, chunk_size,
                      autocast_ctx, lo, hi, world_size, device):
    """
    loss.unitless: each loss term's full weighted loss L_k, detached -- the magnitudes term_coeffs
    normalizes by, which the grad sweep needs before its first tile. A no_grad band sweep of the forward
    (one logits tile per block / branch serves every term sharing its scalars), all-reduced across bands. "sim" centering is
    reproduced through the full-batch sim mean (the cos mean factorization -- the same value in every
    branch frame); grad_proj* leave the forward untouched.
    """
    center = crit.cfg["logits"]["bce"]["center"]
    branches = _bif_branches(img, txt) if crit.bifurcated else ((img, txt),)
    sums = torch.zeros(len(term_fns), dtype=torch.float64, device=device)
    with torch.no_grad():
        cgs = [torch.dot(img.mean(0), txt.mean(0)) if center == "sim" else None] * n_scalar_pairs(crit)
        for rs in range(lo, hi, chunk_size):
            re = rs + chunk_size
            with autocast_ctx():
                term_blocks = [term_fn(rs, re) for term_fn in term_fns]
                term_invs = [_bif_block_invariants(crit, targs, B) if crit.bifurcated else None for targs in term_blocks]
                for rows_live, cols in branches:
                    _, logits_fs = _crit_block_logits_f(crit, rows_live[rs:re], cols, compute_logits, center, cgs, half_live=crit.bifurcated)
                    nums, _ = _block_term_nums(crit, logits_fs, term_blocks, term_invs, consts_list, rs, re, class_encs_b, B)
                    sums += torch.stack(nums).double()
    if world_size > 1:
        dist.all_reduce(sums)
    return list((sums / B).float())

def _crit_center_grad_mean(crit, img, txt, term_fns, coeffs, class_encs_b, B, consts_list,
                           compute_logits, chunk_size, autocast_ctx, lo, hi, world_size, device):
    """
    grad_proj/grad_proj2: the detached full-batch mean of the incoming gradient at the criterion's
    projection node, so every tile of the grad sweep subtracts the same constant the full-batch
    _ZeroSumGrad projection would (a per-tile mean would be wrong). Sweeps this rank's band with
    tile-local autograd (the sim tile as leaf, embeddings detached: graphs stay O(C*B), no param
    .grad touched) over the blended loss -- the terms' numerators under their blend coefficients
    `coeffs` (Criterion.term_coeffs) -- which captures the focal-weight gradient terms exactly; band
    partials are all-reduced. Returns the constant at the node the mode projects: the sim node for grad_proj
    (e^t folded in by autograd), the scaled-sim node for grad_proj2 (the post-clamp e^t divided
    back out, recovered as a probe gradient through compute_logits -- half_live changes no values
    and no dL/dsim, so the plain probe holds for bifurcated branches too).

    One constant per logit-scalar pair (n_scalar_pairs), returned as a list: every compute_logits call
    projects at its own node, which under separate logit scalars sees only its own term's incoming grad
    (under grad_proj2 at its own e^t). A bifurcated criterion gets a per-branch (i2t, t2i) tuple instead
    of a scalar: each branch's projection node sees only its own incoming grad (the branches join only
    at the scalar loss), and the two frames' grad means differ under per-anchor row weighting, so a
    shared constant would be silently wrong.
    """
    clamp = crit.cfg["logits"]["scale"]["clamp"]
    branches = _bif_branches(img, txt) if crit.bifurcated else ((img, txt),)
    n_pairs = n_scalar_pairs(crit)
    g_sums = torch.zeros(n_pairs, len(branches), dtype=torch.float64, device=device)
    for rs in range(lo, hi, chunk_size):
        re = rs + chunk_size
        tile_losses, sim_leaves = [], []
        with autocast_ctx():
            term_blocks = [term_fn(rs, re) for term_fn in term_fns]
            term_invs = [_bif_block_invariants(crit, targs, B) if crit.bifurcated else None for targs in term_blocks]
            for rows_live, cols in branches:
                sim_leaf = compute_sim(rows_live[rs:re].detach(), cols.detach(), crit.cfg["sim"]).requires_grad_(True)
                logits_fs = [compute_logits(sim_leaf, clamp, None, secondary=bool(g)).float() for g in range(n_pairs)]
                nums, _ = _block_term_nums(crit, logits_fs, term_blocks, term_invs, consts_list, rs, re, class_encs_b, B)
                wnums = [c * num / B for c, num in zip(coeffs, nums)]
                tile_losses.append(wnums if crit.sep_scalars else [sum(wnums)])  # the loss through each pair's logits
                sim_leaves.append(sim_leaf)
        for j, (pair_losses, sim_leaf) in enumerate(zip(tile_losses, sim_leaves)):
            for g, pair_loss in enumerate(pair_losses):
                g_sums[g, j] += torch.autograd.grad(pair_loss, sim_leaf)[0].double().sum()
    if world_size > 1:
        dist.all_reduce(g_sums)
    cs = (g_sums / (B * B)).float()  # mean incoming grad at the sim node(s)
    if crit.cfg["logits"]["bce"]["center"] == "grad_proj2":
        # the scaled-sim node's grad = (sim node's grad) / e^t; recover each pair's post-clamp e^t as a probe gradient
        for g in range(n_pairs):
            probe = torch.zeros(1, 1, device=device, requires_grad=True)
            e_det = torch.autograd.grad(compute_logits(probe, clamp, None, secondary=bool(g)).sum(), probe)[0].reshape(()).detach()
            cs[g] = cs[g] / e_det
    cs = cs.detach()
    return [tuple(cs_g) if crit.bifurcated else cs_g[0] for cs_g in cs]

def _gsum_hook(acc):
    """Tensor backward hook accumulating the incoming grad's sum into `acc`. Returns None so the
    gradient itself passes through untouched (a non-None return would replace it)."""
    def hook(g):
        acc.add_(g.double().sum())
    return hook

def chunked_bce_loss_backward(img, txt, class_encs_b, targ_data_b, crit, compute_logits, chunk_size, mixed_prec,
                              device, rank, world_size, sim_grad_sums=True, sim_targ_stats=True, hpsm_kappas=(0.0,)):
    """
    Tiled + row-band-sharded global-batch BCE-family loss + backward (GradCache-style representation
    gradients). Computes the exact same weighted loss and gradients as the full-batch path
    (BCECriterion.__call__ / BifurcatedBCECriterion.__call__ via _global_batch_loss) over the full BxB
    matrix, but never materializes it and shares the work across ranks: the BxB rows split into
    world_size equal bands (SigLIP-style decomposition), this rank sums only its band [rank*b, (rank+1)*b)
    over row-blocks of C x B (b an exact multiple of C = chunk_size), and each block's gradient is
    backpropagated into the embedding leaves as computed, so peak VRAM is O(C*B) and per-rank compute is
    O(B^2/world_size). The loss/raw totals and batch stats are all-reduced, so every rank returns
    identical full-batch values; the leaves' .grad hold this band's PARTIAL dL/dembs, which sum to the
    full gradient across ranks (the caller completes them -- see batch_step_chunked). Exact up to
    floating-point summation order.

    The loss is the criterion's blend of loss terms (loss_term_spec_fns, the tiled Criterion.loss_terms):
    under loss.blend.type targ the one loss against the blended target tiles (blend_targ_block_fn: sum_k
    w_k Q_k over the live target specs, as the full-batch targ_memb / targ_dist), under loss.blend.type
    loss each spec's own loss weighted w_k -- every term on the block's one logits tile, or each on its
    own scalar pair's tile under separate logit scalars (Criterion.sep_scalars). All
    cross-tile-coupled normalizers are precomputed detached constants (_precompute_crit_consts, and under
    loss.unitless the terms' loss magnitudes -- _term_loss_values), so the backward is single-pass.

    - img, txt --------- detached [B, D] embedding leaves (requires_grad); receive band-partial dL/dembs
                         in their .grad. A bifurcated criterion sweeps two branch tiles per block
                         (module header), routing each branch's grads into one leaf.
    - crit ------------- the BCECriterion / BifurcatedBCECriterion.
    - compute_logits --- VLMWrapper.compute_logits(sim, clamp, center, center_global, half_live, secondary)
                         -> logits tile; center_global carries the precomputed full-batch centering
                         quantity (module header) so tiled centering is exact, secondary selects the
                         second logit-scalar pair (separate logit scalars: one logits tile per loss term).
    - rank, world_size - this rank's band index / number of bands (1 -> unsharded full sweep).
    - sim_grad_sums ----- False skips the sim-grad-sum hooks and returns grad_sum_sim None.
    - sim_targ_stats ---- False skips the per-tile stats accumulation and returns batch_stats None.
    - hpsm_kappas ------- kappa values the sim_margin* stats are reported at (one entry each).

    Returns (loss, loss_raw, batch_stats, grad_sum_sim), all detached; gradients left in the leaves' /
    params' .grad. grad_sum_sim = sum(dL/dsim), the full-batch sum accumulated tile-by-tile via backward
    hooks (all-reduced across bands) -- the same value the full-batch path reads off the retained sim
    grads.
    """
    B = img.size(0)
    b = B // world_size
    # checking that the BxB rows split into world_size equal bands of whole chunk_size-row blocks:
    # ragged bands would silently double-count rows across ranks (wrong gradients, no error)
    if b * world_size != B or b % chunk_size != 0:
        raise ValueError(
            f"global batch ({B}) must split into world_size ({world_size}) equal row-bands, each an exact "
            f"multiple of hardware.loss_chunk_size ({chunk_size}); got band size {b}"
        )
    lo, hi = rank * b, (rank + 1) * b

    def autocast_ctx():
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16) if mixed_prec else nullcontext()

    center = crit.cfg["logits"]["bce"]["center"]
    # config rejects this combo too (TrainConfig); guard direct callers against a silently-wrong tile mean
    assert not (center == "sim" and crit.cfg["sim"] != "cos"), "center: sim under chunking requires cos sim"

    terms = loss_term_spec_fns(crit, make_targ_block_fns(crit, class_encs_b, targ_data_b, B, device))
    term_ws = [w for w, _ in terms]
    term_fns = [blend_targ_block_fn(term_spec_fns) for _, term_spec_fns in terms]
    consts_list = _precompute_crit_consts(crit, terms, class_encs_b, B, chunk_size, device, lo, hi, world_size)
    L_values = (
        _term_loss_values(crit, img, txt, term_fns, consts_list, class_encs_b, B, compute_logits, chunk_size,
                          autocast_ctx, lo, hi, world_size, device)
        if crit.cfg["unitless"] else None
    )
    coeffs = crit.term_coeffs(term_ws, L_values)

    # grad_proj*: the detached full-batch incoming-grad mean -- the constant every tile of the grad sweep
    # subtracts in place of the full path's g.mean() (_ZeroSumGradConst)
    center_const = (
        _crit_center_grad_mean(crit, img, txt, term_fns, coeffs, class_encs_b, B, consts_list, compute_logits,
                               chunk_size, autocast_ctx, lo, hi, world_size, device)
        if center in ("grad_proj", "grad_proj2") else None
    )

    # sim-grad-sum metric (learning-curve strip): accumulated tile-by-tile via backward hooks. Under
    # "sim" centering part of the full-path dL/dsim routes through the global mean, which here lives
    # on the leaves and bypasses the tiles -- a hook on the in-graph mean captures it (dm/dsim sums
    # to exactly 1), so the folded total matches the full-batch sim.grad.sum().
    grad_sum = torch.zeros((), dtype=torch.float64, device=device)

    wbce_tot = torch.zeros((), dtype=torch.float64, device=device)
    raw_tot = torch.zeros((), dtype=torch.float64, device=device)
    stats = _SimTargStatsAccum(device, hpsm_kappas, B)
    n_pairs = n_scalar_pairs(crit)

    for rs in range(lo, hi, chunk_size):
        re = rs + chunk_size  # the band is an exact multiple of chunk_size (checked above)
        with autocast_ctx():
            term_blocks = [term_fn(rs, re) for term_fn in term_fns]
            term_invs = [_bif_block_invariants(crit, targs, B) if crit.bifurcated else None for targs in term_blocks]
            if crit.bifurcated:
                num = 0.0
                for j, (rows_live, cols) in enumerate(_bif_branches(img, txt)):
                    cgs = [cs_g[j] for cs_g in center_const] if center_const is not None else [None] * n_pairs
                    if center == "sim":
                        # per-branch in-graph mean: the centering's backward routes into the
                        # branch's live tower only, mirroring the branch's detach pattern
                        cg = torch.dot(rows_live.mean(0), cols.mean(0))
                        if sim_grad_sums and cg.requires_grad:
                            cg.register_hook(_gsum_hook(grad_sum))
                        cgs = [cg] * n_pairs
                    sim_block, logits_fs = _crit_block_logits_f(crit, rows_live[rs:re], cols, compute_logits, center, cgs, half_live=True)
                    if sim_grad_sums and sim_block.requires_grad:
                        sim_block.register_hook(_gsum_hook(grad_sum))
                    nums, raws = _block_term_nums(crit, logits_fs, term_blocks, term_invs, consts_list, rs, re, class_encs_b, B)
                    num = num + sum(c * b_num for c, b_num in zip(coeffs, nums))  # branches summed un-halved (full-batch: mean1 + mean2 = (sum1 + sum2)/B)
                    raw_tot += sum(w * raw for w, raw in zip(term_ws, raws))
                    if j == 0:
                        sim_block_stat = sim_block.detach()  # i2t frame -- values match the non-bif sim
                        logits_block_stat = logits_fs[0].detach()  # the primary pair's, as the full-batch stats read
            else:
                cgs = center_const if center_const is not None else [None] * n_pairs
                if center == "sim":
                    # in-graph per block: the centering's backward routes through the mean embeddings
                    # into every leaf row, completing the exact full-batch projection across blocks
                    cg = torch.dot(img.mean(0), txt.mean(0))
                    if sim_grad_sums and cg.requires_grad:
                        cg.register_hook(_gsum_hook(grad_sum))
                    cgs = [cg] * n_pairs
                sim_block, logits_fs = _crit_block_logits_f(crit, img[rs:re], txt, compute_logits, center, cgs)
                if sim_grad_sums and sim_block.requires_grad:
                    sim_block.register_hook(_gsum_hook(grad_sum))
                nums, raws = _block_term_nums(crit, logits_fs, term_blocks, term_invs, consts_list, rs, re, class_encs_b, B)
                num = sum(c * b_num for c, b_num in zip(coeffs, nums))
                raw_tot += sum(w * raw for w, raw in zip(term_ws, raws))
                sim_block_stat = sim_block.detach()
                logits_block_stat = logits_fs[0].detach()  # the primary pair's, as the full-batch stats read
            block_loss = num / B
            wbce_tot += num.detach().double()
        block_loss.backward()
        if sim_targ_stats:
            # the blended target tile (targ_memb): the terms' tiles under their weights, either blend type
            stats.update(sim_block_stat, sum(w * targs for w, targs in zip(term_ws, term_blocks)), logits_block_stat)

    if world_size > 1:  # fold the band-partial loss totals; the leaves' .grad stay band-partial
        packed = torch.stack([wbce_tot, raw_tot, grad_sum])
        dist.all_reduce(packed)
        wbce_tot, raw_tot, grad_sum = packed.unbind()

    loss = (wbce_tot / B).float()
    loss_raw = (raw_tot / B).float()
    grad_sum_sim = grad_sum.item() if sim_grad_sums else None
    batch_stats = stats.finalize(world_size) if sim_targ_stats else None
    if batch_stats is not None and crit.lambda_eff is not None:  # rank-identical: the term magnitudes are all-reduced
        batch_stats["lambda_eff"] = crit.lambda_eff.item()
    return loss, loss_raw, batch_stats, grad_sum_sim
