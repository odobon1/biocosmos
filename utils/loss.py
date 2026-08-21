import torch
import torch.nn.functional as F
import torch.distributed as dist
import abc
from contextlib import nullcontext
import math

from utils.rank_encs import compute_rank_dists
from utils.phylo import PhyloVCV
from utils.imb import build_wting, compute_cls_imb_wts
from utils.head import compute_sim

import pdb


_phylo_vcv_cache: dict[tuple, PhyloVCV] = {}
_htarg_shuf: bool = False
_phylo_seed: int | None = None


def configure_htarg_shuf(htarg_shuf: bool, seed: int | None) -> None:
    """Set phylo-target shuffling for this run; call once at setup before any loss is computed."""
    global _htarg_shuf, _phylo_seed
    _htarg_shuf = htarg_shuf
    _phylo_seed = seed

def get_phylo_vcv(dataset: str) -> PhyloVCV:
    key = (dataset, _htarg_shuf, _phylo_seed)
    if key not in _phylo_vcv_cache:
        _phylo_vcv_cache[key] = PhyloVCV(dataset=dataset, htarg_shuf=_htarg_shuf, seed=_phylo_seed)
    return _phylo_vcv_cache[key]

def compute_targets(targ_type, batch_size, class_encs_b, targ_data_b, device):
    if targ_type == "iw":
        targs = compute_targs_iw(batch_size)
    elif targ_type == "sw":
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

class Criterion(abc.ABC):
    """
    A loss paired with the class-imbalance weighting it consumes. The weighting dimensionality is a
    property of the loss (`wting_dim`) -- 1D per-class weights for InfoNCE and bifurcated BCE
    (per-anchor), 2D per-class-pair weights for BCE.

    Only the class counts and the normalization scalar are held; batch weights are computed from
    them on the fly, so no n_classes (1D) / n_classes^2 (2D) weight buffer persists for the run.
    """

    wting_dim: int
    bifurcated = False  # True -> consumes the (i2t, t2i) branch logits pair (see BifurcatedBCECriterion)

    def __init__(self, cfg_loss, dataset, split, train_pt, device, batch_size):
        self.cfg = cfg_loss
        self.device = device
        self.batch_size = batch_size
        counts, self.wt_mean = build_wting(cfg_loss["wting"]["cls_imb"], dataset, split, train_pt, self.wting_dim, batch_size)
        self.counts = counts.to(device)

    @staticmethod
    def build(cfg_loss, dataset, split, train_pt, device, batch_size):
        crit_cls = {
            "infonce": InfoNCECriterion,
            "bce":     BCECriterion,
            "bif_bce": BifurcatedBCECriterion,
        }[cfg_loss["crit"]]

        return crit_cls(cfg_loss, dataset, split, train_pt, device, batch_size)

    def _targets(self, batch_size, class_encs_b, targ_data_b):
        return compute_targets(self.cfg["targ"], batch_size, class_encs_b, targ_data_b, self.device)

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
        [img-row, txt-col]. `logit_scale` is this criterion's learnable log logit scale param
        (model.logit_scale for loss1, model.logit_scale2 for loss2), raw (pre-clamp).

        Returns:
        - loss ------- Weighted scalar loss (== loss_raw when not training)
        - loss_raw --- Unweighted scalar loss
        - targs ------ Target matrix; pt[B, B]
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

    def __call__(self, logits, class_encs_b, targ_data_b, train, logit_scale):
        B = logits.size(0)
        
        Y = self._targets(B, class_encs_b, targ_data_b)  # pt[B, B]
        Y_mass = Y.sum(dim=1)

        if self.cfg["infonce"]["tsm"]["type"] == "linear":
            Y_scaled = Y / Y_mass[:, None]  # pt[B, B]; for MP + HCon (note: symmetrical for MP, non-symmetrical for HCon)
        elif self.cfg["infonce"]["tsm"]["type"] == "softmax":
            tau_Y = self.cfg["infonce"]["tsm"]["sm_temp"]
            if tau_Y == "pinned":
                if self.cfg["logits"]["temp"]["clamp"]:
                    logit_scale = logit_scale.clamp(max=math.log(100))
                Y_scaled = F.softmax(2 * Y * torch.exp(logit_scale), dim=1)  # pt[B, B]; for HCon (note: symmetrical for MP, non-symmetrical for HCon)
            else:
                Y_scaled = F.softmax(2 * Y / tau_Y, dim=1)  # pt[B, B]; for MP + HCon (note: symmetrical for MP, non-symmetrical for HCon)

        loss_i2t_raw = -Y_scaled * F.log_softmax(logits,   dim=1)  # pt[B, B]
        loss_t2i_raw = -Y_scaled * F.log_softmax(logits.T, dim=1)  # pt[B, B]

        # per-anchor CE (row sum) averaged over anchors -- CLIP's /B, the scale the weighted loss carries
        loss_raw = 0.5 * (loss_i2t_raw.sum(dim=1).mean() + loss_t2i_raw.sum(dim=1).mean())

        if not train:
            return loss_raw, loss_raw, Y

        W_ci = self._cls_imb_wts(class_encs_b)  # class-imbalance weights; pt[B]
        if self.cfg["wting"]["cls_imb"]["norm"]:
            W_ci = W_ci / W_ci.mean()  # pt[B]

        # Note: 2D-focal is still used despite 1D class-imbalance weighting (reduces to standard focal loss in the SP setting)
        W_foc_i2t = self._focal_2d(logits,   Y_scaled)  # pt[B, B]
        W_foc_t2i = self._focal_2d(logits.T, Y_scaled)  # pt[B, B]

        W_i2t = W_foc_i2t * W_ci[:, None]  # pt[B, B]
        W_t2i = W_foc_t2i * W_ci[:, None]  # pt[B, B]

        loss_i2t = (W_i2t * loss_i2t_raw).sum(dim=1)
        loss_t2i = (W_t2i * loss_t2i_raw).sum(dim=1)

        loss = 0.5 * (loss_i2t.mean() + loss_t2i.mean())

        return loss, loss_raw, Y

class BCECriterion(Criterion):
    """
    Sigmoid BCE weighted by 2D per-class-pair weights.
    """

    wting_dim = 2

    def _preds(self, Z):
        return torch.sigmoid(Z)

    def __call__(self, logits, class_encs_b, targ_data_b, train, logit_scale):
        B = logits.size(0)

        Y = self._targets(B, class_encs_b, targ_data_b)  # pt[B, B]

        # fp32: cos-path logits are bf16 under autocast, where sigmoid saturates to exactly 1.0 at
        # |logit| >~ 6 (zeroing focal weights on the easy set, quantizing the rest); upcast once and
        # reuse for both focal preds and the BCE loss. No-op when logits are already fp32 (geo sim).
        logits_f = logits.float()

        # Unweighted loss matrix
        loss_2d_raw = F.binary_cross_entropy_with_logits(logits_f, Y, reduction="none")  # pt[B, B]
        loss_raw = loss_2d_raw.detach().sum() / B

        if not train:
            return loss_raw, loss_raw, Y

        W_ci = self._cls_imb_wts(class_encs_b)  # class-imbalance weights; pt[B, B]
        if self.cfg["wting"]["cls_imb"]["norm"]:
            W_ci = W_ci / W_ci.mean()
        W = W_ci * self._focal_2d(logits_f, Y)  # pt[B, B]
        if self.cfg["wting"]["bce"]["dsmr"]:
            mass_pos = torch.sum(Y)
            mass_neg = torch.sum(1.0 - Y)
            W = W * _dsmr_weight(Y, mass_pos, mass_neg, B)

        loss = (W * loss_2d_raw).sum() / B

        return loss, loss_raw, Y

class BifurcatedBCECriterion(Criterion):
    """
    Sigmoid BCE with a per-direction anchor branch, weighted by 1D per-class (per-anchor) weights.

    Bifurcated: `logits` is the branch pair (logits_bif_i2t, logits_bif_t2i), both [img-row,
    txt-col]; the t2i branch is consumed as its transposed [txt-row, img-col] view, so each
    direction's anchors are rows and per-anchor weighting/reduction is row-wise in both branches
    (pairing the transposed view against Y relies on Y being symmetric, which holds for every
    targ type). The i2t branch backprops into the image tower only (txt detached upstream), t2i
    into the text tower only. The un-halved branch sum makes the loss value (and loss_raw) 2x the
    non-bifurcated reading, but with identical weighting on both branches every gradient matches
    non-bifurcated 1x: towers live in one branch each, and the logit scale/bias are half-live
    upstream (compute_logits) so their two branch contributions sum to 1x. A loss2 unit-scale
    blend accordingly normalizes this loss by L/2 -- its gradient-scale-equivalent value -- so
    `mix` keeps the same gradient ratio as with an equivalent non-bifurcated loss
    (_global_batch_loss).
    """

    wting_dim = 1
    bifurcated = True

    def _preds(self, Z):
        return torch.sigmoid(Z)

    def __call__(self, logits, class_encs_b, targ_data_b, train, logit_scale):
        logits_i2t = logits[0]
        logits_t2i = logits[1].T  # anchors as rows

        B = logits_i2t.size(0)

        Y = self._targets(B, class_encs_b, targ_data_b)  # pt[B, B]

        # fp32: cos-path logits are bf16 under autocast, where sigmoid saturates to exactly 1.0 at
        # |logit| >~ 6 (zeroing focal weights on the easy set, quantizing the rest); upcast once and
        # reuse for both focal preds and the BCE loss. No-op when logits are already fp32 (geo sim).
        logits_i2t_f = logits_i2t.float()
        logits_t2i_f = logits_t2i.float()

        # Unweighted loss matrices
        loss_i2t_2d_raw = F.binary_cross_entropy_with_logits(logits_i2t_f, Y, reduction="none")  # pt[B, B]
        loss_t2i_2d_raw = F.binary_cross_entropy_with_logits(logits_t2i_f, Y, reduction="none")  # pt[B, B]
        loss_raw = loss_i2t_2d_raw.detach().sum() / B + loss_t2i_2d_raw.detach().sum() / B

        if not train:
            return loss_raw, loss_raw, Y

        # per-anchor weights are row weights in both branches (rows = each direction's anchors;
        # paired image/text share the class, so w_ci indexes both directions)
        w_ci = self._cls_imb_wts(class_encs_b)  # class-imbalance weights; pt[B]
        if self.cfg["wting"]["cls_imb"]["norm"]:
            w_ci = w_ci / w_ci.mean()
        W_i2t = w_ci[:, None] * self._focal_2d(logits_i2t_f, Y)  # pt[B, B]
        W_t2i = w_ci[:, None] * self._focal_2d(logits_t2i_f, Y)  # pt[B, B]
        if self.cfg["wting"]["bce"]["dsmr"]:
            W_dsmr = _dsmr_weight_rows(Y, B)  # pt[B, B]
            W_i2t = W_i2t * W_dsmr
            W_t2i = W_t2i * W_dsmr

        # 2D --> 1D: per-anchor (row) reduction
        loss_i2t_1d = (W_i2t * loss_i2t_2d_raw).sum(dim=1)  # pt[B]
        loss_t2i_1d = (W_t2i * loss_t2i_2d_raw).sum(dim=1)  # pt[B]

        if self.cfg["bce"]["targ_mass_neut"]:
            Y_mass = Y.sum(dim=1)  # pt[B]
            loss_i2t_1d = loss_i2t_1d / Y_mass
            loss_t2i_1d = loss_t2i_1d / Y_mass

        # Batch-mean over anchors, branches summed un-halved
        loss = loss_i2t_1d.mean() + loss_t2i_1d.mean()

        return loss, loss_raw, Y

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
# Supports the full BCE-family config space (bce and bif_bce, incl. sw/iw/tax/phylo targets,
# cls_imb.norm, a BCE-family secondary-loss mix, and mix_unit_scale) -- only InfoNCE is excluded
# (chunking_supported). The reductions that couple across the whole BxB matrix -- the cls_imb.norm
# weight-mean normalizers (a 2D band sweep for bce; bif_bce's 1D per-anchor vector is O(B) and built
# outright), bce's global DSMR mass, and the per-loss mix_unit_scale scalar -- are all DETACHED
# constants, so they are precomputed (cheap embedding-free closed forms + no_grad band sweeps,
# all-reduced to rank-identical values) before the single grad-carrying backward sweep applies them
# as constants. See _precompute_crit_consts.
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

def chunking_supported(cfg_loss, cfg_loss2):
    """
    The tiled loss reproduces every BCE-family config (bce and bif_bce) but not InfoNCE (its
    row/column softmax couples along columns, which a row-block cannot tile). Config treats
    hardware.loss_chunk_size as inert (full BxB path) when this returns False.
    """
    if cfg_loss["crit"] not in ("bce", "bif_bce"):
        return False
    if cfg_loss2["mix"] != 0.0 and cfg_loss2["crit"] not in ("bce", "bif_bce"):
        return False
    return True

def make_targ_block_fn(targ_type, class_encs_b, targ_data_b, B, device):
    """
    Build a closure (rs, re) -> [re-rs, B] target row-block (rows rs:re vs all B columns) matching the
    full-batch compute_targets for the given targ_type. Reusable per-tile inputs (tax rank vectors,
    phylo correlation lookups) are precomputed once here so the sweeps only slice per block.
    """
    if targ_type == "sw":
        return lambda rs, re: (class_encs_b[rs:re].unsqueeze(1) == class_encs_b.unsqueeze(0)).float()
    if targ_type == "iw":
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

def bce_dsmr_mass(targ_type, targ_block_fn, class_encs_b, B, chunk_size, lo, hi, world_size):
    """
    Global DSMR mass over the full BxB target matrix: mass_pos = sum(targs), mass_neg = B^2 - sum(targs)
    (== sum(1 - targs) for targets in [0, 1]). For sw/iw (0/1 targets) mass_pos is the O(B) closed form
    sum_k count_k^2 / B (rank-identical, no collective); for soft tax/phylo targets it is summed over
    this rank's band [lo, hi) of target tiles (embedding-free) and all-reduced across bands.
    Matches torch.sum(targs) / torch.sum(1 - targs) in BCECriterion.__call__.
    """
    device = class_encs_b.device
    if targ_type == "sw":
        counts = torch.bincount(class_encs_b).to(torch.float64)
        mass_pos = (counts * counts).sum()
    elif targ_type == "iw":
        mass_pos = torch.tensor(float(B), dtype=torch.float64, device=device)
    else:  # tax, phylo -- soft targets: sum over this rank's band of tiles, fold across bands
        mass_pos = torch.zeros((), dtype=torch.float64, device=device)
        for rs in range(lo, hi, chunk_size):
            mass_pos += targ_block_fn(rs, rs + chunk_size).double().sum()
        if world_size > 1:
            dist.all_reduce(mass_pos)
    mass_pos = mass_pos.to(device=device, dtype=torch.float32)
    mass_neg = torch.tensor(float(B) * float(B), dtype=torch.float32, device=device) - mass_pos
    return mass_pos, mass_neg


class _SimTargStatsAccum:
    """
    Streams one loss branch's per-batch sim/target distribution stats over the loss tiles so the
    chunked path can report the same batch_stats keys as sim_targ_batch_stats without holding the
    full BxB matrices. min/max/mean are exact; the median is over a strided subsample of each tile
    (an exact BxB median would need the whole matrix).
    """
    def __init__(self, device):
        self.sim_min = torch.tensor(float("inf"), device=device)
        self.sim_max = torch.tensor(float("-inf"), device=device)
        self.sim_sum = torch.zeros((), dtype=torch.float64, device=device)
        self.targ_min = torch.tensor(float("inf"), device=device)
        self.targ_max = torch.tensor(float("-inf"), device=device)
        self.targ_sum = torch.zeros((), dtype=torch.float64, device=device)
        self.count = 0
        self.sim_samp = []
        self.targ_samp = []

    def update(self, sim_tile, targs_tile):
        s = sim_tile.reshape(-1).float()
        t = targs_tile.reshape(-1).float()
        self.sim_min = torch.minimum(self.sim_min, s.min())
        self.sim_max = torch.maximum(self.sim_max, s.max())
        self.sim_sum += s.double().sum()
        self.targ_min = torch.minimum(self.targ_min, t.min())
        self.targ_max = torch.maximum(self.targ_max, t.max())
        self.targ_sum += t.double().sum()
        self.count += s.numel()
        stride = max(1, s.numel() // 4096)  # bound the median subsample per tile
        self.sim_samp.append(s[::stride])
        self.targ_samp.append(t[::stride])

    def finalize(self, world_size, idx):
        sim_samp = torch.cat(self.sim_samp)
        targ_samp = torch.cat(self.targ_samp)
        sim_min, sim_max, sim_sum = self.sim_min, self.sim_max, self.sim_sum
        targ_min, targ_max, targ_sum = self.targ_min, self.targ_max, self.targ_sum
        count = self.count
        if world_size > 1:  # fold per-band partials; the bands partition the BxB rows exactly
            ext = torch.stack([-sim_min, -targ_min, sim_max, targ_max])
            dist.all_reduce(ext, op=dist.ReduceOp.MAX)
            sim_min, targ_min, sim_max, targ_max = -ext[0], -ext[1], ext[2], ext[3]
            sums = torch.stack([sim_sum, targ_sum])
            dist.all_reduce(sums)
            sim_sum, targ_sum = sums[0], sums[1]
            count *= world_size  # equal bands -> equal per-rank counts
            # median subsamples: equal bands + equal tile sizes -> equal lengths on every rank, so a
            # plain all_gather reassembles the exact same subsample pool a single full sweep produces
            samp = torch.stack([sim_samp, targ_samp])
            parts = [torch.empty_like(samp) for _ in range(world_size)]
            dist.all_gather(parts, samp)
            sim_samp = torch.cat([p[0] for p in parts])
            targ_samp = torch.cat([p[1] for p in parts])
        sim_median = sim_samp.median()
        targ_median = targ_samp.median()
        return {
            f"sim{idx}_min":     sim_min.item(),
            f"sim{idx}_max":     sim_max.item(),
            f"sim{idx}_median":  sim_median.item(),
            f"sim{idx}_mean":    (sim_sum / count).item(),
            f"targ{idx}_min":    targ_min.item(),
            f"targ{idx}_max":    targ_max.item(),
            f"targ{idx}_median": targ_median.item(),
            f"targ{idx}_mean":   (targ_sum / count).item(),
        }


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
    analogue of the full-batch opposite-tower detaches (_loss_for_crit_full_batch).
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

def _crit_block_logits_f(crit, secondary, rows, cols, compute_logits, center, center_global=None, half_live=False):
    """
    [C, B] similarity tile and its float32 logits tile for a criterion (its sim_type + logit
    scale/bias). `rows`/`cols` are the tile's anchor rows and full column embeddings -- (img rows,
    txt) non-bifurcated; a bifurcated branch passes its own (rows_live, cols) frame (_bif_branches)
    with half_live=True so the un-halved branch sum carries 1x logit scale/bias grads.

    `center`/`center_global` are passed through to compute_logits: the chunked sweeps supply the
    full-batch quantity (in-graph global sim mean for "sim"; detached full-batch incoming-grad mean
    for grad_proj/grad_proj2) so tiles reproduce full-batch centering exactly, never the per-tile mean.
    """
    sim_block = compute_sim(rows, cols, crit.cfg["sim"])
    logits_block = compute_logits(sim_block, crit.cfg["logits"]["temp"]["clamp"], center, secondary=secondary, center_global=center_global, half_live=half_live)
    return sim_block, logits_block.float()

def _precompute_crit_consts(crit, secondary, img, txt, targ_block_fn, class_encs_b, B,
                            compute_logits, chunk_size, mixed_prec, device, need_L, autocast_ctx,
                            lo, hi, world_size, center, center_global_det):
    """
    Detached global constants for one criterion (see module header). For bce, cls_imb_mean (mean of
    W_ci) and dsmr_mass are embedding-free; for bif_bce the consts are just the full normalized 1D
    per-anchor weight vector (O(B), built outright -- its cls_imb.norm mean is over B values, and
    row-wise DSMR needs no global mass), so the consts dicts are structurally distinct and a bif
    crit can never reach the 2D _crit_block_weight_bce. L_value (the criterion's full weighted loss,
    needed for mix_unit_scale) requires a no_grad tile sweep, run only when mix_unit_scale is
    active. All sweeps cover only this rank's row-band [lo, hi); the partial sums are all-reduced so
    every rank derives identical constants.
    `center`/`center_global_det` reproduce the criterion's centered forward in the L sweep (for
    "sim" the detached global sim mean; grad_proj* leave the forward untouched).
    Returns (consts dict for _crit_block_weight_bce / _bif_block_num_raw, L_value|None).
    """
    cfg_w = crit.cfg["wting"]
    targ_type = crit.cfg["targ"]

    if crit.bifurcated:
        # full [B] per-anchor weight vector, incl. the cls_imb.norm batch-mean division -- exactly
        # as BifurcatedBCECriterion.__call__ computes it (rank-identical, no collective)
        w_ci = compute_cls_imb_wts(cfg_w["cls_imb"], crit.counts, class_encs_b, crit.wting_dim, crit.wt_mean, crit.batch_size)
        if cfg_w["cls_imb"]["norm"]:
            w_ci = w_ci / w_ci.mean()
        consts = {"w_ci": w_ci}
    else:
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

        dsmr_mass = bce_dsmr_mass(targ_type, targ_block_fn, class_encs_b, B, chunk_size, lo, hi, world_size) if cfg_w["bce"]["dsmr"] else None
        consts = {"cls_imb_mean": cls_imb_mean, "dsmr_mass": dsmr_mass}

    L_value = None
    if need_L:
        sum_Wbce = torch.zeros((), dtype=torch.float64, device=device)
        with torch.no_grad():
            for rs in range(lo, hi, chunk_size):
                re = rs + chunk_size
                with autocast_ctx():
                    targs_block = targ_block_fn(rs, re)
                    if crit.bifurcated:  # both branches: their weighted sums differ (asymmetric logits, frame-dependent row weights)
                        W_dsmr, neut_mass = _bif_block_invariants(crit, targs_block, B)
                        for rows_live, cols in _bif_branches(img, txt):
                            _, logits_f = _crit_block_logits_f(crit, secondary, rows_live[rs:re], cols, compute_logits, center, center_global_det, half_live=True)
                            num, _ = _bif_block_num_raw(crit, logits_f, targs_block, consts["w_ci"][rs:re], W_dsmr, neut_mass)
                            sum_Wbce += num.double()
                    else:
                        _, logits_f = _crit_block_logits_f(crit, secondary, img[rs:re], txt, compute_logits, center, center_global_det)
                        W, bce = _crit_block_weight_bce(crit, logits_f, targs_block, class_encs_b[rs:re], class_encs_b, B, consts)
                        sum_Wbce += (W * bce).double().sum()
        if world_size > 1:
            dist.all_reduce(sum_Wbce)
        L_value = (sum_Wbce / B).float()

    return consts, L_value

def _crit_center_grad_mean(crit, secondary, img, txt, targ_fn, class_encs_b, B, consts, coeff,
                           compute_logits, chunk_size, autocast_ctx, lo, hi, world_size, device):
    """
    grad_proj/grad_proj2: the detached full-batch mean of the incoming gradient at the criterion's
    projection node, so every tile of the grad sweep subtracts the same constant the full-batch
    _ZeroSumGrad projection would (a per-tile mean would be wrong). Sweeps this rank's band with
    tile-local autograd (the sim tile as leaf, embeddings detached: graphs stay O(C*B), no param
    .grad touched), which captures the focal-weight gradient terms exactly; band partials are
    all-reduced. Returns the constant at the node the mode projects: the sim node for grad_proj
    (e^t folded in by autograd), the scaled-sim node for grad_proj2 (the post-clamp e^t divided
    back out, recovered as a probe gradient through compute_logits -- half_live changes no values
    and no dL/dsim, so the plain probe holds for bifurcated branches too).

    A bifurcated criterion gets a per-branch (i2t, t2i) tuple instead of a scalar: each branch's
    projection node sees only its own incoming grad (the branches join only at the scalar loss),
    and the two frames' grad means differ under per-anchor row weighting, so a shared constant
    would be silently wrong.
    """
    clamp = crit.cfg["logits"]["temp"]["clamp"]
    branches = _bif_branches(img, txt) if crit.bifurcated else ((img, txt),)
    g_sums = torch.zeros(len(branches), dtype=torch.float64, device=device)
    for rs in range(lo, hi, chunk_size):
        re = rs + chunk_size
        tile_losses, sim_leaves = [], []
        with autocast_ctx():
            targs_block = targ_fn(rs, re)
            if crit.bifurcated:
                W_dsmr, neut_mass = _bif_block_invariants(crit, targs_block, B)
            for rows_live, cols in branches:
                sim_leaf = compute_sim(rows_live[rs:re].detach(), cols.detach(), crit.cfg["sim"]).requires_grad_(True)
                logits_f = compute_logits(sim_leaf, clamp, None, secondary=secondary).float()
                if crit.bifurcated:
                    num, _ = _bif_block_num_raw(crit, logits_f, targs_block, consts["w_ci"][rs:re], W_dsmr, neut_mass)
                    tile_loss = num / B
                else:
                    W, bce = _crit_block_weight_bce(crit, logits_f, targs_block, class_encs_b[rs:re], class_encs_b, B, consts)
                    tile_loss = (W * bce).sum() / B
                tile_losses.append(tile_loss)
                sim_leaves.append(sim_leaf)
        for j, (tile_loss, sim_leaf) in enumerate(zip(tile_losses, sim_leaves)):
            g_sums[j] += torch.autograd.grad(tile_loss, sim_leaf)[0].double().sum()
    if world_size > 1:
        dist.all_reduce(g_sums)
    cs = (coeff * g_sums / (B * B)).float()  # mean incoming grad at the sim node(s), loss-mix coefficient folded in
    if crit.cfg["logits"]["bce"]["center"] == "grad_proj2":
        # the scaled-sim node's grad = (sim node's grad) / e^t; recover the post-clamp e^t as a probe gradient
        probe = torch.zeros(1, 1, device=device, requires_grad=True)
        e_det = torch.autograd.grad(compute_logits(probe, clamp, None, secondary=secondary).sum(), probe)[0].reshape(()).detach()
        cs = cs / e_det
    cs = cs.detach()
    return tuple(cs) if crit.bifurcated else cs[0]

def _gsum_hook(acc):
    """Tensor backward hook accumulating the incoming grad's sum into `acc`. Returns None so the
    gradient itself passes through untouched (a non-None return would replace it)."""
    def hook(g):
        acc.add_(g.double().sum())
    return hook

def chunked_bce_loss_backward(img, txt, class_encs_b, targ_data_b, crit1, crit2, mix, mix_unit_scale,
                              compute_logits, chunk_size, mixed_prec, device, rank, world_size):
    """
    Tiled + row-band-sharded global-batch BCE-family loss + backward (GradCache-style representation
    gradients). Computes the exact same weighted loss and gradients as the full-batch path
    (BCECriterion.__call__ / BifurcatedBCECriterion.__call__ blended by _global_batch_loss) over the
    full BxB matrix, but never materializes it and shares the work
    across ranks: the BxB rows split into world_size equal bands (SigLIP-style decomposition), this rank
    sums only its band [rank*b, (rank+1)*b) over row-blocks of C x B (b an exact multiple of C =
    chunk_size), and each block's gradient is backpropagated into the embedding leaves as computed, so
    peak VRAM is O(C*B) and per-rank compute is O(B^2/world_size). The loss/raw totals and batch stats
    are all-reduced, so every rank returns identical full-batch values; the leaves' .grad hold this
    band's PARTIAL dL/dembs, which sum to the full gradient across ranks (the caller completes them --
    see batch_step_chunked). Exact up to floating-point summation order.

    Supports a BCE-family loss mix: loss = (1 - mix)*s1*L1 + mix*s2*L2, where Lk is criterion k's
    weighted loss and sk = 1/Lk.detach() if mix_unit_scale else 1 (a bifurcated Lk normalizes by
    Lk/2, its gradient-scale-equivalent value, as in _global_batch_loss; mix == 0 -> just crit1).
    All cross-tile-coupled normalizers are precomputed detached constants (_precompute_crit_consts),
    so the backward is single-pass.

    - img, txt --------- detached [B, D] embedding leaves (requires_grad); receive band-partial dL/dembs
                         in their .grad. A bifurcated criterion sweeps two branch tiles per block
                         (module header), routing each branch's grads into one leaf.
    - crit1, crit2 ----- primary / secondary BCECriterion / BifurcatedBCECriterion (crit2 None when
                         mix == 0).
    - compute_logits --- VLMWrapper.compute_logits(sim, clamp, center, secondary, center_global) -> logits
                         tile; center_global carries the precomputed full-batch centering quantity
                         (module header) so tiled centering is exact.
    - rank, world_size - this rank's band index / number of bands (1 -> unsharded full sweep).

    Returns (loss, loss_raw, batch_stats, grad_sum_sims), all detached; gradients left in the leaves' /
    params' .grad. grad_sum_sims = (sum(dL/dsim1), sum(dL/dsim2)|None), the full-batch sums accumulated
    tile-by-tile via backward hooks (all-reduced across bands) -- the same values the full-batch path
    reads off the retained sim grads.
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
    crits = [(crit1, False)] + ([(crit2, True)] if mix != 0.0 else [])

    def autocast_ctx():
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16) if mixed_prec else nullcontext()

    centers = [crit.cfg["logits"]["bce"]["center"] for crit, _ in crits]
    for k, (crit, _) in enumerate(crits):
        # config rejects this combo too (TrainConfig); guard direct callers against a silently-wrong tile mean
        assert not (centers[k] == "sim" and crit.cfg["sim"] != "cos"), "center: sim under chunking requires cos sim"
    m_det = None
    if "sim" in centers:
        # full-batch sim-matrix mean via the cos bilinearity: mean_ij(img_i . txt_j) = mean(img) . mean(txt)
        with torch.no_grad():
            m_det = torch.dot(img.mean(0), txt.mean(0))

    need_L = mix != 0.0 and mix_unit_scale
    targ_fns, consts_list, L_values = [], [], []
    for k, (crit, secondary) in enumerate(crits):
        targ_fn = make_targ_block_fn(crit.cfg["targ"], class_encs_b, targ_data_b, B, device)
        consts, L_val = _precompute_crit_consts(crit, secondary, img, txt, targ_fn, class_encs_b, B,
                                                compute_logits, chunk_size, mixed_prec, device, need_L, autocast_ctx,
                                                lo, hi, world_size,
                                                centers[k], m_det if centers[k] == "sim" else None)
        targ_fns.append(targ_fn); consts_list.append(consts); L_values.append(L_val)

    mix_w = [1.0] if mix == 0.0 else [1.0 - mix, mix]
    if need_L:
        # mix_unit_scale: /Lk.detach(); a bifurcated loss normalizes by Lk/2, its gradient-scale-
        # equivalent value (un-halved branch sum reads 2x on 1x grads -- see _global_batch_loss)
        coeffs = [mix_w[k] / (L_values[k] / (2.0 if crits[k][0].bifurcated else 1.0)).clamp_min(1e-12) for k in range(len(crits))]
    else:
        coeffs = list(mix_w)

    # grad_proj*: the detached full-batch incoming-grad mean per criterion -- the constant every tile
    # of the grad sweep subtracts in place of the full path's g.mean() (_ZeroSumGradConst)
    center_consts = [
        _crit_center_grad_mean(crit, secondary, img, txt, targ_fns[k], class_encs_b, B, consts_list[k],
                               coeffs[k], compute_logits, chunk_size, autocast_ctx, lo, hi, world_size, device)
        if centers[k] in ("grad_proj", "grad_proj2") else None
        for k, (crit, secondary) in enumerate(crits)
    ]

    # sim-grad-sum metric (learning-curve strip): accumulated tile-by-tile via backward hooks. Under
    # "sim" centering part of the full-path dL/dsim routes through the global mean, which here lives
    # on the leaves and bypasses the tiles -- a hook on the in-graph mean captures it (dm/dsim sums
    # to exactly 1), so the folded totals match the full-batch sim.grad.sum().
    grad_sums = [torch.zeros((), dtype=torch.float64, device=device) for _ in crits]

    wbce_tot = [torch.zeros((), dtype=torch.float64, device=device) for _ in crits]
    raw_tot = [torch.zeros((), dtype=torch.float64, device=device) for _ in crits]
    stats = [_SimTargStatsAccum(device) for _ in crits]

    for rs in range(lo, hi, chunk_size):
        re = rs + chunk_size  # the band is an exact multiple of chunk_size (checked above)
        targ_blocks = []
        sim_blocks = []
        with autocast_ctx():
            block_loss = 0.0
            for k, (crit, secondary) in enumerate(crits):
                targs_block = targ_fns[k](rs, re)
                if crit.bifurcated:
                    W_dsmr, neut_mass = _bif_block_invariants(crit, targs_block, B)
                    num = 0.0
                    for j, (rows_live, cols) in enumerate(_bif_branches(img, txt)):
                        cg = center_consts[k][j] if center_consts[k] is not None else None
                        if centers[k] == "sim":
                            # per-branch in-graph mean: the centering's backward routes into the
                            # branch's live tower only, mirroring the branch's detach pattern
                            cg = torch.dot(rows_live.mean(0), cols.mean(0))
                            if cg.requires_grad:
                                cg.register_hook(_gsum_hook(grad_sums[k]))
                        sim_block, logits_f = _crit_block_logits_f(crit, secondary, rows_live[rs:re], cols, compute_logits, centers[k], cg, half_live=True)
                        if sim_block.requires_grad:
                            sim_block.register_hook(_gsum_hook(grad_sums[k]))
                        b_num, bce = _bif_block_num_raw(crit, logits_f, targs_block, consts_list[k]["w_ci"][rs:re], W_dsmr, neut_mass)
                        num = num + b_num  # branches summed un-halved (full-batch: mean1 + mean2 = (sum1 + sum2)/B)
                        raw_tot[k] += bce.sum().detach().double()
                        if j == 0:
                            sim_block_k = sim_block.detach()  # i2t frame -- values match the non-bif sim
                else:
                    cg = center_consts[k]
                    if centers[k] == "sim":
                        # in-graph per block: the centering's backward routes through the mean embeddings
                        # into every leaf row, completing the exact full-batch projection across blocks
                        cg = torch.dot(img.mean(0), txt.mean(0))
                        if cg.requires_grad:
                            cg.register_hook(_gsum_hook(grad_sums[k]))
                    sim_block, logits_f = _crit_block_logits_f(crit, secondary, img[rs:re], txt, compute_logits, centers[k], cg)
                    if sim_block.requires_grad:
                        sim_block.register_hook(_gsum_hook(grad_sums[k]))
                    W, bce = _crit_block_weight_bce(crit, logits_f, targs_block, class_encs_b[rs:re], class_encs_b, B, consts_list[k])
                    num = (W * bce).sum()
                    raw_tot[k] += bce.sum().detach().double()
                    sim_block_k = sim_block.detach()
                block_loss = block_loss + coeffs[k] * num / B
                wbce_tot[k] += num.detach().double()
                targ_blocks.append(targs_block.detach())
                sim_blocks.append(sim_block_k)
        block_loss.backward()
        for k in range(len(crits)):
            stats[k].update(sim_blocks[k], targ_blocks[k])

    if world_size > 1:  # fold the band-partial loss totals; the leaves' .grad stay band-partial
        packed = torch.stack(wbce_tot + raw_tot + grad_sums)
        dist.all_reduce(packed)
        wbce_tot = [packed[k] for k in range(len(crits))]
        raw_tot = [packed[len(crits) + k] for k in range(len(crits))]
        grad_sums = [packed[2 * len(crits) + k] for k in range(len(crits))]

    loss = torch.zeros((), dtype=torch.float64, device=device)
    loss_raw = torch.zeros((), dtype=torch.float64, device=device)
    for k in range(len(crits)):
        loss += coeffs[k] * (wbce_tot[k] / B)
        loss_raw += mix_w[k] * (raw_tot[k] / B)
    grad_sum_sims = (grad_sums[0].item(), grad_sums[1].item() if len(crits) == 2 else None)
    batch_stats = {}
    for k in range(len(crits)):  # fixed order: finalize runs collectives, so ranks must agree
        batch_stats.update(stats[k].finalize(world_size, idx=k + 1))
    return loss.float(), loss_raw.float(), batch_stats, grad_sum_sims
