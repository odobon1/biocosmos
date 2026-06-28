import numpy as np
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
import math
import os
import multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import chain, islice
from PIL import Image
from PIL.GifImagePlugin import getheader, getdata

from utils.data import load_cid_2_penult, load_cid_2_nshot
from utils.ddp import rank0
from utils.utils import DATASET_ALIAS2NAME, save_pickle, load_pickle
from utils.config import DATASET2MARKER_SIZE


_EVAL_ALIAS2NAME = {"val": "Validation", "test": "Test"}


@dataclass(frozen=True)
class VizContext:
    """Identity of the eval being visualized -- drives plot titles and the per-dataset color/label
    lookups. Built once from the train config and threaded through the manifold-viz entry points."""
    setting: str
    dataset: str
    split: str
    eval_type: str


@dataclass(frozen=True)
class RenderStyle:
    """Static styling + strobe schedule for the composite renderers. Bundled because the same set
    rides through generate_* -> composite_plot/strobe_gif/evolution_gif -> _composite_canvas, and is
    pickled to the render workers. `method` ("t-SNE"/"PCA") and `col_titles` vary per (method, grid);
    the rest are fixed per generate_* call (composite_plot ignores n_stoch_layers/frame_ms)."""
    method: str
    marker_size: int
    legend_by_role: dict
    n_stoch_layers: int
    frame_ms: float
    bg_color: str | None
    col_titles: list | None = None


def _manifold_title(method, viz_context, subject, suffix=""):
    """Suptitle for a manifold grid, e.g. 't-SNE: Full-Set (ID) Validation -- hp, Nymphalidae, 50k'."""
    return f"{method}: {subject} {_EVAL_ALIAS2NAME[viz_context.eval_type]} -- {viz_context.setting}, {DATASET_ALIAS2NAME[viz_context.dataset]}{suffix}"

_GIF_DPI = 100  # evolution-GIF frame resolution (lower than the 300-dpi static PNGs)
_OOD_LABEL = "__OOD__"  # sentinel label for OOD points in the n-shot panel (always drawn black)

def _log(msg):
    """Print only on rank 0 (or when not under DDP). The projection compute now runs on every rank, so
    its progress prints would otherwise be duplicated world_size times."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(msg)

def nshot_color_map(nst_names):
    """Map each n-shot bucket to its learning-curve color + the OOD sentinel to black. The curves
    plot buckets via matplotlib's default color cycle in reversed-bucket order (see plot_metrics),
    so we mirror that order here to keep bucket colors consistent between the curves and these plots."""
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    cmap = {name: np.array(to_rgb(cycle[i % len(cycle)])) for i, name in enumerate(reversed(list(nst_names)))}
    cmap[_OOD_LABEL] = np.array([0.0, 0.0, 0.0])
    return cmap

def _legend_specs(color_nshot, nst_names):
    """Per-color-role legend descriptor for a panel's coloring: a text tag for leaf/penult panels, and
    a swatch legend (zero-shot/OOD black, then the ID buckets) for the n-shot panel. Consumed by
    _apply_legend, threaded onto every t-SNE/PCA axis."""
    entries = [("zero-shot", color_nshot[_OOD_LABEL])] + [(name, color_nshot[name]) for name in nst_names]
    return {
        "leaf":   ("text", "leaf-class coloring"),
        "penult": ("text", "penultimate-class coloring"),
        "nshot":  ("legend", entries),
    }

def _apply_legend(ax, desc, fontsize):
    """Draw a panel's coloring legend onto `ax`: a swatch legend for the n-shot panel, else a text tag."""
    kind, payload = desc
    if kind == "legend":
        handles = [Patch(facecolor=color, edgecolor="none", label=label) for label, color in payload]
        ax.legend(handles=handles, loc="upper right", fontsize=fontsize, framealpha=0.85,
                  handlelength=1.0, borderpad=0.4, labelspacing=0.3)
    else:
        ax.text(0.015, 0.985, payload, transform=ax.transAxes, va="top", ha="left", fontsize=fontsize,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="none"))


def _shard_counts(n, world):
    """Near-even contiguous row counts per rank (the first n % world ranks each take one extra row)."""
    base, extra = divmod(n, world)
    return [base + (1 if r < extra else 0) for r in range(world)]

def _zero_self(M, offset=0):
    """Zero each row's self-affinity entry in place: M[a, offset + a] = 0 for every row a. For a square
    matrix with offset 0 this is the diagonal; for a row-shard whose global row indices run
    offset..offset+rows it zeros the self column of each local row."""
    a = torch.arange(M.shape[0], device=M.device)
    M[a, offset + a] = 0.0

def _transpose_shard(Pc, counts, starts):
    """All-to-all transpose of a row-sharded matrix. `Pc` is this rank's row-block (n_local x N) of the
    full N x N conditional; returns PcT (n_local x N) holding this rank's COLUMNS of the full matrix
    transposed -- PcT[a, j] = Pc_full[j, local_row a] -- assembled from the matching column-block of
    every rank's row-block. Implemented with batch_isend_irecv so it runs on both gloo and NCCL; the
    rank's own block is copied locally (no send)."""
    rank, world = dist.get_rank(), dist.get_world_size()
    nl = counts[rank]
    send = [Pc[:, starts[r]:starts[r] + counts[r]].contiguous() for r in range(world)]  # block -> dest r
    recv = [torch.empty((counts[q], nl), device=Pc.device) for q in range(world)]        # block <- src q
    recv[rank] = send[rank]
    ops = [dist.P2POp(dist.irecv, recv[q], q) for q in range(world) if q != rank]
    ops += [dist.P2POp(dist.isend, send[r], r) for r in range(world) if r != rank]
    if ops:  # empty only at world_size 1 (own block already copied); batch_isend_irecv rejects []
        for req in dist.batch_isend_irecv(ops):
            req.wait()
    return torch.cat(recv, dim=0).t().contiguous()  # (N x n_local) -> (n_local x N)

def _allgather_rows(Yr, counts):
    """All-gather the variable-height local row blocks `Yr` (n_local x d) into the full layout (N x d).
    Pads each block to the max height (shards can differ by one row) for the uniform-shape all_gather,
    then slices each rank's real rows back out."""
    world = dist.get_world_size()
    maxc, d = max(counts), Yr.shape[1]
    buf = torch.zeros((maxc, d), device=Yr.device)
    buf[:Yr.shape[0]] = Yr
    parts = [torch.empty((maxc, d), device=Yr.device) for _ in range(world)]
    dist.all_gather(parts, buf)
    return torch.cat([parts[q][:counts[q]] for q in range(world)], dim=0)

def _hbeta_search(D2, perplexity, self_offset=0, tol=1e-5, max_iter=100):
    """Per-point Gaussian precision (beta = 1/2sigma^2) tuned so each row of the conditional
    affinity matrix has the target `perplexity`, via vectorized binary search over all points at
    once. `D2` is the (rows, N) squared-distance matrix (rows == N for the single-GPU path, or a
    row-shard for the sharded path); returns the row-normalized conditionals P_{j|i} (self-affinity
    zeroed -- column `self_offset + row` for each row). Mirrors sklearn's _binary_search_perplexity."""
    n = D2.shape[0]
    dev = D2.device
    beta = torch.ones(n, 1, device=dev)
    betamin = torch.full((n, 1), -float("inf"), device=dev)
    betamax = torch.full((n, 1), float("inf"), device=dev)
    logU = math.log(perplexity)
    def conditionals(beta):  # exp(-D2*beta) with self-affinity zeroed, no extra N×N mask
        Pexp = torch.exp(-D2 * beta)
        _zero_self(Pexp, self_offset)
        return Pexp
    for _ in range(max_iter):
        Pexp = conditionals(beta)
        sumP = Pexp.sum(1, keepdim=True).clamp_min(1e-12)
        H = torch.log(sumP) + beta * (D2 * Pexp).sum(1, keepdim=True) / sumP  # row entropy
        diff = H - logU
        pos = diff > 0
        betamin = torch.where(pos, beta, betamin)
        betamax = torch.where(pos, betamax, beta)
        beta = torch.where(
            pos,
            torch.where(torch.isinf(betamax), beta * 2, (beta + betamax) / 2),
            torch.where(torch.isinf(betamin), beta / 2, (beta + betamin) / 2),
        )
        if diff.abs().max() < tol:
            break
    Pexp = conditionals(beta)
    return Pexp / Pexp.sum(1, keepdim=True).clamp_min(1e-12)

def _tsne_torch(X, init, perplexities, n_iter=1000, exaggeration=12.0, explore_iter=250, device=None):
    """Exact (O(n^2)) t-SNE on the GPU via torch, run for each perplexity in `perplexities` and returned
    as {perplexity: 2D layout}. Dispatches to the multi-rank sharded implementation when running under DDP
    with world_size > 1 (the N×N matrices are split across ranks so each holds only ~N²/world_size), else
    the single-process implementation. Both build the perplexity-independent high-dim distance matrix once
    and reuse it across all perplexities, follow sklearn's schedule, and are structurally equivalent.
    `init` is the 2D starting layout (the reused PCA init)."""
    if dist.is_initialized() and dist.get_world_size() > 1:
        return _tsne_torch_sharded(X, init, perplexities, n_iter, exaggeration, explore_iter, device)
    return _tsne_torch_single(X, init, perplexities, n_iter, exaggeration, explore_iter, device)

def _tsne_torch_single(X, init, perplexities, n_iter=1000, exaggeration=12.0, explore_iter=250, device=None):
    """
    Exact (O(n^2)) t-SNE on the GPU via torch, run for each perplexity in `perplexities` and returned as
    {perplexity: 2D layout}. The high-dim squared-distance matrix is perplexity-independent, so it (with
    the PCA init, learning rate, and preallocated work buffers) is built once and reused across all
    perplexities; only the perplexity-matched affinities P differ per run. Each run minimizes KL(P||Q)
    under the Student-t low-dim kernel by momentum gradient descent with adaptive gains, following
    sklearn's schedule (PCA init, early exaggeration for the first `explore_iter` steps, momentum
    0.5->0.8, learning_rate='auto'), so the result is structurally equivalent to sklearn t-SNE -- not
    bit-identical. `init` is the 2D starting layout (the reused PCA init).

    Memory is pinned to the n×n distance matrix + exactly 3 live n×n buffers (P, plus two preallocated
    work buffers reused in place every iteration) so it fits large n: early exaggeration is folded into P
    in-place (no separate P_exag), the diagonal is zeroed instead of multiplying by an n×n mask, and the
    low-dim affinities are built via the gram trick into the preallocated buffers (no cdist temporaries
    and no per-iteration allocation churn).
    """
    eps = 1e-12
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    X = torch.as_tensor(X, dtype=torch.float32, device=dev)
    n = X.shape[0]
    D2 = torch.cdist(X, X).pow_(2)  # high-dim squared distances; perplexity-independent -> built once, reused
    lr = max(n / exaggeration / 4.0, 50.0)  # sklearn learning_rate='auto'
    Y0 = torch.as_tensor(init, dtype=torch.float32, device=dev)  # shared PCA init, cloned per perplexity
    num = torch.empty((n, n), device=dev)  # preallocated Student-t affinity buffer (reused each iter/perp)
    Q = torch.empty((n, n), device=dev)    # preallocated PQ-term buffer (reused each iter/perp)

    out = {}
    for perplexity in perplexities:
        P = _hbeta_search(D2, perplexity)
        P = ((P + P.t()) / (2 * n)).clamp_min(eps)  # symmetrize -> joint, sums to 1
        P.mul_(exaggeration)  # early exaggeration folded into P (undone at explore_iter) -> no 2nd n×n buffer
        Y = Y0.clone()
        update = torch.zeros_like(Y)
        gains = torch.ones_like(Y)
        for it in range(n_iter):
            if it == explore_iter:
                P.div_(exaggeration)  # end early exaggeration
            momentum = 0.5 if it < explore_iter else 0.8
            # Student-t affinities num = 1/(1+||yi-yj||^2), self zeroed, into the reused buffer (gram trick)
            ry = (Y * Y).sum(1)
            torch.matmul(Y, Y.t(), out=num)
            num.mul_(-2.0).add_(ry[:, None]).add_(ry[None, :]).clamp_min_(0.0).add_(1.0).reciprocal_()
            num.fill_diagonal_(0.0)
            torch.div(num, num.sum().clamp_min(eps), out=Q).clamp_min_(eps)
            Q.neg_().add_(P).mul_(num)  # in-place: Q := (P - Q) * num  (the PQ term; P carries the exaggeration)
            grad = 4.0 * (Q.sum(1, keepdim=True) * Y - Q @ Y)
            inc = (update * grad) < 0  # adaptive gains: grow when sign flips, shrink otherwise
            gains = torch.where(inc, gains + 0.2, gains * 0.8).clamp_min(0.01)
            update = momentum * update - lr * gains * grad
            Y = Y + update
            Y = Y - Y.mean(0, keepdim=True)  # keep centered (t-SNE is translation-invariant)
        out[perplexity] = Y.detach().cpu().numpy()
    return out

def _tsne_torch_sharded(X, init, perplexities, n_iter, exaggeration, explore_iter, device):
    """
    Multi-rank exact t-SNE, run for each perplexity in `perplexities` and returned as {perplexity: 2D
    layout}. The NxN affinity / Student-t matrices are row-sharded across ranks (each rank owns a
    contiguous block of the N points), so per-rank memory is ~3·N²/world_size instead of 3·N². This both
    lifts the single-GPU O(N²) memory ceiling and splits the work. This rank's nlxN high-dim
    squared-distance rows are perplexity-independent, so they (with the shard layout and preallocated
    buffers) are built once and reused across all perplexities. X is replicated (it is already
    all-gathered to every rank during eval) and the 2-D layout Y is replicated, all-gathered from the
    per-rank updated rows every iteration (only Nx2). Each iteration all-reduces the scalar Student-t
    normalizer Z; the one-time P-symmetrization transpose is a point-to-point all-to-all
    (_transpose_shard). The per-row optimizer state (gains/update) is sharded with the rows. Same
    schedule/algorithm as _tsne_torch_single -- structurally equivalent (only the distributed Z
    reduction differs in floating-point summation order).

    Must be entered collectively by every rank (it issues collective ops); the perplexity list is
    identical across ranks (shared config), so the collectives stay symmetric. X and init must be
    identical across ranks (they are: X is all-gathered and init is the deterministic shared PCA).
    """
    eps = 1e-12
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rank, world = dist.get_rank(), dist.get_world_size()
    X = torch.as_tensor(X, dtype=torch.float32, device=dev)
    n = X.shape[0]
    counts = _shard_counts(n, world)
    starts = [sum(counts[:r]) for r in range(world)]
    r0, nl = starts[rank], counts[rank]
    Xr = X[r0:r0 + nl]  # this rank's rows of X (nl × d)
    D2 = torch.cdist(Xr, X).pow_(2)  # this rank's nl×N squared-distance rows; perplexity-independent -> reused
    lr = max(n / exaggeration / 4.0, 50.0)  # sklearn learning_rate='auto'
    Y0 = torch.as_tensor(init, dtype=torch.float32, device=dev)  # N × 2 shared PCA init, cloned per perplexity
    num = torch.empty((nl, n), device=dev)  # preallocated local Student-t rows (reused each iter/perp)
    Q = torch.empty((nl, n), device=dev)    # preallocated local PQ-term buffer (reused each iter/perp)

    out = {}
    for perplexity in perplexities:
        # high-dim joint affinities P, row-sharded. The conditional rows come from a per-row perplexity
        # search (independent across rows -> shards cleanly). Symmetrizing to the joint P needs P's
        # transpose, i.e. this rank's columns of the full conditional, gathered from every rank's row-block.
        Pc = _hbeta_search(D2, perplexity, self_offset=r0)  # nl × N conditional
        P = ((Pc + _transpose_shard(Pc, counts, starts)) / (2 * n)).clamp_min(eps)  # nl × N joint
        del Pc
        P.mul_(exaggeration)  # early exaggeration folded into P (undone at explore_iter)
        Y = Y0.clone()  # N × 2, replicated
        update = torch.zeros((nl, 2), device=dev)  # per-row optimizer state, sharded with the rows
        gains = torch.ones((nl, 2), device=dev)
        for it in range(n_iter):
            if it == explore_iter:
                P.div_(exaggeration)  # end early exaggeration
            momentum = 0.5 if it < explore_iter else 0.8
            Yr = Y[r0:r0 + nl]
            # Student-t affinities for the local rows: num[a,j] = 1/(1+||y_{r0+a} - y_j||^2), self zeroed
            ry = (Y * Y).sum(1)
            torch.matmul(Yr, Y.t(), out=num)
            num.mul_(-2.0).add_(ry[r0:r0 + nl, None]).add_(ry[None, :]).clamp_min_(0.0).add_(1.0).reciprocal_()
            _zero_self(num, r0)
            Z = num.sum()
            dist.all_reduce(Z, op=dist.ReduceOp.SUM)  # global normalizer over all N×N pairs
            torch.div(num, Z.clamp_min(eps), out=Q).clamp_min_(eps)
            Q.neg_().add_(P).mul_(num)  # in-place: Q := (P - Q) * num  (PQ term; P carries the exaggeration)
            grad = 4.0 * (Q.sum(1, keepdim=True) * Yr - Q @ Y)  # nl × 2 local-row gradient (needs full Y)
            inc = (update * grad) < 0  # adaptive gains: grow when sign flips, shrink otherwise
            gains = torch.where(inc, gains + 0.2, gains * 0.8).clamp_min(0.01)
            update = momentum * update - lr * gains * grad
            Y = _allgather_rows(Yr + update, counts)  # stitch the updated local rows back into the full Y
            Y = Y - Y.mean(0, keepdim=True)  # keep centered (every rank identical after the all-gather)
        out[perplexity] = Y.detach().cpu().numpy()
    return out

def compute_tsne(embeddings, perplexities, n_iter=1000, init=None):
    """
    Reduce embedding dimensionality to 2D via GPU t-SNE (`_tsne_torch`), one t-SNE per perplexity in
    `perplexities` (the high-dim distance matrix is built once and reused across them). Returns
    {perplexity: 2D layout}. `embeddings` may be a numpy array or a torch tensor (a GPU tensor is passed
    straight through to the sharded path, no host copy). `init` is the 2D starting layout; when omitted, a
    PCA init is computed (the caller normally passes the reused PCA).
    """
    _log(f"Running GPU t-SNE on {embeddings.shape[0]} samples (dim={embeddings.shape[1]}) for perplexities {list(perplexities)}...")
    if init is None:
        init = _pca_init(compute_pca(np.asarray(embeddings)))
    return _tsne_torch(embeddings, np.asarray(init), perplexities=perplexities, n_iter=n_iter)

def compute_pca(embeddings):
    """
    Reduce embedding dimensionality to 2D via PCA. Uses the same solver/seed t-SNE uses for its
    internal `init="pca"`, so this projection can be fed straight back as the t-SNE init (see
    `_pca_init`) instead of computing the same PCA twice.
    """
    _log(f"Running PCA on {embeddings.shape[0]} samples (dim={embeddings.shape[1]})...")
    pca = PCA(n_components=2, svd_solver="randomized", random_state=42)
    embs_2d = pca.fit_transform(embeddings)
    return embs_2d

def _pca_init(pca_2d):
    """Scale a 2D PCA projection to t-SNE's PCA-init convention (PC1 std -> 1e-4) so it can be reused
    as the t-SNE `init` rather than having TSNE recompute the same PCA internally."""
    return (pca_2d / np.std(pca_2d[:, 0]) * 1e-4).astype(np.float32)

def orient_tsne(proj, labels, ref=None, tau=1.0):
    """
    Pin a canonical orientation for a t-SNE projection (whose axes and handedness are arbitrary) so the
    layout neither spins nor mirror-flips across an ordered sequence of evals. The transform is rigid
    about the origin (rotation + optional reflection), so all structure is preserved.

    The first eval (`ref=None`) bootstraps the canonical frame: rotate the highest-cardinality class's CoM
    onto the +y axis, then reflect about y so the top-3 class CoM triangle has a fixed (positive)
    chirality. Every later eval is aligned to a running per-class reference constellation via
    count-weighted orthogonal Procrustes (reflection allowed): it picks the rotation+reflection of this
    eval's class CoMs that best matches the reference. Deciding handedness from the WHOLE constellation,
    rather than a single anchor triangle, is what makes it robust -- the old top-3-triangle rule flipped
    whenever those three anchors happened to be near-collinear (a degenerate triangle whose chirality sign
    is noise), which is exactly what produced mirror-flipping between evals. The eval set is fixed across
    checkpoints, so the same classes anchor every eval; ties on cardinality break to the smaller label.

    `ref` is the running reference {class: CoM} (None on the first eval); `tau` in (0, 1] EMA-mixes each
    eval's oriented CoMs into it (1.0 = align to the previous eval only; smaller damps drift across evals).
    Returns (oriented_proj, ref).
    """
    labels = np.asarray(labels)
    vals, counts = np.unique(labels, return_counts=True)  # vals sorted -> deterministic tie-break
    order = sorted(range(len(vals)), key=lambda i: (-counts[i], vals[i]))  # classes, most populous first
    coms = {v: proj[labels == v].mean(axis=0) for v in vals}  # this eval's raw per-class CoMs

    if ref is None:
        # bootstrap the canonical frame: rotate top-1 CoM onto +y, fix handedness via top-3 chirality
        com = coms[vals[order[0]]]
        flip = 1.0
        if len(order) >= 3:
            c1, c2 = coms[vals[order[1]]], coms[vals[order[2]]]
            cross = (c1[0] - com[0]) * (c2[1] - com[1]) - (c1[1] - com[1]) * (c2[0] - com[0])
            if cross < 0:
                flip = -1.0
        u = com / max(np.linalg.norm(com), 1e-12)
        theta = np.pi / 2 - np.arctan2(u[1], u[0])
        cth, sth = np.cos(theta), np.sin(theta)
        Q = np.array([[cth, sth], [-sth, cth]]) @ np.array([[flip, 0.0], [0.0, 1.0]])  # rotate, then reflect x
    else:
        # align this eval's class-CoM constellation to the reference: count-weighted orthogonal Procrustes
        # with reflection. Q = U V^T (M = C^T W R, M = U S V^T) maximizes the match of (proj @ Q)'s CoMs
        # to the reference; det(Q) = +/-1, so a mirror is chosen iff it fits the reference better.
        shared = [v for v in vals if v in ref]
        count_of = dict(zip(vals, counts))
        w = np.array([count_of[v] for v in shared], dtype=float)
        C = np.array([coms[v] for v in shared])
        R = np.array([ref[v] for v in shared])
        U, _, Vt = np.linalg.svd(C.T @ (w[:, None] * R))
        Q = U @ Vt

    proj = proj @ Q
    coms_oriented = {v: proj[labels == v].mean(axis=0) for v in vals}
    if ref is None:
        ref = coms_oriented
    else:
        ref = {v: (1.0 - tau) * ref.get(v, coms_oriented[v]) + tau * coms_oriented[v] for v in vals}
    return proj, ref

_HUE_STRIDE = (math.sqrt(5) - 1) / 2  # 1/phi; golden-ratio low-discrepancy hue stride
_BAND_STRIDE = math.sqrt(2) - 1       # independent low-discrepancy stride for the vibrancy band

def assign_colors(labels, counts, cfg_color, hue_offset=0.0):
    """
    One stable color per class, assigned in order of sample count so that classes with SIMILAR
    counts get maximally different colors and near-identical colors only ever land on classes
    with very different counts. Classes are ranked by descending count (highest-cardinality class
    at rank 0, ties broken by label); each step in rank advances the hue by the golden ratio
    (consecutive counts ~137 deg apart in hue) and advances the vibrancy band by an independent
    low-discrepancy stride (honoring band `weight`s). The band sequence starts at offset 0, so the
    top class always lands in the first band. Two classes only collide in color when both their hue
    rank and band rank near-coincide, which -- being two low-discrepancy sequences -- happens only
    at large count-rank separations.

    Determined entirely by the class set + counts + bands + seed, so a class keeps its color across
    every eval / t-SNE / PCA / ID-OOD plot. `counts` may be a Counter (missing classes count as 0).
    `hue_offset` shifts the hue sequence (for distinguishing leaf vs penult colormaps).
    """
    bands = cfg_color["bands"]
    w = np.array([b["weight"] for b in bands], dtype=float)
    cdf = np.cumsum(w) / w.sum()

    # `seed` rigidly rotates the hue sequence (a phase offset), so it shifts the whole palette
    # without disturbing the golden-ratio spacing -- the count-ordering guarantees hold.
    rng = np.random.default_rng(cfg_color["seed"])
    off_h = (rng.random() + hue_offset) % 1.0

    classes = sorted(set(labels), key=lambda c: (-counts[c], c))  # rank by descending count, ties by label
    cmap = {}
    for r, cls in enumerate(classes):
        h = (r * _HUE_STRIDE + off_h) % 1.0
        b = bands[min(int(np.searchsorted(cdf, (r * _BAND_STRIDE) % 1.0, side="right")), len(bands) - 1)]
        cmap[cls] = sns.husl_palette(1, h=h, s=b["saturation"], l=b["lightness"])[0]
    return cmap

def _common_limits(projs, margin=0.05):
    """Union bounding box (with a margin) over a list of (N,2) projections -> (xlim, ylim).
    Used to freeze the axes/gridlines across evals so they don't jump frame-to-frame."""
    allp = np.concatenate(projs, axis=0)
    (xmin, ymin), (xmax, ymax) = allp.min(axis=0), allp.max(axis=0)
    dx, dy = (xmax - xmin) * margin, (ymax - ymin) * margin
    return (xmin - dx, xmax + dx), (ymin - dy, ymax + dy)

def _square_limits(projs, margin=0.05):
    """Symmetric square box (centered at origin, with a margin) covering a list of (N,2) projections
    -> (xlim, ylim) sharing one +/- bound. t-SNE plots are equal-aspect and origin-centered, so their
    axes auto-scale to a single square bound that fits the data (instead of a fixed config bound that
    doesn't transfer across datasets). Over a list of per-eval projections this unions them, freezing
    one bound across the whole evolution GIF."""
    allp = np.concatenate(projs, axis=0)
    bound = np.abs(allp).max() * (1 + margin)
    return (-bound, bound), (-bound, bound)

def _limits_for(method, projs):
    """Axis limits for a panel: square origin-centered bounds for t-SNE, the data bounding box for PCA."""
    return _square_limits(projs) if method == "t-SNE" else _common_limits(projs)

def _rgba(colors, alpha):
    """Stack per-point RGB (N,3) + alpha (scalar or (N,)) into an (N,4) RGBA array."""
    rgba = np.empty((len(colors), 4))
    rgba[:, :3] = colors
    rgba[:, 3] = alpha
    return rgba

def _canvas_frame(fig):
    """Render the figure's current state straight off the Agg canvas to a PIL RGB frame (no
    intermediate PNG encode/decode, no tight-bbox measuring pass)."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return Image.fromarray(buf[..., :3].copy())

def _fixed_palette(frames):
    """ONE shared 256-color palette (MEDIANCUT, no dither) built from `frames` at once. Reused to quantize
    every GIF frame so the palette is fixed across the GIF: PIL's default builds a fresh adaptive palette
    per frame, which remaps identical colors to different palette slots each frame and makes otherwise-
    static regions shimmer. Dithering is off so each color maps to its nearest palette entry
    deterministically (same color -> same slot in every frame)."""
    stacked = Image.fromarray(np.concatenate([np.asarray(f) for f in frames], axis=0))
    return stacked.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)

def _save_gif(frames, fpath_gif, frame_ms):
    """Write a forever-looping GIF from a list of PIL frames (sizes reconciled to the first frame), all
    quantized to one shared fixed palette (see _fixed_palette)."""
    frames = [f if f.size == frames[0].size else f.resize(frames[0].size) for f in frames]
    palette = _fixed_palette(frames)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    fpath_gif.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(fpath_gif, save_all=True, append_images=frames[1:], duration=frame_ms, loop=0, disposal=2)

def _save_gif_stream(frame_iter, fpath_gif, frame_ms, palette_sample):
    """Like _save_gif but for an arbitrarily long frame stream: peak memory is bounded to `palette_sample`
    frames regardless of the total frame count (== n_evals * n_stoch_layers for the evolution GIF), so it
    doesn't scale with the number of checkpoints. The shared fixed palette is built from the first
    `palette_sample` frames -- representative because an evolution GIF's frames share one color SET (same
    eval points, same color maps; only positions move) -- then each frame is quantized and written one at a
    time via the legacy getheader/getdata frame blocks. Image.save(append_images=...) can't stream: it
    buffers every diffed frame before writing, so its peak memory scales with the frame count."""
    it = iter(frame_iter)
    sample = list(islice(it, palette_sample))
    size0 = sample[0].size
    sample = [f if f.size == size0 else f.resize(size0) for f in sample]
    palette = _fixed_palette(sample)
    quant = lambda f: (f if f.size == size0 else f.resize(size0)).quantize(palette=palette, dither=Image.Dither.NONE)
    fpath_gif.parent.mkdir(parents=True, exist_ok=True)
    with open(fpath_gif, "wb") as fp:
        for block in getheader(quant(sample[0]), info={"loop": 0})[0]:  # signature + screen descriptor + global palette + loop ext
            fp.write(block)
        for f in chain(sample, it):  # at most one streamed frame resident beyond the palette sample
            for block in getdata(quant(f), duration=frame_ms, disposal=2):  # local header + LZW data per frame
                fp.write(block)
        fp.write(b";")  # GIF trailer

def _render_init():
    """Worker init: force the non-interactive Agg backend (workers only rasterize plots to disk)."""
    import matplotlib
    matplotlib.use("Agg")

def _parallel_render(jobs):
    """Fan independent plot jobs out over the allocated cores. Each job is (func, args) that writes one
    PNG/GIF and is pure CPU (matplotlib/numpy/PIL) -- it never touches CUDA -- so the outputs are fully
    independent. Uses the `forkserver` start method: its server is spawned clean (no CUDA/NCCL/thread
    state inherited from this rank) and preloads this module once, so workers fork cheaply instead of
    re-importing the heavy torch stack per job. Runs serially when there's nothing to fan out."""
    if len(jobs) <= 1:
        for func, args in jobs:
            func(*args)
        return
    ctx = multiprocessing.get_context("forkserver")
    ctx.set_forkserver_preload(["utils.manifold_viz"])  # import the stack once in the server, not per worker
    # RENDER_MAX_WORKERS caps the fan-out (the campaign sets it so a background render shares cores with the
    # next trial's training instead of oversubscribing them); unset/0 -> every core (manual offline re-render)
    cap = int(os.environ.get("RENDER_MAX_WORKERS", "0")) or len(os.sched_getaffinity(0))
    workers = min(len(jobs), cap)
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_render_init) as pool:
        for fut in [pool.submit(func, *args) for func, args in jobs]:
            fut.result()  # surface any worker exception in the parent

# Each panel: which projection, what labels color it, the filename stem, and which partition (if any)
# it masks. Panels are never plotted alone -- they're tiled into the stacked leaf/penult pairs and the
# fullset composite (see _GRIDS).
# (proj_key, label_role, color_role, stem, alpha_role)
_PANELS = [
    ("id",      "penult", "penult", "id_penult",          "full"),
    ("id",      "cid",    "leaf",   "id_leaf",            "full"),
    ("ood",     "penult", "penult", "ood_penult",         "full"),
    ("ood",     "cid",    "leaf",   "ood_leaf",           "full"),
    ("fullset", "penult", "penult", "fullset_penult",     "full"),
    ("fullset", "cid",    "leaf",   "fullset_leaf",       "full"),
    ("fullset", "penult", "penult", "fullset_id_penult",  "id"),
    ("fullset", "cid",    "leaf",   "fullset_id_leaf",    "id"),
    ("fullset", "penult", "penult", "fullset_ood_penult", "ood"),
    ("fullset", "cid",    "leaf",   "fullset_ood_leaf",   "ood"),
    ("fullset", "nshot",  "nshot",  "fullset_nshot",      "full"),
]
_STEM_COLOR_ROLE = {stem: color_role for _, _, color_role, stem, _ in _PANELS}  # stem -> leaf/penult/nshot
_STEM_PROJKEY = {stem: proj_key for proj_key, _, _, stem, _ in _PANELS}  # stem -> id/ood/fullset projection

def _resolve_panels(projs, cids_id, cids_ood, penults_id, penults_ood, color_leaf, color_penult, nshot_id, color_nshot):
    """Yield (proj, labels, color_map, alpha, stem) for each panel in `_PANELS`."""
    cids_full = list(cids_id) + list(cids_ood)
    penults_full = list(penults_id) + list(penults_ood)
    is_id = np.arange(len(cids_full)) < len(cids_id)  # ID rows first, then OOD
    labels_by = {
        ("id", "cid"): list(cids_id),       ("id", "penult"): list(penults_id),
        ("ood", "cid"): list(cids_ood),     ("ood", "penult"): list(penults_ood),
        ("fullset", "cid"): cids_full,      ("fullset", "penult"): penults_full,
        # n-shot panel: ID points by bucket, OOD points all black (the sentinel)
        ("fullset", "nshot"): list(nshot_id) + [_OOD_LABEL] * len(cids_ood),
    }
    colors_by = {"leaf": color_leaf, "penult": color_penult, "nshot": color_nshot}
    alphas_by = {"full": 1.0, "id": is_id.astype(float), "ood": (~is_id).astype(float)}
    for proj_key, label_role, color_role, stem, alpha_role in _PANELS:
        yield (projs[proj_key], labels_by[(proj_key, label_role)], colors_by[color_role],
               alphas_by[alpha_role], stem)

# Every standalone output is a flush grid of panels: the leaf(left)/penult(right) subject pairs and
# the 2x4 fullset composite (cols = OOD / ID / full / n-shot; rows = leaf / penult, n-shot's penult
# cell blank). Each entry is (out_name, suptitle-subject, grid-of-stems); None = blank cell.
_GRID = [
    ["fullset_ood_leaf",   "fullset_id_leaf",   "fullset_leaf",   "fullset_nshot"],
    ["fullset_ood_penult", "fullset_id_penult", "fullset_penult", None],
]
_GRIDS = [
    ("id",            "ID",                 [["id_leaf",          "id_penult"]]),
    ("ood",           "OOD",                [["ood_leaf",         "ood_penult"]]),
    ("fullset",       "Full-Set (ID + OOD)",[["fullset_leaf",     "fullset_penult"]]),
    ("fullset_id",    "Full-Set (ID)",      [["fullset_id_leaf",  "fullset_id_penult"]]),
    ("fullset_ood",   "Full-Set (OOD)",     [["fullset_ood_leaf", "fullset_ood_penult"]]),
    ("fullset_panel", "Full-Set",           _GRID),
]
_GRID_STEMS = {stem for _, _, grid in _GRIDS for row in grid for stem in row if stem}  # every plotted stem
_COMPOSITE_COL_TITLES = ["OOD", "ID", "ID + OOD", "n-shot"]  # column headers for the 2x4 fullset composite

def _grid_group(out_name):
    """(panel-group subdir, output file stem) for a grid: the 2x4 fullset composite lands under 7panel/
    renamed to 'fullset', every leaf/penult stacked pair under 2panel/ keeping its grid name. The group
    dir nests above the method dir -> viz/<group>/{tsne,pca}/<stem>.<ext>."""
    return ("7panel", "fullset") if out_name == "fullset_panel" else ("2panel", out_name)

# (out_name, suptitle-subject, leaf_stem, penult_stem) for the cross-method 4panel plots: one per
# 2panel subject (the single-row leaf/penult grids), excluding the 7panel composite. The 4panel tiles
# both projection methods into one figure (PCA top, t-SNE bottom) for a subject, so -- unlike 2panel/
# 7panel -- it has no per-method subdir: viz/4panel/<out_name>.<ext>.
_4PANEL_SUBJECTS = [(out_name, subject, grid[0][0], grid[0][1])
                    for out_name, subject, grid in _GRIDS if out_name != "fullset_panel"]
_QUAD_METHODS = ["PCA", "t-SNE"]  # figure rows, top -> bottom

def _perp_suffix(perplexities, perp):
    """Dir suffix for a t-SNE perplexity: '' for a single-perplexity config (current behavior, plain
    tsne/ and 4panel/), else '_<perp>' so each perplexity gets its own dirs (tsne_15/, 4panel_15/...)."""
    return "" if len(perplexities) == 1 else f"_{perp}"

_CELL = 6.58  # square plotting-cell size (in)

def _grid_layout(nrows, ncols, header=False):
    """(figsize, subplots_adjust kwargs) giving exactly-square cells (so the equal-aspect t-SNE panels
    sit flush) with outer margins reserved for the boundary tick labels + axis labels. Inner cells stay
    flush (wspace=hspace=0); the single-row pairs need no right margin (no right-side axis). `header`
    reserves extra top room for centered per-column titles (composite only) below the suptitle."""
    l, r, b = 0.75, (0.15 if nrows == 1 else 0.75), 0.6
    t = 1.15 if header else 0.55  # inch insets: left, right, bottom, top
    fig_w, fig_h = l + ncols * _CELL + r, b + nrows * _CELL + t
    adjust = dict(left=l / fig_w, right=1 - r / fig_w, bottom=b / fig_h, top=1 - t / fig_h, wspace=0, hspace=0)
    return (fig_w, fig_h), adjust

def _stems_of(grid):
    return [stem for row in grid for stem in row if stem]

def _cell(grid, r, c):
    """Stem at grid[r][c], or None for out-of-bounds / blank cells (used to test edge exposure)."""
    return grid[r][c] if 0 <= r < len(grid) and 0 <= c < len(grid[r]) else None

def _composite_canvas(grid, limits, suptitle, dpi, style):
    """Build a reusable flush grid figure (scaffolding laid out once). `grid` is a list of rows of
    stems (None = blank cell); `limits` maps stem -> (xlim, ylim); `style.legend_by_role` supplies each
    cell's coloring legend (text for leaf/penult, swatch legend for n-shot). `style.col_titles` (composite
    only) is one centered header per column, drawn over the top row below the suptitle. No subplot titles; ticks
    + axis labels appear only on the grid's outer boundary -- a panel exposes a y axis on whichever
    horizontal side has no neighbor and an x axis on the bottom when the cell below is missing (a None
    cell counts as missing), so inner edges stay bare and flush. The single-row leaf(left)/penult(right)
    pairs get an x axis on both panels and a y axis on the left panel only. Returns (fig, {stem:
    scatter}); callers push per-frame data (offsets/colors/marker size) onto the scatters via
    _composite_frame."""
    method, legend_by_role, col_titles, bg_color = style.method, style.legend_by_role, style.col_titles, style.bg_color
    nrows, ncols = len(grid), max(len(row) for row in grid)
    figsize, adjust = _grid_layout(nrows, ncols, header=col_titles is not None)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, squeeze=False)
    fig.suptitle(suptitle, fontsize=22, fontweight="bold", x=adjust["left"], ha="left")  # align to the leftmost plot's left edge
    if col_titles:  # centered per-column headers over the top row (composite only)
        for c, ctitle in enumerate(col_titles):
            axes[0][c].set_title(ctitle, fontsize=16, fontweight="bold")
    pair = nrows == 1  # the side-by-side leaf/penult pairs: x on both, y on the left panel only
    sc_by = {}
    for r, row in enumerate(grid):
        for c in range(ncols):
            ax = axes[r][c]
            stem = row[c] if c < len(row) else None
            if stem is None:
                ax.axis("off")
                continue
            xlim, ylim = limits[stem]  # square origin-centered for t-SNE, data bbox for PCA (_limits_for)
            sc = ax.scatter([], [])  # marker size is set per frame (_composite_frame)
            if bg_color is not None:  # None -> matplotlib default (white) panel background
                ax.set_facecolor(bg_color)
            if method == "t-SNE":
                ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            if pair:
                ax.tick_params(labelsize=9)  # default: bottom + left ticks shown
                ax.set_xlabel(f"{method} Dim. 1", fontsize=11)
                if c == 0:
                    ax.set_ylabel(f"{method} Dim. 2", fontsize=11)
                else:
                    ax.tick_params(left=False, labelleft=False)  # only the left panel keeps its y axis
            else:  # composite: ticks + axis labels only on the grid's outer boundary; inner edges bare
                ax.tick_params(left=False, right=False, bottom=False, top=False, labelsize=9,
                               labelleft=False, labelright=False, labelbottom=False, labeltop=False)
                if _cell(grid, r, c - 1) is None:  # left boundary -> y axis on the left
                    ax.tick_params(axis="y", left=True, labelleft=True)
                    ax.set_ylabel(f"{method} Dim. 2", fontsize=11)
                if _cell(grid, r, c + 1) is None:  # right boundary (edge or blank cell) -> y axis on the right
                    ax.tick_params(axis="y", right=True, labelright=True)
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel(f"{method} Dim. 2", fontsize=11)
                if _cell(grid, r + 1, c) is None:  # bottom boundary -> x axis on the bottom
                    ax.tick_params(axis="x", bottom=True, labelbottom=True)
                    ax.set_xlabel(f"{method} Dim. 1", fontsize=11)
            ax.set_axisbelow(True)  # keep gridlines behind the scatter points (PathCollection zorder 1)
            ax.grid(True, linestyle="--", alpha=0.5)
            _apply_legend(ax, legend_by_role[_STEM_COLOR_ROLE[stem]], fontsize=13)
            sc_by[stem] = sc
    fig.subplots_adjust(**adjust)
    return fig, sc_by

def _composite_frame(sc_by, data, marker_size, order_seed):
    """Push one strobe frame onto every panel: a fresh draw-order shuffle (order_seed) at marker_size."""
    for stem, sc in sc_by.items():
        proj, rgba = data[stem]
        order = np.random.default_rng(order_seed).permutation(len(rgba))
        sc.set_offsets(proj[order])
        sc.set_facecolors(rgba[order])
        sc.set_sizes(np.full(len(rgba), marker_size))

def composite_plot(grid, comp, fpath_png, suptitle, style):
    """Static flush-grid PNG (single frame at the given marker size). `comp` maps stem -> (proj, rgba)."""
    stems = _stems_of(grid)
    limits = {s: _limits_for(style.method, [comp[s][0]]) for s in stems}
    data = {s: (comp[s][0], comp[s][1]) for s in stems}
    fig, sc_by = _composite_canvas(grid, limits, suptitle, _GIF_DPI, style)
    _composite_frame(sc_by, data, style.marker_size, order_seed=0)
    fig.savefig(fpath_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

def composite_strobe_gif(grid, comp, fpath_gif, suptitle, style):
    """Sample-order strobe GIF of a flush grid: n_stoch_layers frames, each a fresh draw-order shuffle
    at the fixed marker size, all panels reshuffled together. `comp` maps stem -> (proj, rgba)."""
    stems = _stems_of(grid)
    limits = {s: _limits_for(style.method, [comp[s][0]]) for s in stems}
    data = {s: (comp[s][0], comp[s][1]) for s in stems}
    fig, sc_by = _composite_canvas(grid, limits, suptitle, _GIF_DPI, style)
    frames = []
    for seed in range(style.n_stoch_layers):
        _composite_frame(sc_by, data, style.marker_size, seed)
        frames.append(_canvas_frame(fig))
    plt.close(fig)
    _save_gif(frames, fpath_gif, style.frame_ms)

def _cids_by(cids_id, cids_ood):
    """{proj key: class-id list} for the three projection subjects (fullset = ID followed by OOD)."""
    return {"id": cids_id, "ood": cids_ood, "fullset": cids_id + cids_ood}

def _resolved_evals(evals, names, stems, methods, cmaps, ema_tau, perp):
    """Yield (name, {(method, stem): (proj, point_colors, alpha)}) one eval at a time, loading each cache
    ONCE and resolving it for every method in `methods` -- so a cross-method caller never re-reads a cache
    per method, and no caller holds more than one eval's data. The t-SNE method uses the `perp` perplexity's
    cached projection, re-oriented against the orientation reference carried across evals in order (the
    same sweep render_evolution/render_eval reproduce); PCA is not oriented (and ignores `perp`). Only the
    panels in `stems` are resolved (the rest of the grid is blank). At most one t-SNE perplexity per call,
    so the reference keys on the proj key alone."""
    color_leaf, color_penult, color_nshot, cid_2_penult, cid_2_nshot = cmaps
    ref = {}  # running per-class CoM orientation reference per t-SNE proj key, carried across evals (t-SNE only)
    for d, name in zip(evals, names):
        tsne_by_perp, pca_projs, cids_id, cids_ood, _ = _load_projections(d)
        penults_id = [cid_2_penult[c] for c in cids_id]
        penults_ood = [cid_2_penult[c] for c in cids_ood]
        nshot_id = [cid_2_nshot[c] for c in cids_id]
        resolved = {}
        for method in methods:
            projs = dict(tsne_by_perp[perp] if method == "t-SNE" else pca_projs)
            if method == "t-SNE":  # re-orient raw t-SNE against the running reference (Procrustes) across evals
                cids_by = _cids_by(cids_id, cids_ood)
                for k in projs:
                    projs[k], ref[k] = orient_tsne(projs[k], cids_by[k], ref.get(k), ema_tau)
            for proj, labels, color_map, alpha, stem in _resolve_panels(
                    projs, cids_id, cids_ood, penults_id, penults_ood, color_leaf, color_penult, nshot_id, color_nshot):
                if stem in stems:
                    resolved[(method, stem)] = (proj, np.array([color_map[label] for label in labels]), alpha)
        yield name, resolved

def composite_evolution_gif(grid, subject, viz_context, evals, names, cmaps, ema_tau, limits, fpath_gif, style, perp):
    """Training-evolution GIF of a flush grid: each eval contributes n_stoch_layers strobe frames, axes
    frozen to the cross-eval union (`limits`, precomputed by render_evolution). Loads + renders one eval's
    cache at a time and streams the frames to disk, so peak memory is independent of the number of
    evals/checkpoints. The suptitle carries the manifold subject (from `_GRIDS`) and the eval name. `perp`
    selects the t-SNE perplexity (ignored for PCA)."""
    stems = _stems_of(grid)
    supt = lambda name: _manifold_title(style.method, viz_context, subject, suffix=f", {name}")
    fig, sc_by = _composite_canvas(grid, limits, supt(names[0]), _GIF_DPI, style)
    def _frames():  # generator: render frames lazily (one eval loaded at a time) so they stream to disk
        for name, resolved in _resolved_evals(evals, names, stems, (style.method,), cmaps, ema_tau, perp):
            fig.suptitle(supt(name), fontsize=22, fontweight="bold", x=fig.subplotpars.left, ha="left")  # align to the leftmost plot's left edge
            data = {s: (resolved[(style.method, s)][0], _rgba(resolved[(style.method, s)][1], resolved[(style.method, s)][2])) for s in stems}
            for seed in range(style.n_stoch_layers):
                _composite_frame(sc_by, data, style.marker_size, seed)
                yield _canvas_frame(fig)
    _save_gif_stream(_frames(), fpath_gif, style.frame_ms, palette_sample=style.n_stoch_layers)
    plt.close(fig)

def _quad_canvas(leaf_stem, penult_stem, limits, suptitle, dpi, style):
    """Build the 2x2 cross-method figure: rows = PCA (top) / t-SNE (bottom), cols = leaf (left) /
    penult (right) coloring of the same subject. Cells are flush (wspace=hspace=0); each cell exposes
    ticks + a method-named axis label only on whichever of its edges lie on the grid's outer boundary
    (inner shared edges stay bare): x on the top row (PCA Dim. 1, top) and bottom row (t-SNE Dim. 1,
    bottom); y on the left column (left) and right column (right), each labeled with its row's method.
    `limits` maps (method, stem) -> (xlim, ylim). Returns (fig, {(method, stem): scatter}); callers push
    per-frame data via _composite_frame (it keys `data` by the same (method, stem) tuples)."""
    cols = [leaf_stem, penult_stem]
    # inch insets: l/r reserve the left/right y axes, b the bottom x axis, t the top x axis + suptitle gap
    l, r, b, t = 0.75, 0.75, 0.6, 1.4
    fig_w, fig_h = l + 2 * _CELL + r, b + 2 * _CELL + t
    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h), dpi=dpi, squeeze=False)
    fig.suptitle(suptitle, fontsize=22, fontweight="bold", x=l / fig_w, ha="left")  # align to the leftmost plot's left edge
    sc_by = {}
    for ri, method in enumerate(_QUAD_METHODS):
        for ci, stem in enumerate(cols):
            ax = axes[ri][ci]
            xlim, ylim = limits[(method, stem)]  # square origin-centered for t-SNE, data bbox for PCA
            sc = ax.scatter([], [])  # marker size is set per frame (_composite_frame)
            if style.bg_color is not None:
                ax.set_facecolor(style.bg_color)
            if method == "t-SNE":
                ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            # flush perimeter axes: ticks + a method-named label only on this cell's outer-boundary edges
            ax.tick_params(left=False, right=False, bottom=False, top=False, labelsize=9,
                           labelleft=False, labelright=False, labelbottom=False, labeltop=False)
            if ci == 0:  # left boundary -> y on the left
                ax.tick_params(axis="y", left=True, labelleft=True)
                ax.set_ylabel(f"{method} Dim. 2", fontsize=11)
            else:  # right boundary -> y on the right
                ax.tick_params(axis="y", right=True, labelright=True)
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(f"{method} Dim. 2", fontsize=11)
            if ri == 0:  # top boundary -> x on the top
                ax.tick_params(axis="x", top=True, labeltop=True)
                ax.xaxis.set_label_position("top")
                ax.set_xlabel(f"{method} Dim. 1", fontsize=11)
            else:  # bottom boundary -> x on the bottom
                ax.tick_params(axis="x", bottom=True, labelbottom=True)
                ax.set_xlabel(f"{method} Dim. 1", fontsize=11)
            ax.set_axisbelow(True)  # keep gridlines behind the scatter points (PathCollection zorder 1)
            ax.grid(True, linestyle="--", alpha=0.5)
            _apply_legend(ax, style.legend_by_role[_STEM_COLOR_ROLE[stem]], fontsize=13)
            sc_by[(method, stem)] = sc
    fig.subplots_adjust(left=l / fig_w, right=1 - r / fig_w, bottom=b / fig_h, top=1 - t / fig_h,
                        wspace=0, hspace=0)
    return fig, sc_by

def quad_render(leaf_stem, penult_stem, data, fpath, suptitle, style):
    """Static PNG (n_stoch_layers == 1) or sample-order strobe GIF (>1) of the 2x2 cross-method grid.
    `data` maps (method, stem) -> (proj, rgba). Per-cell axis limits come from each cell's own method."""
    limits = {k: _limits_for(k[0], [data[k][0]]) for k in data}  # k = (method, stem)
    fig, sc_by = _quad_canvas(leaf_stem, penult_stem, limits, suptitle, _GIF_DPI, style)
    if style.n_stoch_layers == 1:
        _composite_frame(sc_by, data, style.marker_size, order_seed=0)
        fig.savefig(fpath, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        frames = []
        for seed in range(style.n_stoch_layers):
            _composite_frame(sc_by, data, style.marker_size, seed)
            frames.append(_canvas_frame(fig))
        plt.close(fig)
        _save_gif(frames, fpath, style.frame_ms)

def quad_evolution_gif(leaf_stem, penult_stem, subject, viz_context, evals, names, cmaps, ema_tau,
                       limits, fpath_gif, style, perp):
    """Cross-method (PCA top / t-SNE bottom) training-evolution GIF for one subject's leaf/penult pair, at
    t-SNE perplexity `perp`. Sweeps both methods' caches in lockstep (one eval per method loaded at a time)
    and streams the frames to disk, so peak memory is independent of the eval count; axes frozen to the
    precomputed cross-eval union (`limits`, keyed (method, stem))."""
    stems = {leaf_stem, penult_stem}
    supt = lambda name: _manifold_title("PCA + t-SNE", viz_context, subject, suffix=f", {name}")
    fig, sc_by = _quad_canvas(leaf_stem, penult_stem, limits, supt(names[0]), _GIF_DPI, style)
    def _frames():  # generator: one cache load per eval, resolved for both methods, streamed to disk
        for name, resolved in _resolved_evals(evals, names, stems, _QUAD_METHODS, cmaps, ema_tau, perp):
            fig.suptitle(supt(name), fontsize=22, fontweight="bold", x=fig.subplotpars.left, ha="left")
            data = {k: (resolved[k][0], _rgba(resolved[k][1], resolved[k][2]))
                    for k in ((m, s) for m in _QUAD_METHODS for s in stems)}
            for seed in range(style.n_stoch_layers):
                _composite_frame(sc_by, data, style.marker_size, seed)
                yield _canvas_frame(fig)

    _save_gif_stream(_frames(), fpath_gif, style.frame_ms, palette_sample=style.n_stoch_layers)
    plt.close(fig)

def _render_grids(tsne_by_perp, pca_projs, perplexities, cids_id, cids_ood, penults_id, penults_ood,
                  color_leaf, color_penult, nshot_id, color_nshot, legend_by_role,
                  dpath_vis, cfg_manifold_viz, viz_context, tag, plot_flags):
    """Rank-0 render for one eval from ALREADY-ORIENTED PCA + per-perplexity t-SNE projections, into
    dpath_vis/: the per-method stacked leaf(top)/penult(bottom) pairs (ID-only, OOD-only, the combined
    ID+OOD projection and its two partition-masked variants) under 2panel/<method>/, the 2x4 fullset
    composite under 7panel/<method>/, and the cross-method PCA-over-t-SNE 4panel (one per 2panel subject).
    A t-SNE is rendered for each perplexity in `perplexities`; with more than one, the t-SNE method dir
    and the 4panel dir are suffixed by perplexity (tsne_15/, 4panel_15/, ...), else they stay plain. PCA is
    shared. `plot_flags` (dev.manifold_viz) gates which groups are emitted. Each output is a static PNG
    when n_stoch_layers == 1, else a strobe GIF. Pure renderer -- orientation/coloring/cache are the
    caller's job (render_eval)."""
    marker_size = DATASET2MARKER_SIZE[viz_context.dataset]
    n_stoch_layers = cfg_manifold_viz["n_stoch_layers"]
    frame_ms = cfg_manifold_viz["eval_duration"] / n_stoch_layers  # per-frame ms so each eval's GIF lasts eval_duration
    bg_color = cfg_manifold_viz["bg_color"]
    suffix = f", {tag}"
    def bake(projs):  # every panel's (proj, rgba); the grids below tile them into the pairs/composite/4panel
        return {stem: (proj, _rgba(np.array([color_map[label] for label in labels]), alpha))
                for proj, labels, color_map, alpha, stem in _resolve_panels(
                    projs, cids_id, cids_ood, penults_id, penults_ood, color_leaf, color_penult, nshot_id, color_nshot)}
    comp_pca = bake(pca_projs)
    comp_tsne_by_perp = {perp: bake(tsne_by_perp[perp]) for perp in perplexities}
    render = composite_plot if n_stoch_layers == 1 else composite_strobe_gif  # static PNG vs strobe GIF
    ext = "png" if n_stoch_layers == 1 else "gif"
    jobs = []  # render jobs fanned out across cores below
    # per-method grids: 2panel stacked pairs + 7panel composite, under viz/<group>/<method_dir>/ -- PCA once
    # plus one t-SNE per perplexity (method_dir suffixed when there's more than one)
    grid_targets = [("PCA", "pca", comp_pca)] + [
        ("t-SNE", f"tsne{_perp_suffix(perplexities, perp)}", comp_tsne_by_perp[perp]) for perp in perplexities]
    for method, method_dir, comp in grid_targets:
        for out_name, subject, grid in _GRIDS:
            group, stem = _grid_group(out_name)
            if not plot_flags[f"plot_{group}"]:  # 2panel / 7panel toggles (dev.manifold_viz)
                continue
            suptitle = _manifold_title(method, viz_context, subject, suffix=suffix)
            col_titles = _COMPOSITE_COL_TITLES if out_name == "fullset_panel" else None
            style = RenderStyle(method, marker_size, legend_by_role, n_stoch_layers, frame_ms, bg_color, col_titles)
            sub = {s: comp[s] for s in _stems_of(grid)}  # only this grid's panels (smaller to ship to a worker)
            fpath = dpath_vis / group / method_dir / f"{stem}.{ext}"
            fpath.parent.mkdir(parents=True, exist_ok=True)
            jobs.append((render, (grid, sub, fpath, suptitle, style)))
    # cross-method 4panel (PCA top / t-SNE bottom): one per perplexity, under viz/4panel{_<perp>}/
    if plot_flags["plot_4panel"]:
        quad_style = RenderStyle(None, marker_size, legend_by_role, n_stoch_layers, frame_ms, bg_color)
        for perp in perplexities:
            comp_tsne = comp_tsne_by_perp[perp]
            quad_dir = f"4panel{_perp_suffix(perplexities, perp)}"
            for out_name, subject, leaf_stem, penult_stem in _4PANEL_SUBJECTS:
                suptitle = _manifold_title("PCA + t-SNE", viz_context, subject, suffix=suffix)
                data = {("PCA", s): comp_pca[s] for s in (leaf_stem, penult_stem)}
                data.update({("t-SNE", s): comp_tsne[s] for s in (leaf_stem, penult_stem)})
                fpath = dpath_vis / quad_dir / f"{out_name}.{ext}"
                fpath.parent.mkdir(parents=True, exist_ok=True)
                jobs.append((quad_render, (leaf_stem, penult_stem, data, fpath, suptitle, quad_style)))
    _parallel_render(jobs)

def _compute_projections(embs_id, embs_ood, cfg_tsne):
    """COLLECTIVE -- must be entered by every rank together. Compute the RAW t-SNE (sharded across ranks
    when world_size > 1, see `_tsne_torch`) and PCA projections for id/ood/fullset from the all-gathered
    image embeddings (`embs_id`/`embs_ood` are the per-rank GPU tensors, identical on every rank). PCA is
    computed redundantly per rank -- it is deterministic (same input + fixed seed) and feeds the shared
    t-SNE init; the sharded t-SNE all-gathers the full layout each iteration, so every rank returns
    identical (tsne_by_perp, pca_projs). PCA is shared across perplexities (it only feeds the t-SNE init);
    a t-SNE is computed for every perplexity in `cfg_tsne["perplexity"]` (a list), all sharing the one
    high-dim distance matrix built per proj key. Returns (tsne_by_perp, pca_projs) with
    tsne_by_perp = {perplexity: {id/ood/fullset: proj}}."""
    embs = {"id": embs_id, "ood": embs_ood, "fullset": torch.cat([embs_id, embs_ood], dim=0)}
    pca_projs = {k: compute_pca(e.detach().cpu().numpy()) for k, e in embs.items()}
    perps = cfg_tsne["perplexity"]
    # one t-SNE per (key, perplexity); compute_tsne builds the key's high-dim distance matrix once and
    # reuses it across all perplexities -> {key: {perp: proj}}, regrouped below to {perp: {key: proj}}
    tsne_by_key = {
        k: compute_tsne(embs[k], perplexities=perps, init=_pca_init(pca_projs[k]), n_iter=cfg_tsne["n_iter"])
        for k in embs
    }
    tsne_by_perp = {perp: {k: tsne_by_key[k][perp] for k in embs} for perp in perps}
    return tsne_by_perp, pca_projs

def _build_color_maps(viz_context, cids_all, cfg_color):
    """Per-dataset color maps + label lookups for the manifold panels, given the eval set's full
    cid list (`cids_all` = ID + OOD). Colors are count-ordered over `cids_all` so a class keeps its
    color across every plot/eval. Returns (color_leaf, color_penult, color_nshot, cid_2_penult,
    cid_2_nshot, nst_names)."""
    cid_2_penult = load_cid_2_penult(viz_context.dataset)
    cid_2_nshot, nst_names = load_cid_2_nshot(viz_context.dataset, viz_context.split, viz_context.eval_type)
    color_leaf = assign_colors(cid_2_penult.keys(), Counter(cids_all), cfg_color, hue_offset=0.0)
    color_penult = assign_colors(cid_2_penult.values(), Counter(cid_2_penult[c] for c in cids_all), cfg_color, hue_offset=0.5)
    color_nshot = nshot_color_map(nst_names)  # bucket colors matching the learning curves (+ OOD black)
    return color_leaf, color_penult, color_nshot, cid_2_penult, cid_2_nshot, nst_names

def compute_projections(eval_bundle_id, eval_bundle_ood, dpath_cache, cfg_manifold_viz):
    """COLLECTIVE -- every rank must enter. Compute the raw (sharded) t-SNE + PCA from the live,
    all-gathered eval embeddings and, on rank 0, cache them to dpath_cache/projections.npz. This is the
    training pipeline's ONLY in-loop viz work; orientation, coloring, and rendering are a separate pass
    over the cache (render_eval / render_evolution), kept off the collective path. Returns None."""
    _log("computing projections")
    # ALL RANKS: sharded t-SNE (one per perplexity) + PCA projections (collective ops inside)
    tsne_by_perp, pca_projs = _compute_projections(
        eval_bundle_id["embs_img"], eval_bundle_ood["embs_img"], cfg_manifold_viz["tsne"])
    if dist.is_initialized() and dist.get_rank() != 0:
        return  # non-rank-0 ranks only participate in the collective compute
    dpath_cache.mkdir(parents=True, exist_ok=True)
    cache = {"cids_id": np.asarray(eval_bundle_id["cids_img"]), "cids_ood": np.asarray(eval_bundle_ood["cids_img"]),
             "perplexities": np.asarray(list(tsne_by_perp))}
    for k in pca_projs:
        cache[f"pca_{k}"] = pca_projs[k]
    for perp, tsne in tsne_by_perp.items():
        for k in tsne:
            cache[f"tsne_{perp}_{k}"] = tsne[k]
    np.savez(dpath_cache / "projections.npz", **cache)
    _log("projections complete")

def _load_projections(dpath_cache):
    """Load an eval's cached raw projections from dpath_cache/projections.npz:
    (tsne_by_perp, pca_projs, cids_id, cids_ood, perplexities). tsne_by_perp = {perplexity: {id/ood/fullset:
    proj}}; pca_projs is keyed id/ood/fullset (shared across perplexities)."""
    npz = np.load(dpath_cache / "projections.npz")
    keys = ("id", "ood", "fullset")
    perplexities = [int(p) for p in npz["perplexities"]]
    tsne_by_perp = {perp: {k: npz[f"tsne_{perp}_{k}"] for k in keys} for perp in perplexities}
    return (tsne_by_perp, {k: npz[f"pca_{k}"] for k in keys},
            list(npz["cids_id"]), list(npz["cids_ood"]), perplexities)

def _ordered_eval_dirs(dpath_evals):
    """Eval dirs that hold a cached projections.npz, in chronological order (_base, thresholds, final)."""
    return sorted((d for d in dpath_evals.iterdir() if (d / "projections.npz").exists()),
                  key=lambda d: _eval_sort_key(d.name))

def _ema_through(dpath_evals, eval_name, ema_tau):
    """Accumulate the t-SNE orientation reference over the evals chronologically BEFORE `eval_name`, so it
    seeds `eval_name`'s orientation -- identical to that eval's frame in the evolution GIF (which sweeps
    the same caches in the same order). Recomputed from the on-disk caches each call, so the pipeline
    carries no live/resume orientation state. Returns {key: ref} ({} when `eval_name` is the first eval)."""
    ref = {}  # keyed (perplexity, proj key) -- each perplexity's t-SNE has its own orientation reference
    for d in _ordered_eval_dirs(dpath_evals):
        if d.name == eval_name:
            break
        tsne_by_perp, _, cids_id, cids_ood, _ = _load_projections(d)
        cids_by = _cids_by(cids_id, cids_ood)
        for perp, tsne in tsne_by_perp.items():
            for k in tsne:
                _, ref[(perp, k)] = orient_tsne(tsne[k], cids_by[k], ref.get((perp, k)), ema_tau)
    return ref

def _save_orient_ref(dpath_eval, ref, ema_tau):
    """Cache this eval's OUTGOING orientation reference (the running per-(perplexity, proj key) {class: CoM}
    through this eval) so the next eval's render reads it in O(1) instead of re-sweeping every prior eval
    (see `_incoming_ref`). ema_tau is stored alongside so a render under a different smoothing factor
    recomputes rather than reusing a stale reference."""
    save_pickle({"ema_tau": ema_tau, "ref": ref}, dpath_eval / "orient_ref.pkl")

def _load_orient_ref(dpath_eval, ema_tau):
    """This eval's cached outgoing orientation reference, or None when absent or written under a different
    ema_tau (forcing a correct recompute)."""
    fpath = dpath_eval / "orient_ref.pkl"
    if not fpath.exists():
        return None
    blob = load_pickle(fpath)
    return blob["ref"] if blob["ema_tau"] == ema_tau else None

def _incoming_ref(dpath_evals, eval_name, ema_tau):
    """The orientation reference accumulated over the evals chronologically BEFORE `eval_name`, read in
    O(1) from the immediately-preceding eval's cached outgoing reference -- the memo of the same
    chronological orient sweep `_ema_through` would do. Falls back to recomputing from the raw caches
    (`_ema_through`) when that cache is absent or was written under a different ema_tau, so the result is
    always identical to the full sweep. Returns {(perplexity, proj key): {class: CoM}} ({} for the first eval)."""
    ordered = _ordered_eval_dirs(dpath_evals)
    idx = [d.name for d in ordered].index(eval_name)
    if idx == 0:
        return {}
    cached = _load_orient_ref(ordered[idx - 1], ema_tau)
    return cached if cached is not None else _ema_through(dpath_evals, eval_name, ema_tau)

def _no_panels_enabled(plot_flags):
    """True when every manifold-viz panel group is toggled off (dev.manifold_viz.plot_2/4/7panel) -- the
    render passes then short-circuit, doing no setup/render and writing no (empty) viz dirs."""
    return not (plot_flags["plot_2panel"] or plot_flags["plot_4panel"] or plot_flags["plot_7panel"])

@rank0
def render_eval(dpath_evals, eval_name, cfg_manifold_viz, viz_context, plot_flags):
    """Rank-0. Render one eval's per-eval plots from its cached projections
    (dpath_evals/<eval_name>/projections.npz) into <eval_name>/viz/. The t-SNE orientation uses the
    reference accumulated over the prior evals on disk, so it matches that eval's frame in the evolution
    GIF -- and needs no live state. `plot_flags` (dev.manifold_viz) gates which panel groups are emitted."""
    if _no_panels_enabled(plot_flags):
        return
    dpath_eval = dpath_evals / eval_name
    tsne_by_perp, pca_projs, cids_id, cids_ood, perplexities = _load_projections(dpath_eval)
    cids_all = list(cids_id) + list(cids_ood)
    # color maps span the whole dataset so a class is colored identically in every plot; colors
    # are assigned in order of how many plotted (ID+OOD) samples each class/penult-group has
    color_leaf, color_penult, color_nshot, cid_2_penult, cid_2_nshot, nst_names = \
        _build_color_maps(viz_context, cids_all, cfg_manifold_viz["color"])
    penults_id = [cid_2_penult[c] for c in cids_id]
    penults_ood = [cid_2_penult[c] for c in cids_ood]
    nshot_id = [cid_2_nshot[c] for c in cids_id]  # OOD samples are drawn black, not bucketed
    ema_tau = cfg_manifold_viz["tsne"]["ema_tau"]
    ref = _incoming_ref(dpath_evals, eval_name, ema_tau)  # reference through the prior evals (O(1) cache read)
    cids_by = _cids_by(cids_id, cids_ood)
    tsne_oriented = {}
    for perp, tsne in tsne_by_perp.items():
        tsne_oriented[perp] = {}
        for k in tsne:
            tsne_oriented[perp][k], ref[(perp, k)] = orient_tsne(tsne[k], cids_by[k], ref.get((perp, k)), ema_tau)
    _save_orient_ref(dpath_eval, ref, ema_tau)  # cache outgoing ref (before plotting) so the next eval reads it in O(1)
    tag = "base" if eval_name == "_base" else eval_name
    _render_grids(tsne_oriented, pca_projs, perplexities, cids_id, cids_ood, penults_id, penults_ood,
                  color_leaf, color_penult, nshot_id, color_nshot, _legend_specs(color_nshot, nst_names),
                  dpath_eval / "viz", cfg_manifold_viz, viz_context, tag, plot_flags)

def _eval_sort_key(name):
    """Chronological order of eval dirs: _base first, numeric thresholds ascending, final last."""
    if name == "_base":
        return (0, 0)
    if name == "final":
        return (2, 0)
    return (1, int(name[:-1]) * 1000 if name.endswith("k") else int(name))

@rank0
def _evolution_limits(evals, ema_tau):
    """One streaming pass over the caches accumulating the cross-eval axis bounds per (method, proj key)
    while holding only one eval at a time: the PCA bounding box (running min/max) and the t-SNE square
    bound (running max |coord| over ORIENTED projections -- orientation rotates points about the origin,
    so the bound must be taken post-orientation, with the same orientation sweep as the render). Reuses
    _common_limits/_square_limits on the accumulated extremes so the frozen axes are identical to
    materializing every eval. PCA is shared across perplexities; t-SNE bounds are accumulated per
    perplexity. Returns {(method, perp, proj key): (xlim, ylim)} (perp is None for PCA) for proj key in
    id/ood/fullset."""
    keys = ("id", "ood", "fullset")
    pca_lo, pca_hi = {k: None for k in keys}, {k: None for k in keys}
    tsne_absmax = {}  # (perp, k) -> running max |coord| over oriented projections
    ref = {}          # (perp, k) -> running orientation reference (per-class CoM)
    for d in evals:
        tsne_by_perp, pca_projs, cids_id, cids_ood, _ = _load_projections(d)
        cids_by = _cids_by(cids_id, cids_ood)
        for k in keys:
            lo, hi = pca_projs[k].min(axis=0), pca_projs[k].max(axis=0)
            pca_lo[k] = lo if pca_lo[k] is None else np.minimum(pca_lo[k], lo)
            pca_hi[k] = hi if pca_hi[k] is None else np.maximum(pca_hi[k], hi)
        for perp, tsne in tsne_by_perp.items():
            for k in keys:
                oriented, ref[(perp, k)] = orient_tsne(tsne[k], cids_by[k], ref.get((perp, k)), ema_tau)
                tsne_absmax[(perp, k)] = max(tsne_absmax.get((perp, k), 0.0), float(np.abs(oriented).max()))
    limits = {("PCA", None, k): _common_limits([np.stack([pca_lo[k], pca_hi[k]])]) for k in keys}
    for (perp, k), absmax in tsne_absmax.items():
        limits[("t-SNE", perp, k)] = _square_limits([np.array([[absmax, absmax]])])
    return limits

def render_evolution(dpath_evals, dpath_out, cfg_manifold_viz, viz_context, plot_flags):
    """Rank-0. Assemble one GIF per grid (`_GRIDS`) showing the training evolution
    (_base -> ... -> final): each eval contributes the strobe schedule's frames, then the GIF hard-cuts
    to the next eval, axes/gridlines frozen across evals so only the points move. Reads each eval's
    cached projections.npz and writes the per-method grids under dpath_out/{2panel,7panel}/<method>/ plus
    the cross-method PCA-over-t-SNE 4panel under dpath_out/4panel/. Re-orients t-SNE by aligning each eval
    to a running reference swept across evals -- the same orientation `render_eval` reproduces per eval. Caches
    are streamed one eval at a time (frozen limits precomputed in a single pass; each render worker then
    loads + renders one cache at a time), so peak memory doesn't scale with the number of checkpoints."""
    if _no_panels_enabled(plot_flags):
        return
    evals = _ordered_eval_dirs(dpath_evals)
    if not evals:
        return
    names = ["base" if d.name == "_base" else d.name for d in evals]

    cfg_color = cfg_manifold_viz["color"]
    marker_size = DATASET2MARKER_SIZE[viz_context.dataset]
    bg_color = cfg_manifold_viz["bg_color"]
    n_stoch_layers = cfg_manifold_viz["n_stoch_layers"]
    frame_ms = cfg_manifold_viz["eval_duration"] / n_stoch_layers  # per-frame ms so each eval shows for eval_duration
    ema_tau = cfg_manifold_viz["tsne"]["ema_tau"]
    # eval set is fixed across checkpoints, so any eval's cids give the same (count-ordered) colors
    _, _, cids_id, cids_ood, perplexities = _load_projections(evals[-1])
    color_leaf, color_penult, color_nshot, cid_2_penult, cid_2_nshot, nst_names = \
        _build_color_maps(viz_context, cids_id + cids_ood, cfg_color)
    cmaps = (color_leaf, color_penult, color_nshot, cid_2_penult, cid_2_nshot)  # shipped to workers (O(classes))
    legends = _legend_specs(color_nshot, nst_names)  # per-color-role coloring legend for every panel
    limits_by = _evolution_limits(evals, ema_tau)  # frozen axes per (method, proj key), single streaming pass

    jobs = []  # one evolution-GIF job per (render target, grid); fanned out across cores below
    # per-method grids: PCA once + one t-SNE per perplexity (method dir suffixed when there's more than one)
    grid_targets = [("PCA", "pca", None)] + [
        ("t-SNE", f"tsne{_perp_suffix(perplexities, perp)}", perp) for perp in perplexities]
    for method, method_dir, perp in grid_targets:
        for out_name, subject, grid in _GRIDS:
            group, stem = _grid_group(out_name)
            if not plot_flags[f"plot_{group}"]:  # 2panel / 7panel toggles (dev.manifold_viz)
                continue
            stems = _stems_of(grid)
            col_titles = _COMPOSITE_COL_TITLES if out_name == "fullset_panel" else None
            style = RenderStyle(method, marker_size, legends, n_stoch_layers, frame_ms, bg_color, col_titles)
            limits = {s: limits_by[(method, perp, _STEM_PROJKEY[s])] for s in stems}
            fpath = dpath_out / group / method_dir / f"{stem}.gif"
            fpath.parent.mkdir(parents=True, exist_ok=True)
            jobs.append((composite_evolution_gif, (grid, subject, viz_context, evals, names, cmaps,
                                                   ema_tau, limits, fpath, style, perp)))
    # cross-method 4panel evolution GIFs (PCA top / t-SNE bottom): one per perplexity, under viz/4panel{_<perp>}/
    if plot_flags["plot_4panel"]:
        quad_style = RenderStyle(None, marker_size, legends, n_stoch_layers, frame_ms, bg_color)
        for perp in perplexities:
            quad_dir = f"4panel{_perp_suffix(perplexities, perp)}"
            for out_name, subject, leaf_stem, penult_stem in _4PANEL_SUBJECTS:
                limits = {("PCA", s): limits_by[("PCA", None, _STEM_PROJKEY[s])] for s in (leaf_stem, penult_stem)}
                limits.update({("t-SNE", s): limits_by[("t-SNE", perp, _STEM_PROJKEY[s])] for s in (leaf_stem, penult_stem)})
                fpath = dpath_out / quad_dir / f"{out_name}.gif"
                fpath.parent.mkdir(parents=True, exist_ok=True)
                jobs.append((quad_evolution_gif, (leaf_stem, penult_stem, subject, viz_context, evals, names,
                                                  cmaps, ema_tau, limits, fpath, quad_style, perp)))
    _log("rendering evolution")
    _parallel_render(jobs)
    _log("evolution complete")
