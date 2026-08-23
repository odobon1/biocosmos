import numpy as np
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, Patch
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
    """Static styling for the composite renderers. Bundled because the same set rides through
    generate_* -> composite_plot/evolving_gif -> _composite_canvas, and is pickled to the render
    workers. `method` (a `_METHODS` entry) and `col_titles` vary per (method, grid); the rest are fixed
    per generate_* call (composite_plot, which writes a still, ignores frame_ms)."""
    method: str
    marker_size: int
    legend_by_role: dict
    frame_ms: float  # evolving-GIF duration per eval; each eval contributes one frame
    bg_color: str | None
    col_titles: list | None = None


def _manifold_title(method, viz_context, subject, suffix=""):
    """Suptitle for a manifold grid, e.g. 't-SNE: Joint (ID) Validation -- hp, Nymphalidae, 50k'."""
    return f"{method}: {subject} {_EVAL_ALIAS2NAME[viz_context.eval_type]} -- {viz_context.setting}, {DATASET_ALIAS2NAME[viz_context.dataset]}{suffix}"

_GIF_DPI = 100  # evolving-GIF frame resolution (lower than the 300-dpi static PNGs)
_OOD_LABEL = "__OOD__"  # sentinel label for OOD points in the n-shot panel (always drawn black)

# The projection methods, in figure order (left -> right across the cross-method grids). Each name maps to
# its output subdir, which doubles as its `<prefix>_<proj key>` key prefix in the projection caches. t-SNE and
# UMAP produce layouts whose axes and handedness are arbitrary (_RIGID): square origin-centered axes, and
# cross-eval alignment by full rotation + reflection. PCA's axes are the principal directions and must not be
# rotated, so it keeps its data bounding box and only has its per-component SIGN aligned across evals.
_METHODS = ["PCA", "t-SNE", "UMAP", "UMAP-sphere"]
_METHOD_DIR = {"PCA": "pca", "t-SNE": "tsne", "UMAP": "umap", "UMAP-sphere": "umap_sphere"}
_RIGID = {"t-SNE", "UMAP"}        # arbitrary planar frame: square origin-centered axes, 2D Procrustes
_SPHERICAL = {"UMAP-sphere"}      # layout lives ON a 2-sphere: 3D cell + opaque ball, 3D Procrustes
_PROJ_KEYS = ("id", "ood", "joint")  # the projection subjects every method is fit for

# The embeddings are L2-normalized (models.py), so they already lie on a unit hypersphere and cosine is
# the distance the loss actually uses. Both UMAP variants therefore run on the same cosine kNN graph --
# which is what lets them share one neighbor search (see `_umap_knn`). On normalized vectors cosine and
# euclidean induce the SAME neighbor sets anyway, so this is a free change of units for the flat fit.
_UMAP_METRIC = "cosine"
# (elev, azim) of the single viewpoint every sphere panel is drawn from. elev=0 puts the camera in the
# equatorial plane, which is what makes the north pole land dead top of the disc and the equator run
# horizontally through its center; azim=0 then faces the zero azimuth the bootstrap frame assigns to the
# second-largest class (see `orient_sphere`).
_SPHERE_VIEW = (0.0, 0.0)
_DRAW_ORDER_SEED = 0          # fixed: the draw order is shuffled, but reproducibly
_SPHERE_MARGIN = 1.05         # axes bound: the unit ball plus room for rim markers, which sit at r == 1
_SPHERE_GRID_STEP = 30        # degrees between parallels and between meridians

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

def _knn(Xr, X, k, r0, chunk):
    """Chunked exact k-nearest-neighbor search for the local rows `Xr` against the full `X`, self
    excluded (the local rows are X[r0:r0+len(Xr)]): returns (d2, idx) -- (n_local, k) squared
    distances and global neighbor column indices. Row-chunked so the transient distance buffer is
    (chunk, N), never (n_local, N)."""
    nl = Xr.shape[0]
    dev = X.device
    d2 = torch.empty((nl, k), device=dev)
    idx = torch.empty((nl, k), dtype=torch.long, device=dev)
    for s in range(0, nl, chunk):
        Dc = torch.cdist(Xr[s:s + chunk], X).pow_(2)  # (c, N)
        a = torch.arange(Dc.shape[0], device=dev)
        Dc[a, r0 + s + a] = float("inf")  # exclude self from the neighbor set
        d2[s:s + chunk], idx[s:s + chunk] = Dc.topk(k, dim=1, largest=False)
    return d2, idx

def _allgather_rows(Yr, counts):
    """All-gather the variable-height local row blocks `Yr` (n_local x d, any dtype) into the full
    (N x d). Pads each block to the max height (shards can differ by one row) for the uniform-shape
    all_gather, then slices each rank's real rows back out."""
    world = dist.get_world_size()
    maxc, d = max(counts), Yr.shape[1]
    buf = torch.zeros((maxc, d), dtype=Yr.dtype, device=Yr.device)
    buf[:Yr.shape[0]] = Yr
    parts = [torch.empty((maxc, d), dtype=Yr.dtype, device=Yr.device) for _ in range(world)]
    dist.all_gather(parts, buf)
    return torch.cat([parts[q][:counts[q]] for q in range(world)], dim=0)

def _hbeta_search(D2, perplexity, tol=1e-5, max_iter=100):
    """Per-point Gaussian precision (beta = 1/2sigma^2) tuned so each row of the conditional
    affinity matrix has the target `perplexity`, via vectorized binary search over all points at
    once. `D2` is the (rows, k) squared distances from each point to its k nearest neighbors (self
    excluded, see `_knn`); returns the row-normalized conditionals P_{j|i} over those neighbors.
    Mirrors sklearn's _binary_search_perplexity in its default (Barnes-Hut) mode: conditionals are
    restricted to the k = 3*perplexity nearest neighbors, whose dropped tail decays like
    exp(-beta*d^2) and is numerically negligible."""
    n = D2.shape[0]
    dev = D2.device
    beta = torch.ones(n, 1, device=dev)
    betamin = torch.full((n, 1), -float("inf"), device=dev)
    betamax = torch.full((n, 1), float("inf"), device=dev)
    logU = math.log(perplexity)
    for _ in range(max_iter):
        Pexp = torch.exp(-D2 * beta)
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
    Pexp = torch.exp(-D2 * beta)
    return Pexp / Pexp.sum(1, keepdim=True).clamp_min(1e-12)

def _sparse_joint_p(d2, idx, perplexity, n, r0, counts):
    """kNN conditionals -> symmetrized joint P for this rank's rows, as CSR pieces.

    Runs the perplexity search on the (n_local, k) neighbor distances, then symmetrizes to the
    joint P = (P_cond + P_cond^T) / (2n) (sums to 1 globally). The transpose contribution for a
    local row i -- P_{i|j} for every j that selected i as a neighbor -- lives on j's rank, so under
    DDP every rank's conditional edges are all-gathered first (edge counts per rank are static:
    counts[q] * k). Symmetrization is a COO coalesce: each edge is emitted with its transpose and
    duplicate coordinates sum. Returns (rowptr, cols, vals): this rank's rows of the joint P in CSR
    form with local row indexing (coalesce yields row-major sorted indices, so the mask-slice below
    preserves CSR order)."""
    nl, k = d2.shape
    dev = d2.device
    Pc = _hbeta_search(d2, perplexity)
    rows = (r0 + torch.arange(nl, device=dev)).repeat_interleave(k)
    cols = idx.reshape(-1)
    vals = Pc.reshape(-1)
    if dist.is_initialized() and dist.get_world_size() > 1:
        edge_counts = [c * k for c in counts]
        rows = _allgather_rows(rows[:, None], edge_counts)[:, 0]
        cols = _allgather_rows(cols[:, None], edge_counts)[:, 0]
        vals = _allgather_rows(vals[:, None], edge_counts)[:, 0]
    indices = torch.stack([torch.cat([rows, cols]), torch.cat([cols, rows])])
    P = torch.sparse_coo_tensor(indices, torch.cat([vals, vals]), (n, n)).coalesce()
    r, c = P.indices()
    keep = (r >= r0) & (r < r0 + nl)
    r = r[keep] - r0
    rowptr = torch.zeros(nl + 1, dtype=torch.long, device=dev)
    rowptr[1:] = torch.bincount(r, minlength=nl).cumsum(0)
    return rowptr, c[keep].contiguous(), (P.values()[keep] / (2 * n)).contiguous()

def _tsne_torch(X, init, perplexity, chunk_elems, n_iter=1000, exaggeration=12.0, explore_iter=250, device=None):
    """
    Barnes-Hut-style t-SNE on the GPU via torch, returning the 2D layout. Minimizes KL(P||Q) under
    the Student-t low-dim kernel by momentum gradient descent with adaptive gains, following
    sklearn's schedule (PCA init, early exaggeration for the first `explore_iter` steps, momentum
    0.5->0.8, learning_rate='auto') and sklearn's default sparse-affinity approximation: the
    high-dim conditionals are restricted to each point's k = 3*perplexity nearest neighbors (the
    dropped tail is numerically ~0), so results are structurally equivalent to default sklearn
    TSNE -- not bit-identical. The pairwise repulsion term stays exact, recomputed each iteration
    from the current layout in row chunks. `init` is the 2D starting layout (the reused PCA init).

    Memory: O(N*k) for the sparse joint P plus one transient (chunk, N) work buffer sized by
    `chunk_elems` (chunk = chunk_elems // N rows; default ~1 GiB) -- no N x N matrix is ever
    materialized, so N is bounded by compute time (O(N^2) per iteration, bandwidth-bound), not by GPU
    memory.

    Under DDP with world_size > 1 the points are row-sharded: each rank owns its rows' P edges,
    gradient, and optimizer state, all-gathers the (N x 2) layout every iteration and all-reduces
    the scalar Student-t normalizer Z. Must then be entered collectively by every rank with
    identical X and init (they are: X is all-gathered during eval and init is the deterministic
    shared PCA); world_size 1 degenerates to the same code with no collective ops.
    """
    eps = 1e-12
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ddp = dist.is_initialized() and dist.get_world_size() > 1
    rank, world = (dist.get_rank(), dist.get_world_size()) if ddp else (0, 1)
    X = torch.as_tensor(X, dtype=torch.float32, device=dev)
    n = X.shape[0]
    counts = _shard_counts(n, world)
    r0, nl = sum(counts[:rank]), counts[rank]
    k = min(int(3 * perplexity), n - 1)  # sklearn Barnes-Hut neighbor count
    chunk = max(1, chunk_elems // n)  # rows per transient (chunk, N) buffer (chunk_elems = 2^hw.eval.tsne_chunk_log2)
    lr = max(n / exaggeration / 4.0, 50.0)  # sklearn learning_rate='auto'

    d2, idx = _knn(X[r0:r0 + nl], X, k, r0, chunk)
    rowptr, pcols, pvals = _sparse_joint_p(d2, idx, perplexity, n, r0, counts)
    del d2, idx
    prows = torch.repeat_interleave(torch.arange(nl, device=dev), rowptr.diff())  # local row per edge
    pvals.mul_(exaggeration)  # early exaggeration folded into the P values (undone at explore_iter)

    Y = torch.as_tensor(init, dtype=torch.float32, device=dev)  # N x 2 shared PCA init, replicated
    update = torch.zeros((nl, 2), device=dev)  # per-row optimizer state, sharded with the rows
    gains = torch.ones((nl, 2), device=dev)
    Wc = torch.empty((min(chunk, max(nl, 1)), n), device=dev)  # reused (chunk, N) repulsion buffer
    ones = torch.ones((n, 1), device=dev)
    rep = torch.empty((nl, 2), device=dev)
    for it in range(n_iter):
        if it == explore_iter:
            pvals.div_(exaggeration)  # end early exaggeration
        momentum = 0.5 if it < explore_iter else 0.8
        Yr = Y[r0:r0 + nl]
        # attraction over the sparse P edges: F_i = sum_j p_ij w_ij (y_i - y_j), w = 1/(1+|yi-yj|^2).
        # Row-aggregated with one CSR matmul against [Y | 1] (deterministic -- no scatter atomics):
        # cols [:2] give sum_j pw_ij y_j, col [2:] gives sum_j pw_ij.
        pw = pvals / (1.0 + (Yr[prows] - Y[pcols]).pow(2).sum(1))
        agg = torch.sparse_csr_tensor(rowptr, pcols, pw, size=(nl, n)) @ torch.cat([Y, ones], dim=1)
        attr = agg[:, 2:] * Yr - agg[:, :2]
        # repulsion, exact and chunked: F_i = [(sum_j w_ij^2) y_i - W.^2 @ Y] / Z with Z = sum_ij w_ij;
        # W rows are built by the gram trick into the reused buffer, self zeroed, squared in place
        # after accumulating Z. Never materializes more than (chunk, N).
        ry = (Y * Y).sum(1)
        Z = torch.zeros((), device=dev)
        for s in range(0, nl, chunk):
            W = Wc[: min(chunk, nl - s)]
            torch.matmul(Yr[s:s + W.shape[0]], Y.t(), out=W)
            W.mul_(-2.0).add_(ry[r0 + s:r0 + s + W.shape[0], None]).add_(ry[None, :]).clamp_min_(0.0).add_(1.0).reciprocal_()
            _zero_self(W, r0 + s)
            Z += W.sum()
            W.pow_(2)
            rep[s:s + W.shape[0]] = W.sum(1, keepdim=True) * Yr[s:s + W.shape[0]] - W @ Y
        if ddp:
            dist.all_reduce(Z, op=dist.ReduceOp.SUM)  # global normalizer over all N x N pairs
        grad = 4.0 * (attr - rep / Z.clamp_min(eps))  # nl x 2 local-row gradient
        inc = (update * grad) < 0  # adaptive gains: grow when sign flips, shrink otherwise
        gains = torch.where(inc, gains + 0.2, gains * 0.8).clamp_min(0.01)
        update = momentum * update - lr * gains * grad
        Yl = Yr + update
        Y = _allgather_rows(Yl, counts) if ddp else Yl  # stitch the updated rows back into the full Y
        Y = Y - Y.mean(0, keepdim=True)  # keep centered (every rank identical after the all-gather)
    return Y.detach().cpu().numpy()

def compute_tsne(embeddings, perplexity, init, chunk_elems, n_iter=1000):
    """
    Reduce embedding dimensionality to 2D via GPU t-SNE (`_tsne_torch`). Returns the 2D layout.
    `embeddings` may be a numpy array or a torch tensor (a GPU tensor is used in place, no host
    copy). `init` is the 2D starting layout (the reused PCA init). `chunk_elems`
    (2^hw.eval.tsne_chunk_log2) tiles the transient GPU buffers -- memory <-> speed only.
    """
    _log(f"Running GPU t-SNE on {embeddings.shape[0]} samples (dim={embeddings.shape[1]}) at perplexity {perplexity}...")
    return _tsne_torch(embeddings, np.asarray(init), perplexity=perplexity, chunk_elems=chunk_elems, n_iter=n_iter)

def _pca_2d(embeddings):
    """Reduce embedding dimensionality to 2D via PCA, silently. Uses the same solver/seed t-SNE uses for
    its internal `init="pca"`, so this projection can be fed straight back as the t-SNE init (see
    `_pca_init`) instead of computing the same PCA twice."""
    return PCA(n_components=2, svd_solver="randomized", random_state=42).fit_transform(embeddings)

def compute_pca(embeddings):
    """`_pca_2d`, announced -- for the in-loop compute, whose progress prints are wanted. The post-trial
    UMAP stage calls `_pca_2d` instead: it runs in the detached render worker, which shares a terminal
    with the next trial's progress bar, so it stays silent."""
    _log(f"Running PCA on {embeddings.shape[0]} samples (dim={embeddings.shape[1]})...")
    return _pca_2d(embeddings)

def _pca_init(pca_2d):
    """Scale a 2D PCA projection to t-SNE's PCA-init convention (PC1 std -> 1e-4) so it can be reused
    as the t-SNE `init` rather than having TSNE recompute the same PCA internally."""
    return (pca_2d / np.std(pca_2d[:, 0]) * 1e-4).astype(np.float32)

def compute_umap(embeddings, cfg_umap, init, spherical=False, knn=None):
    """
    Reduce embedding dimensionality to 2D via UMAP. Runs on CPU -- it is fit in the post-trial render
    worker off the collective GPU path (see `compute_umap_projections`), never in the training loop.

    `spherical` swaps the output space for a 2-sphere and returns (N, 3) UNIT VECTORS instead of an
    (N, 2) plane; `knn` is a precomputed neighbor graph from `_umap_knn`, shared between the two variants.

    `init` is the starting layout (None -> UMAP's spectral init). Sweeping an eval sequence, each eval is initialized from the
    PREVIOUS eval's layout so consecutive fits settle in the same basin instead of each landing in an
    arbitrary one -- continuity at the source, which cross-eval orientation alone cannot supply (it can
    only remove residual rigid motion, not a genuinely reorganized layout).

    The layout is mean-centered on return. umap-learn leaves its output wherever the optimization lands,
    but everything downstream assumes an origin-centered layout the way `_tsne_torch` (which recenters
    every iteration) produces: `orient_proj` applies a rotation ABOUT THE ORIGIN with no translation
    term, so an off-center cloud would be aligned by swinging it through an arc rather than by matching
    its shape, and `_square_limits` frames the _RIGID methods symmetrically about the origin, so an
    off-center cloud would sit in a corner of a mostly-empty panel.
    """
    import umap  # heavy (numba JIT) import, paid only in the render worker
    kwargs = {} if cfg_umap["n_iter"] is None else {"n_epochs": cfg_umap["n_iter"]}
    if spherical:
        kwargs["output_metric"] = "haversine"  # optimize great-circle distance on S^2, not in a plane
    reducer = umap.UMAP(
        n_components=2,
        metric=_UMAP_METRIC,
        n_neighbors=_umap_k(embeddings.shape[0], cfg_umap),
        min_dist=cfg_umap["min_dist"],
        init="spectral" if init is None else np.ascontiguousarray(init, dtype=np.float32),
        precomputed_knn=(None, None, None) if knn is None else knn,
        verbose=False,
        **kwargs,
    )
    out = np.asarray(reducer.fit_transform(embeddings), dtype=np.float32)
    if not spherical:
        return out - out.mean(axis=0, keepdims=True)
    # haversine output is a pair of ANGLES, and unbounded -- nothing wraps them, so they routinely land
    # many turns outside [0, pi] x [0, 2pi). Mapping through sin/cos wraps them exactly, and the unit
    # vector is what everything downstream (orientation, culling, plotting) actually wants.
    theta, phi = out.T
    return np.stack([np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(theta)], axis=1).astype(np.float32)

def _umap_k(n, cfg_umap):
    """n_neighbors, clamped below the sample count (UMAP requires n_neighbors < n_samples)."""
    return min(cfg_umap["n_neighbors"], n - 1)

def _umap_knn(embeddings, cfg_umap):
    """The cosine kNN graph, computed ONCE and handed to both UMAP variants via `precomputed_knn`.

    The two fits differ only in their OUTPUT space -- same metric, same n_neighbors -- so they describe
    the same input graph, and building it twice is duplicated work. Note the saving is small in wall
    clock: measured on 6k x 512 the search is ~0.1s against ~0.5s for the flat layout and ~16s for the
    spherical one, so the haversine layout optimization, not the neighbor search, is what a spherical
    fit actually costs. Sharing is kept for the guarantee rather than the seconds: both variants are
    provably fit from an identical graph, so any difference between their plots is output-space only."""
    from umap.umap_ import nearest_neighbors

    return nearest_neighbors(embeddings, n_neighbors=_umap_k(embeddings.shape[0], cfg_umap),
                             metric=_UMAP_METRIC, metric_kwds={}, angular=True, random_state=None,
                             low_memory=True, verbose=False)

def _xyz_to_angles(xyz):
    """Unit vectors -> the (polar, azimuth) pair UMAP's haversine output space works in, so a previous
    eval's sphere layout can seed the next eval's fit."""
    theta = np.arccos(np.clip(xyz[:, 2], -1.0, 1.0))
    phi = np.arctan2(xyz[:, 1], xyz[:, 0]) % (2 * np.pi)
    return np.stack([theta, phi], axis=1).astype(np.float32)

def orient_proj(proj, labels, ref=None, tau=1.0):
    """
    Pin a canonical orientation for an arbitrary-frame projection (t-SNE / UMAP, whose axes and handedness
    carry no meaning) so the layout neither spins nor mirror-flips across an ordered sequence of evals. The
    transform is rigid about the origin (rotation + optional reflection), so all structure is preserved.
    PCA is oriented by `orient_pca` instead -- its axes ARE meaningful and must not be rotated.

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

def orient_pca(proj, labels, ref=None, tau=1.0):
    """
    Pin PCA's per-component SIGN across an ordered sequence of evals. Unlike t-SNE/UMAP the layout must NOT
    be rotated -- PCA's axes are the principal directions, and rotating them would destroy what the plot is
    showing. The only cross-eval ambiguity is each component's sign: sklearn resolves it from the data
    (`svd_flip` keys off the max-magnitude loading), which is data-dependent and so flips between evals and
    mirrors the plot. Picks the diagonal +/-1 transform whose count-weighted per-class CoMs best match the
    running reference; the objective separates per axis, so each sign is the closed-form sign of that axis's
    weighted CoM inner product (no search). The first eval (`ref=None`) defines the frame as-is.

    Same (ref, tau) contract as `orient_proj`: `ref` is the running {class: CoM} reference (None on the
    first eval), `tau` in (0, 1] EMA-mixes this eval's oriented CoMs into it. Returns (oriented_proj, ref).
    """
    labels = np.asarray(labels)
    vals, counts = np.unique(labels, return_counts=True)
    coms = {v: proj[labels == v].mean(axis=0) for v in vals}
    if ref is not None:
        shared = [v for v in vals if v in ref]
        count_of = dict(zip(vals, counts))
        w = np.array([count_of[v] for v in shared], dtype=float)
        C = np.array([coms[v] for v in shared])
        R = np.array([ref[v] for v in shared])
        sign = np.sign((w[:, None] * C * R).sum(axis=0))  # per axis, independently
        sign[sign == 0.0] = 1.0  # an axis with zero weighted overlap carries no evidence -> leave it
        proj = proj * sign

    coms_oriented = {v: proj[labels == v].mean(axis=0) for v in vals}
    if ref is None:
        ref = coms_oriented
    else:
        ref = {v: (1.0 - tau) * ref.get(v, coms_oriented[v]) + tau * coms_oriented[v] for v in vals}
    return proj, ref

def _rot_to_pole(v):
    """Row-vector rotation (`xyz @ R`) taking `v`'s direction onto +z, the north pole. Rodrigues about
    the axis perpendicular to both; identity when `v` already points at a pole."""
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.eye(3)
    v = v / n
    axis = np.cross(v, [0.0, 0.0, 1.0])
    sin_a = np.linalg.norm(axis)
    if sin_a < 1e-12:  # already on the z axis: identity if north, a half-turn if south
        return np.eye(3) if v[2] > 0 else np.diag([1.0, -1.0, -1.0])
    axis = axis / sin_a
    K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    R = np.eye(3) + sin_a * K + (1.0 - v[2]) * (K @ K)  # cos_a == v[2]
    return R.T  # column-vector Rodrigues -> row-vector convention

def _spin_z(Q2):
    """Embed a 2x2 transform of the xy plane into 3x3, leaving z (and so the poles) untouched."""
    Q = np.eye(3)
    Q[:2, :2] = Q2
    return Q

def orient_sphere(xyz, labels, ref=None, tau=1.0):
    """
    Pin a canonical orientation for a layout on the 2-sphere -- the planar scheme one dimension up.

    Every aligned eval gets ONE count-weighted orthogonal Procrustes solved in SO(3) (reflection
    allowed, as in the planar case) against the running reference of per-class CoMs: the single rigid
    motion of the whole ball that moves the cluster constellation -- seen and unseen clusters alike --
    the least. Nothing is pinned after the first eval, so nothing can force a large rotation: the ball
    and its gridlines stay fixed from the viewer's standpoint while the clusters glide across them
    between evals, the way the planar t-SNE evolutions behave. (The CoMs sit inside the ball, which is
    fine -- only their directions drive the fit.) Minimal rotation also keeps the same side of the ball
    facing the camera from eval to eval, which matters because only ONE hemisphere is ever drawn
    (see `_SPHERE_VIEW`).

    The first eval (`ref=None`) bootstraps a canonical frame the way `orient_proj` does: the
    highest-cardinality class starts at the north pole, the second-largest on the central meridian
    (zero azimuth), and the third-largest fixes the handedness. Those are starting landmarks, not
    constraints -- later evals drift off them only as much as the layout genuinely moves.

    Same (ref, tau) contract as `orient_proj`. Returns (oriented unit vectors, ref).
    """
    labels = np.asarray(labels)
    vals, counts = np.unique(labels, return_counts=True)  # vals sorted -> deterministic tie-break
    coms = lambda pts: {v: pts[labels == v].mean(axis=0) for v in vals}

    if ref is None:  # bootstrap the canonical frame (see docstring)
        rank = sorted(range(len(vals)), key=lambda i: (-counts[i], vals[i]))  # classes, most populous first
        xyz = xyz @ _rot_to_pole(coms(xyz)[vals[rank[0]]])  # top class -> north pole
        com = coms(xyz)
        Q2 = np.eye(2)
        if len(rank) >= 2:  # second class defines the zero azimuth (the central meridian)
            a = com[vals[rank[1]]][:2]
            if np.linalg.norm(a) > 1e-12:
                c, sn = a / np.linalg.norm(a)
                Q2 = np.array([[c, -sn], [sn, c]])
        if len(rank) >= 3 and (com[vals[rank[2]]][:2] @ Q2)[1] < 0:  # third fixes the handedness
            Q2 = Q2 @ np.diag([1.0, -1.0])
        xyz = xyz @ _spin_z(Q2)
    else:  # the rigid motion of the WHOLE ball that best matches the reference constellation
        com = coms(xyz)
        shared = [v for v in vals if v in ref]
        count_of = dict(zip(vals, counts))
        w = np.array([count_of[v] for v in shared], dtype=float)
        C = np.array([com[v] for v in shared])
        R = np.array([ref[v] for v in shared])
        U, _, Vt = np.linalg.svd(C.T @ (w[:, None] * R))
        xyz = xyz @ (U @ Vt)
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)  # rigid in exact arithmetic; renormalize for drift

    coms_oriented = coms(xyz)
    if ref is None:
        ref = coms_oriented
    else:
        ref = {v: (1.0 - tau) * ref.get(v, coms_oriented[v]) + tau * coms_oriented[v] for v in vals}
    return xyz, ref

def _orient(method, proj, labels, ref, tau):
    """Cross-eval orientation for one method's projection: rigid 2D for the arbitrary-frame planar
    methods (_RIGID), rigid 3D for the spherical one (_SPHERICAL), sign-only for PCA. Returns
    (oriented_proj, updated_ref)."""
    if method in _SPHERICAL:
        return orient_sphere(proj, labels, ref, tau)
    return (orient_proj if method in _RIGID else orient_pca)(proj, labels, ref, tau)

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
    -> (xlim, ylim) sharing one +/- bound. The _RIGID methods' plots are equal-aspect and origin-centered, so their
    axes auto-scale to a single square bound that fits the data (instead of a fixed config bound that
    doesn't transfer across datasets). Over a list of per-eval projections this unions them, freezing
    one bound across the whole evolving GIF."""
    allp = np.concatenate(projs, axis=0)
    bound = np.abs(allp).max() * (1 + margin)
    return (-bound, bound), (-bound, bound)

_SPHERE_LIMITS = ((-1.0, 1.0), (-1.0, 1.0))  # the ball is unit-radius; nothing to fit (see _setup_sphere_ax)

def _limits_for(method, projs):
    """Axis limits for a panel: fixed for the spherical method (the ball never changes size), square
    origin-centered bounds for the arbitrary-frame planar methods (_RIGID, plotted equal-aspect about
    the origin), the data bounding box for PCA."""
    if method in _SPHERICAL:
        return _SPHERE_LIMITS
    return _square_limits(projs) if method in _RIGID else _common_limits(projs)

def _cam_dir(elev, azim):
    """Unit vector from the origin toward the camera, for matplotlib's (elev, azim) in degrees."""
    e, a = np.radians(elev), np.radians(azim)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])

_SPHERE_CAM = _cam_dir(*_SPHERE_VIEW)

def _sphere_basis():
    """Orthonormal (right, up) spanning the image plane of a camera at `_SPHERE_VIEW` looking at the
    origin -- the basis `_sphere_to_screen` projects onto."""
    right = np.cross([0.0, 0.0, 1.0], _SPHERE_CAM)
    right /= np.linalg.norm(right)
    return right, np.cross(_SPHERE_CAM, right)

def _sphere_to_screen(xyz):
    """Orthographic projection of unit vectors onto the fixed viewpoint's image plane. Points on the unit
    sphere land inside the unit disc, so a sphere panel is just a 2D scatter over a unit circle."""
    right, up = _SPHERE_BASIS
    return np.stack([xyz @ right, xyz @ up], axis=1)

_SPHERE_BASIS = _sphere_basis()

def _sphere_gridlines(ax):
    """Draw the visible hemisphere's parallels and meridians onto the ball. They give the layout a frame
    of reference that stays fixed for the viewer while the clusters glide across it (`orient_sphere`
    aligns evals by minimal rotation; its bootstrap starts the largest class at the pole). Each arc is sampled in 3D, culled to the camera-facing side and
    projected; culled samples become NaN, which breaks the polyline rather than chording it straight
    across the back of the ball."""
    step = np.radians(_SPHERE_GRID_STEP)
    t = np.linspace(0.0, 2 * np.pi, 361)
    u = np.linspace(0.0, np.pi, 181)
    arcs = [np.stack([np.sin(th) * np.cos(t), np.sin(th) * np.sin(t), np.full_like(t, np.cos(th))], axis=1)
            for th in np.arange(step, np.pi - 1e-9, step)]                    # parallels
    arcs += [np.stack([np.sin(u) * np.cos(ph), np.sin(u) * np.sin(ph), np.cos(u)], axis=1)
             for ph in np.arange(0.0, 2 * np.pi - 1e-9, step)]                # meridians
    for pts in arcs:
        xy = _sphere_to_screen(pts)
        xy[pts @ _SPHERE_CAM <= 0] = np.nan  # hide the far side; NaN breaks the line instead of chording it
        ax.plot(xy[:, 0], xy[:, 1], color="0.55", linewidth=0.6, linestyle="--", alpha=0.5, zorder=1)

def _setup_sphere_ax(ax, bg_color):
    """Turn a plain 2D axes into one opaque ball seen from `_SPHERE_VIEW`: a filled unit circle in the
    manifold-viz panel color on a WHITE background, with the camera-facing hemisphere's points scattered
    over it (`_composite_frame` culls the far side and projects the near side with `_sphere_to_screen`)
    and spherical gridlines for reference.

    Deliberately NOT a 3D axes. The viewpoint is fixed and the far hemisphere is never drawn, so nothing
    here needs real 3D -- and mpl3d actively gets in the way: `Path3DCollection` re-sorts its points by
    depth on every draw (`argsort(vzs)`), which silently discards the shuffled draw order every other
    panel gets. Projecting by hand keeps that order, and drops the occlusion, z-fighting and zorder
    workarounds a 3D ball needed."""
    ax.add_patch(Circle((0.0, 0.0), 1.0, facecolor=bg_color or "white", edgecolor="none", zorder=0))
    _sphere_gridlines(ax)  # zorder 1: over the ball, under the points
    ax.set_xlim(-_SPHERE_MARGIN, _SPHERE_MARGIN)
    ax.set_ylim(-_SPHERE_MARGIN, _SPHERE_MARGIN)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.set_facecolor("white")  # behind the ball, deliberately NOT the panel color
    return ax.scatter([], [], zorder=2)

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

def _save_gif_stream(frame_iter, fpath_gif, frame_ms):
    """Write a forever-looping GIF from an arbitrarily long frame stream: peak memory is one frame,
    regardless of the total count (one per eval for the evolving GIFs), so it doesn't scale with the
    number of checkpoints. One shared fixed palette is built from the FIRST frame -- representative
    because an evolving GIF's frames share one color SET (same eval points, same color maps; only
    positions move) -- then each frame is quantized against it and written one at a time via the legacy
    getheader/getdata frame blocks. Image.save(append_images=...) can't stream: it buffers every diffed
    frame before writing, so its peak memory scales with the frame count. The shared palette also stops
    otherwise-static regions shimmering (see _fixed_palette)."""
    it = iter(frame_iter)
    sample = list(islice(it, 1))
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
    ctx.set_forkserver_preload(["utils.manif_viz"])  # import the stack once in the server, not per worker
    # RENDER_MAX_WORKERS caps the fan-out (the campaign sets it so a background render shares cores with the
    # next trial's training instead of oversubscribing them); unset/0 -> every core (manual offline re-render)
    cap = int(os.environ.get("RENDER_MAX_WORKERS", "0")) or len(os.sched_getaffinity(0))
    workers = min(len(jobs), cap)
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_render_init) as pool:
        for fut in [pool.submit(func, *args) for func, args in jobs]:
            fut.result()  # surface any worker exception in the parent

# Each panel: which projection, what labels color it, the filename stem, and which partition (if any)
# it masks. Panels are never plotted alone -- they're tiled into the stacked leaf/penult pairs and the
# joint composite (see _GRIDS).
# (proj_key, label_role, color_role, stem, alpha_role)
_PANELS = [
    ("id",    "penult", "penult", "id_penult",        "full"),
    ("id",    "cid",    "leaf",   "id_leaf",          "full"),
    ("ood",   "penult", "penult", "ood_penult",       "full"),
    ("ood",   "cid",    "leaf",   "ood_leaf",         "full"),
    ("joint", "penult", "penult", "joint_penult",     "full"),
    ("joint", "cid",    "leaf",   "joint_leaf",       "full"),
    ("joint", "penult", "penult", "joint_id_penult",  "id"),
    ("joint", "cid",    "leaf",   "joint_id_leaf",    "id"),
    ("joint", "penult", "penult", "joint_ood_penult", "ood"),
    ("joint", "cid",    "leaf",   "joint_ood_leaf",   "ood"),
    ("joint", "nshot",  "nshot",  "joint_nshot",      "full"),
]
_STEM_COLOR_ROLE = {stem: color_role for _, _, color_role, stem, _ in _PANELS}  # stem -> leaf/penult/nshot
_STEM_PROJKEY = {stem: proj_key for proj_key, _, _, stem, _ in _PANELS}  # stem -> id/ood/joint projection

def _resolve_panels(projs, cids_id, cids_ood, penults_id, penults_ood, color_leaf, color_penult, nshot_id, color_nshot):
    """Yield (proj, labels, color_map, alpha, stem) for each panel in `_PANELS`."""
    cids_full = list(cids_id) + list(cids_ood)
    penults_full = list(penults_id) + list(penults_ood)
    is_id = np.arange(len(cids_full)) < len(cids_id)  # ID rows first, then OOD
    labels_by = {
        ("id", "cid"): list(cids_id),       ("id", "penult"): list(penults_id),
        ("ood", "cid"): list(cids_ood),     ("ood", "penult"): list(penults_ood),
        ("joint", "cid"): cids_full,      ("joint", "penult"): penults_full,
        # n-shot panel: ID points by bucket, OOD points all black (the sentinel)
        ("joint", "nshot"): list(nshot_id) + [_OOD_LABEL] * len(cids_ood),
    }
    colors_by = {"leaf": color_leaf, "penult": color_penult, "nshot": color_nshot}
    alphas_by = {"full": 1.0, "id": is_id.astype(float), "ood": (~is_id).astype(float)}
    for proj_key, label_role, color_role, stem, alpha_role in _PANELS:
        yield (projs[proj_key], labels_by[(proj_key, label_role)], colors_by[color_role],
               alphas_by[alpha_role], stem)

# Every standalone output is a flush grid of panels: the leaf(left)/penult(right) subject pairs and
# the 2x4 joint composite (cols = OOD / ID / full / n-shot; rows = leaf / penult, n-shot's penult
# cell blank). Each entry is (out_name, suptitle-subject, grid-of-stems); None = blank cell.
_GRID = [
    ["joint_ood_leaf",   "joint_id_leaf",   "joint_leaf",   "joint_nshot"],
    ["joint_ood_penult", "joint_id_penult", "joint_penult", None],
]
_GRIDS = [
    ("id",          "ID",               [["id_leaf",        "id_penult"]]),
    ("ood",         "OOD",              [["ood_leaf",       "ood_penult"]]),
    ("joint",       "Joint (ID + OOD)", [["joint_leaf",     "joint_penult"]]),
    ("joint_id",    "Joint (ID)",       [["joint_id_leaf",  "joint_id_penult"]]),
    ("joint_ood",   "Joint (OOD)",      [["joint_ood_leaf", "joint_ood_penult"]]),
    ("joint_panel", "Joint",            _GRID),
]
_COMPOSITE_COL_TITLES = ["OOD", "ID", "ID + OOD", "n-shot"]  # column headers for the 2x4 joint composite

def _grid_group(out_name):
    """(panel-group subdir, output file stem) for a grid: the 2x4 joint composite lands under 7panel/
    renamed to 'joint', every leaf/penult stacked pair under 2panel/ keeping its grid name. The group
    dir nests above the method dir -> viz/<group>/{pca,tsne,umap,umap_sphere}/<stem>.png."""
    return ("7panel", "joint") if out_name == "joint_panel" else ("2panel", out_name)

# (out_name, suptitle-subject, leaf_stem, penult_stem) for the cross-method grids: one per 2panel
# subject (the single-row leaf/penult grids), excluding the 7panel composite. A cross-method stack tiles
# EVERY projection method into one figure (one column per method, `_METHODS` order left -> right) for a
# subject, so -- unlike 2panel/7panel -- it has no per-method subdir: viz/8panel/<out_name>.png. The
# group is named for its cell count (len(_METHODS) x 2 colorings), matching 2panel/7panel; adding a
# projection method changes that count, so the name has to move with it.
_8PANEL_SUBJECTS = [(out_name, subject, grid[0][0], grid[0][1])
                    for out_name, subject, grid in _GRIDS if out_name != "joint_panel"]
_8PANEL_LABEL = " + ".join(_METHODS)  # suptitle method label for the cross-method grids

_CELL = 6.58  # square plotting-cell size (in)

def _grid_layout(nrows, ncols, header=False):
    """(figsize, subplots_adjust kwargs) giving exactly-square cells (so the equal-aspect t-SNE/UMAP panels
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
    spherical = method in _SPHERICAL  # a composite grid is one method throughout, so this is grid-wide
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
            if spherical:  # one opaque ball per cell; no axes, ticks or limits to manage
                sc_by[stem] = _setup_sphere_ax(ax, bg_color)
                _apply_legend(ax, legend_by_role[_STEM_COLOR_ROLE[stem]], fontsize=13)
                continue
            xlim, ylim = limits[stem]  # square origin-centered for t-SNE/UMAP, data bbox for PCA (_limits_for)
            sc = ax.scatter([], [])  # marker size is set per frame (_composite_frame)
            if bg_color is not None:  # None -> matplotlib default (white) panel background
                ax.set_facecolor(bg_color)
            if method in _RIGID:
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

def _composite_frame(sc_by, data, marker_size):
    """Push the points onto every panel in a *shuffled* draw order (fixed seed, so it is reproducible)
    rather than class-by-class, so no single class ends up plotted entirely on top of the others.

    Fully-transparent points (the partition a masked panel hides) are dropped from the draw rather than
    painted invisibly -- the survivors keep their relative draw order (the shuffle restricted to them) and
    axes are frozen independently of the plotted subset (see _evolution_limits/_limits_for), so geometry and
    ordering are unchanged; only the wasted per-point rasterization goes away.

    A (N, 3) projection is a layout on the sphere: those panels additionally drop the hemisphere facing
    away from the camera -- which is what makes the ball opaque -- and project the survivors onto the
    fixed viewpoint's image plane. They are ordinary 2D scatters from there on, shuffle included."""
    for stem, sc in sc_by.items():
        proj, rgba = data[stem]
        order = np.random.default_rng(_DRAW_ORDER_SEED).permutation(len(rgba))
        vis = rgba[order, 3] > 0  # masked-out partitions are dropped, not painted invisibly
        spherical = proj.shape[1] == 3
        if spherical:
            vis &= proj[order] @ _SPHERE_CAM > 0  # near hemisphere only
        order = order[vis]  # survivors keep the shuffle's relative order
        sc.set_offsets(_sphere_to_screen(proj[order]) if spherical else proj[order])
        sc.set_facecolors(rgba[order])
        sc.set_sizes(np.full(len(order), marker_size))

def composite_plot(grid, comp, fpath_png, suptitle, style, limits=None):
    """Static flush-grid PNG (single frame at the given marker size). `comp` maps stem -> (proj, rgba).
    `limits` (stem -> (xlim, ylim)) overrides the per-panel auto-fit -- the pooled pca_bounds='final'
    frame; None auto-fits each panel to its own points."""
    stems = _stems_of(grid)
    if limits is None:
        limits = {s: _limits_for(style.method, [comp[s][0]]) for s in stems}
    data = {s: (comp[s][0], comp[s][1]) for s in stems}
    fig, sc_by = _composite_canvas(grid, limits, suptitle, _GIF_DPI, style)
    _composite_frame(sc_by, data, style.marker_size)
    fig.savefig(fpath_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

def _cids_by(cids_id, cids_ood):
    """{proj key: class-id list} for the three projection subjects (joint = ID followed by OOD)."""
    return {"id": cids_id, "ood": cids_ood, "joint": cids_id + cids_ood}

def _resolved_evals(evals, names, stems, methods, cmaps, ema_tau, orient=True, fname="projections.npz"):
    """Yield (name, {(method, stem): (proj, point_colors, alpha)}) one eval at a time, loading each cache
    ONCE and resolving it for every method in `methods` -- so a cross-method caller never re-reads a cache
    per method, and no caller holds more than one eval's data. When `orient`, every method's cached
    projection is aligned against the orientation reference carried across evals in order (the same sweep
    render_evolution/render_eval reproduce) -- rigid for t-SNE/UMAP, sign-only for PCA (`_orient`).
    `orient=False` (pooled mode) skips orientation entirely -- the pooled projections already share one
    frame across thresholds. `fname` selects the per-eval or pooled cache. Only the panels in `stems` are
    resolved (rest of grid blank)."""
    color_leaf, color_penult, color_nshot, cid_2_penult, cid_2_nshot = cmaps
    ref = {}  # running per-class CoM orientation reference per (method, proj key), carried across evals
    for d, name in zip(evals, names):
        projs_by_method, cids_id, cids_ood = _load_projections(d, fname)
        penults_id = [cid_2_penult[c] for c in cids_id]
        penults_ood = [cid_2_penult[c] for c in cids_ood]
        nshot_id = [cid_2_nshot[c] for c in cids_id]
        resolved = {}
        for method in methods:
            projs = dict(projs_by_method[method])
            if orient:  # align this eval to the running reference (rigid for t-SNE/UMAP, sign-only for PCA)
                cids_by = _cids_by(cids_id, cids_ood)
                for k in projs:
                    projs[k], ref[(method, k)] = _orient(method, projs[k], cids_by[k], ref.get((method, k)), ema_tau)
            for proj, labels, color_map, alpha, stem in _resolve_panels(
                    projs, cids_id, cids_ood, penults_id, penults_ood, color_leaf, color_penult, nshot_id, color_nshot):
                if stem in stems:
                    resolved[(method, stem)] = (proj, np.array([color_map[label] for label in labels]), alpha)
        yield name, resolved

def composite_evolving_gif(grid, subject, viz_context, evals, names, cmaps, ema_tau, limits, fpath_gif, style,
                            orient=True, fname="projections.npz"):
    """Training-evolving GIF of a flush grid: one frame per eval, axes
    frozen to the cross-eval union (`limits`, precomputed by render_evolution). Loads + renders one eval's
    cache at a time and streams the frames to disk, so peak memory is independent of the number of
    evals/checkpoints. The suptitle carries the manifold subject (from `_GRIDS`) and the eval name.
    `orient`/`fname` select per-eval (oriented) vs pooled (shared-frame, `orient=False`) sources."""
    stems = _stems_of(grid)
    supt = lambda name: _manifold_title(style.method, viz_context, subject, suffix=f", {name}")
    fig, sc_by = _composite_canvas(grid, limits, supt(names[0]), _GIF_DPI, style)
    def _frames():  # generator: render frames lazily (one eval loaded at a time) so they stream to disk
        for name, resolved in _resolved_evals(evals, names, stems, (style.method,), cmaps, ema_tau, orient, fname):
            fig.suptitle(supt(name), fontsize=22, fontweight="bold", x=fig.subplotpars.left, ha="left")  # align to the leftmost plot's left edge
            data = {s: (resolved[(style.method, s)][0], _rgba(resolved[(style.method, s)][1], resolved[(style.method, s)][2])) for s in stems}
            _composite_frame(sc_by, data, style.marker_size)
            yield _canvas_frame(fig)
    _save_gif_stream(_frames(), fpath_gif, style.frame_ms)
    plt.close(fig)

def _8panel_canvas(leaf_stem, penult_stem, limits, suptitle, dpi, style):
    """Build the cross-method figure: one COLUMN per method (`_METHODS` order, left -> right) and two
    rows -- leaf coloring on top, penult coloring underneath -- of the same subject. Cells are flush
    (wspace=hspace=0). The method names ride as column headers rather than as axis labels, since a
    column is a method and the two rows share it; ticks appear only on the grid's outer boundary (x on
    the bottom row, y on the left and right columns) and inner shared edges stay bare.

    The spherical method's cells are opaque balls with no axes at all -- still ordinary 2D axes, just
    set up differently (`_setup_sphere_ax`). `limits` maps (method, stem) -> (xlim, ylim) and is ignored
    for the spherical column. Returns (fig, {(method,
    stem): scatter}); callers push per-frame data via _composite_frame, keyed by the same tuples."""
    rows = [leaf_stem, penult_stem]
    ncols = len(_METHODS)
    # inch insets: l/r reserve the left/right y axes, b the bottom x axis, t the column headers + suptitle
    l, r, b, t = 0.75, 0.75, 0.6, 1.55
    fig_w, fig_h = l + ncols * _CELL + r, b + 2 * _CELL + t
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.suptitle(suptitle, fontsize=22, fontweight="bold", x=l / fig_w, ha="left")  # align to the leftmost plot's left edge
    sc_by = {}
    for ri, stem in enumerate(rows):
        for ci, method in enumerate(_METHODS):
            spherical = method in _SPHERICAL
            ax = fig.add_subplot(2, ncols, ri * ncols + ci + 1)
            if ri == 0:
                ax.set_title(method, fontsize=16, fontweight="bold")
            if spherical:
                sc_by[(method, stem)] = _setup_sphere_ax(ax, style.bg_color)
                _apply_legend(ax, style.legend_by_role[_STEM_COLOR_ROLE[stem]], fontsize=13)
                continue
            if style.bg_color is not None:
                ax.set_facecolor(style.bg_color)
            if method in _RIGID:
                ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(limits[(method, stem)][0])
            ax.set_ylim(limits[(method, stem)][1])
            # flush perimeter axes: ticks only on this cell's outer-boundary edges
            ax.tick_params(left=False, right=False, bottom=False, top=False, labelsize=9,
                           labelleft=False, labelright=False, labelbottom=False, labeltop=False)
            if ci == 0:
                ax.tick_params(axis="y", left=True, labelleft=True)
            elif ci == ncols - 1:
                ax.tick_params(axis="y", right=True, labelright=True)
            if ri == 1:
                ax.tick_params(axis="x", bottom=True, labelbottom=True)
            ax.set_axisbelow(True)  # keep gridlines behind the scatter points (PathCollection zorder 1)
            ax.grid(True, linestyle="--", alpha=0.5)
            _apply_legend(ax, style.legend_by_role[_STEM_COLOR_ROLE[stem]], fontsize=13)
            sc_by[(method, stem)] = ax.scatter([], [])  # marker size is set per frame (_composite_frame)
    fig.subplots_adjust(left=l / fig_w, right=1 - r / fig_w, bottom=b / fig_h, top=1 - t / fig_h,
                        wspace=0, hspace=0)
    return fig, sc_by

def _8panel_render(leaf_stem, penult_stem, data, fpath, suptitle, style, limits=None):
    """Static PNG of the cross-method grid.
    `data` maps (method, stem) -> (proj, rgba). Per-cell axis limits come from each cell's own method,
    except cells whose (method, stem) is in `limits` -- the pooled pca_bounds='final' frame overriding the
    PCA row; the _RIGID methods' cells always auto-fit."""
    limits = {**{k: _limits_for(k[0], [data[k][0]]) for k in data}, **(limits or {})}  # k = (method, stem)
    fig, sc_by = _8panel_canvas(leaf_stem, penult_stem, limits, suptitle, _GIF_DPI, style)
    _composite_frame(sc_by, data, style.marker_size)
    fig.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

def _8panel_evolving_gif(leaf_stem, penult_stem, subject, viz_context, evals, names, cmaps, ema_tau,
                         limits, fpath_gif, style, orient=True, fname="projections.npz"):
    """Cross-method (one column per method) training-evolving GIF for one subject's leaf/penult pair.
    Sweeps every method's caches in lockstep (one eval loaded at a time, resolved for all methods) and
    streams the frames to disk, so peak memory is independent of the eval count; axes frozen to the
    precomputed cross-eval union (`limits`, keyed (method, stem)). `orient`/`fname` select per-eval
    (oriented) vs pooled (shared-frame, `orient=False`) sources."""
    stems = {leaf_stem, penult_stem}
    supt = lambda name: _manifold_title(_8PANEL_LABEL, viz_context, subject, suffix=f", {name}")
    fig, sc_by = _8panel_canvas(leaf_stem, penult_stem, limits, supt(names[0]), _GIF_DPI, style)
    def _frames():  # generator: one cache load per eval, resolved for every method, streamed to disk
        for name, resolved in _resolved_evals(evals, names, stems, _METHODS, cmaps, ema_tau, orient, fname):
            fig.suptitle(supt(name), fontsize=22, fontweight="bold", x=fig.subplotpars.left, ha="left")
            data = {k: (resolved[k][0], _rgba(resolved[k][1], resolved[k][2]))
                    for k in ((m, s) for m in _METHODS for s in stems)}
            _composite_frame(sc_by, data, style.marker_size)
            yield _canvas_frame(fig)

    _save_gif_stream(_frames(), fpath_gif, style.frame_ms)
    plt.close(fig)

def _render_grids(projs_by_method, cids_id, cids_ood, penults_id, penults_ood,
                  color_leaf, color_penult, nshot_id, color_nshot, legend_by_role,
                  dpath_vis, cfg_manif_viz, viz_context, tag, pca_limits=None):
    """Rank-0 render for one eval from ALREADY-ORIENTED projections ({method: {proj key: (N,2)}}), into
    dpath_vis/: the per-method stacked leaf(top)/penult(bottom) pairs (ID-only, OOD-only, the combined
    ID+OOD projection and its two partition-masked variants) under 2panel/<method>/, the 2x4 joint
    composite under 7panel/<method>/, and the cross-method grids (one column per method, one output per
    2panel subject) under 8panel/. `cfg_manif_viz`'s plot_2panel/plot_7panel/plot_8panel flags gate
    which groups are emitted. Each output is a static PNG.
    `pca_limits` (proj key -> (xlim, ylim)) freezes every PCA panel to that box instead of auto-fitting to
    this eval's points -- the pooled pca_bounds='final' frame (the _RIGID methods always auto-fit). Pure
    renderer -- orientation/coloring/cache are the caller's job (render_eval)."""
    marker_size = DATASET2MARKER_SIZE[viz_context.dataset]
    frame_ms = cfg_manif_viz["eval_duration"]  # unused for stills; carried for the evolving GIFs' schedule
    bg_color = cfg_manif_viz["bg_color"]
    suffix = f", {tag}"
    def bake(projs):  # every panel's (proj, rgba); the grids below tile them into the pairs/composite/stacks
        return {stem: (proj, _rgba(np.array([color_map[label] for label in labels]), alpha))
                for proj, labels, color_map, alpha, stem in _resolve_panels(
                    projs, cids_id, cids_ood, penults_id, penults_ood, color_leaf, color_penult, nshot_id, color_nshot)}
    comp = {m: bake(projs_by_method[m]) for m in _METHODS}
    jobs = []  # render jobs fanned out across cores below
    # per-method grids: 2panel stacked pairs + 7panel composite, under viz/<group>/<method_dir>/
    for method in _METHODS:
        for out_name, subject, grid in _GRIDS:
            group, stem = _grid_group(out_name)
            if not cfg_manif_viz[f"plot_{group}"]:  # 2panel / 7panel toggles
                continue
            suptitle = _manifold_title(method, viz_context, subject, suffix=suffix)
            col_titles = _COMPOSITE_COL_TITLES if out_name == "joint_panel" else None
            style = RenderStyle(method, marker_size, legend_by_role, frame_ms, bg_color, col_titles)
            sub = {s: comp[method][s] for s in _stems_of(grid)}  # only this grid's panels (smaller to ship to a worker)
            limits = {s: pca_limits[_STEM_PROJKEY[s]] for s in sub} if method == "PCA" and pca_limits else None
            fpath = dpath_vis / group / _METHOD_DIR[method] / f"{stem}.png"
            fpath.parent.mkdir(parents=True, exist_ok=True)
            jobs.append((composite_plot, (grid, sub, fpath, suptitle, style, limits)))
    # cross-method grids (one column per method, _METHODS order left -> right): under viz/8panel/
    if cfg_manif_viz["plot_8panel"]:
        style_8panel = RenderStyle(None, marker_size, legend_by_role, frame_ms, bg_color)
        for out_name, subject, leaf_stem, penult_stem in _8PANEL_SUBJECTS:
            suptitle = _manifold_title(_8PANEL_LABEL, viz_context, subject, suffix=suffix)
            data = {(m, s): comp[m][s] for m in _METHODS for s in (leaf_stem, penult_stem)}
            limits = {("PCA", s): pca_limits[_STEM_PROJKEY[s]] for s in (leaf_stem, penult_stem)} if pca_limits else None
            fpath = dpath_vis / "8panel" / f"{out_name}.png"
            fpath.parent.mkdir(parents=True, exist_ok=True)
            jobs.append((_8panel_render, (leaf_stem, penult_stem, data, fpath, suptitle, style_8panel, limits)))
    _parallel_render(jobs)

def _compute_projections(embs_id, embs_ood, cfg_tsne, chunk_elems):
    """COLLECTIVE -- must be entered by every rank together. Compute the RAW t-SNE (sharded across ranks
    when world_size > 1, see `_tsne_torch`) and PCA projections for id/ood/joint from the all-gathered
    image embeddings (`embs_id`/`embs_ood` are the per-rank GPU tensors, identical on every rank). PCA is
    computed redundantly per rank -- it is deterministic (same input + fixed seed) and feeds the shared
    t-SNE init; the sharded t-SNE all-gathers the full layout each iteration, so every rank returns
    identical (tsne_projs, pca_projs), each keyed id/ood/joint."""
    embs = {"id": embs_id, "ood": embs_ood, "joint": torch.cat([embs_id, embs_ood], dim=0)}
    pca_projs = {k: compute_pca(e.detach().cpu().numpy()) for k, e in embs.items()}
    tsne_projs = {
        k: compute_tsne(embs[k], perplexity=cfg_tsne["perplexity"], init=_pca_init(pca_projs[k]), chunk_elems=chunk_elems, n_iter=cfg_tsne["n_iter"])
        for k in embs
    }
    return tsne_projs, pca_projs

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

def compute_projections(eval_bundle_id, eval_bundle_ood, dpath_cache, cfg_manif_viz, chunk_elems):
    """COLLECTIVE -- every rank must enter. Compute the raw (sharded) t-SNE + PCA from the live,
    all-gathered eval embeddings and, on rank 0, cache them to dpath_cache/projections.npz. This is the
    training pipeline's ONLY in-loop viz work; orientation, coloring, and rendering are a separate pass
    over the cache (render_eval / render_evolution), kept off the collective path. Returns None.

    The raw eval embeddings are also persisted to dpath_cache/embs.npz (float16 + cids): they feed the
    post-trial UMAP fits (`compute_umap_projections`, `compute_umap_pooled`) and the end-of-trial pooled
    t-SNE/PCA (`compute_pooled_projections`), all of which run after the in-loop compute is done."""
    _log("computing projections")
    # ALL RANKS: sharded t-SNE + PCA projections (collective ops inside)
    tsne_projs, pca_projs = _compute_projections(
        eval_bundle_id["embs_img"], eval_bundle_ood["embs_img"], cfg_manif_viz["tsne"], chunk_elems)
    if dist.is_initialized() and dist.get_rank() != 0:
        return  # non-rank-0 ranks only participate in the collective compute
    dpath_cache.mkdir(parents=True, exist_ok=True)
    cache = {"cids_id": np.asarray(eval_bundle_id["cids_img"]), "cids_ood": np.asarray(eval_bundle_ood["cids_img"])}
    for k in pca_projs:
        cache[f"pca_{k}"] = pca_projs[k]
    for k in tsne_projs:
        cache[f"tsne_{k}"] = tsne_projs[k]
    np.savez(dpath_cache / "projections.npz", **cache)
    # raw embeddings (float16) for the post-trial UMAP fits + the end-of-trial pooled projection
    np.savez(dpath_cache / "embs.npz",
             embs_id=eval_bundle_id["embs_img"].detach().cpu().to(torch.float16).numpy(),
             embs_ood=eval_bundle_ood["embs_img"].detach().cpu().to(torch.float16).numpy(),
             cids_id=np.asarray(eval_bundle_id["cids_img"]),
             cids_ood=np.asarray(eval_bundle_ood["cids_img"]))
    _log("projections complete")

_POOLED_SEED = 42  # fixed seed for the pooled subsample: every rank must draw the SAME points so the
                   # pool (and thus the collective sharded t-SNE input) is identical across ranks

def _stratified_subsample(cids, m, seed):
    """~m row indices class-stratified over `cids` (each class contributes ~proportionally to its size),
    deterministic given (cids, seed); returns sorted indices, or all indices when m >= len(cids). Reused
    across thresholds -- the eval set is in identical per-sample order at every threshold, so the same
    indices select the SAME physical samples in every frame, and the pooled plots track those samples
    migrating through the one shared layout."""
    cids = np.asarray(cids)
    n = len(cids)
    if m >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    picks = []
    for c in np.unique(cids):  # sorted -> deterministic class order
        idx_c = np.where(cids == c)[0]
        k = min(len(idx_c), int(round(m * len(idx_c) / n)))
        if k > 0:
            picks.append(rng.choice(idx_c, size=k, replace=False))
    return np.sort(np.concatenate(picks)) if picks else np.arange(0)

def _pooled_block_sizes(m_id, m_ood):
    """(proj key, per-threshold row count) for the pooled layout -- the block each threshold occupies in
    the pooled fit's rows, in `_PROJ_KEYS` order (joint holds its ID rows then its OOD rows)."""
    return (("id", m_id), ("ood", m_ood), ("joint", m_id + m_ood))

def _pooled_pools(dirs, idx_id, idx_ood):
    """Per-threshold subsample blocks (float16 on disk -> float32 for compute), pooled into id / ood /
    joint. Reading e["embs_*"] materializes one threshold's full embeddings transiently, then indexes down
    to the subsample, so peak memory is ~one threshold's embeddings plus the (small) subsample pool. Rows
    are laid out in per-threshold contiguous blocks ([threshold0, threshold1, ...]) so a threshold's block
    is a single contiguous slice (`_pooled_block_sizes`)."""
    id_blocks, ood_blocks = [], []
    for d in dirs:
        with np.load(d / "embs.npz") as e:
            id_blocks.append(e["embs_id"][idx_id].astype(np.float32))
            ood_blocks.append(e["embs_ood"][idx_ood].astype(np.float32))
    pools = {
        "id": np.concatenate(id_blocks, axis=0),
        "ood": np.concatenate(ood_blocks, axis=0),
        "joint": np.concatenate([np.concatenate([i, o], axis=0) for i, o in zip(id_blocks, ood_blocks)], axis=0),
    }
    del id_blocks, ood_blocks  # free the (duplicated in joint) block lists before the fit's buffers allocate
    return pools

def compute_umap_projections(dpath_evals, cfg_manif_viz):
    """CPU, single process. Fit each eval's UMAP from its cached embs.npz and append umap_{id,ood,joint}
    to that eval's projections.npz. Run by the post-trial render worker BEFORE any rendering, so UMAP
    never touches the training loop's collective GPU path.

    Evals are swept chronologically and each fit is initialized from the PREVIOUS eval's UMAP layout --
    the eval set is in identical per-sample order at every threshold, so row i is the same sample in every
    frame and the previous layout is a valid starting point. The first eval seeds from its own cached PCA
    (the same projection t-SNE inits from). Chaining the init keeps the sequence in one basin at the
    source, which cross-eval orientation cannot do on its own: orientation removes residual rotation and
    reflection, not a genuinely reorganized layout.

    Idempotent: an eval that already carries UMAP is not refit, but its cached layout is still read to
    seed the next eval's init, so an interrupted sweep resumes to the same result as an unbroken one."""
    cfg_umap = cfg_manif_viz["umap"]
    prev = None  # previous eval's {proj key: layout} -> this eval's init
    for d in _ordered_eval_dirs(dpath_evals, "embs.npz"):
        with np.load(d / "projections.npz") as npz:
            cache = dict(npz)  # materialize: the same path is rewritten below with UMAP appended
        if "umap_sphere_joint" in cache:  # already fit -- reuse as the next eval's init rather than refitting
            prev = {m: {k: cache[f"{m}_{k}"] for k in _PROJ_KEYS} for m in ("umap", "umap_sphere")}
            continue
        with np.load(d / "embs.npz") as e:
            embs_id, embs_ood = e["embs_id"].astype(np.float32), e["embs_ood"].astype(np.float32)
        by_key = {"id": embs_id, "ood": embs_ood, "joint": np.concatenate([embs_id, embs_ood], axis=0)}
        projs = {}
        for k in _PROJ_KEYS:
            knn = _umap_knn(by_key[k], cfg_umap)  # one neighbor search, both fits
            flat_init = cache[f"pca_{k}"] if prev is None else prev["umap"][k]
            sph_init = None if prev is None else _xyz_to_angles(prev["umap_sphere"][k])
            projs[("umap", k)] = compute_umap(by_key[k], cfg_umap, flat_init, knn=knn)
            projs[("umap_sphere", k)] = compute_umap(by_key[k], cfg_umap, sph_init, spherical=True, knn=knn)
        cache.update({f"{m}_{k}": projs[(m, k)] for m, k in projs})
        np.savez(d / "projections.npz", **cache)
        prev = {m: {k: projs[(m, k)] for k in _PROJ_KEYS} for m in ("umap", "umap_sphere")}

def compute_umap_pooled(dpath_evals, cfg_manif_viz):
    """CPU, single process. Fit ONE shared UMAP over all eval thresholds' pooled embeddings and append the
    per-threshold blocks as umap_{id,ood,joint} to each <eval>/projections_pooled.npz -- the UMAP
    counterpart of `compute_pooled_projections`, run by the post-trial render worker. Reuses the subsample
    that compute_pooled_projections recorded in the pooled cache (idx_id/idx_ood), so the pooled UMAP
    covers exactly the same points in the same row order as the pooled PCA/t-SNE. One fit spans the whole
    sequence, so there is no init chain and no orientation here -- the frame is shared by construction.
    Idempotent (returns immediately when the blocks are already cached)."""
    dirs = _ordered_eval_dirs(dpath_evals, "projections_pooled.npz")
    if not dirs:
        return
    caches = []
    for d in dirs:
        with np.load(d / "projections_pooled.npz") as npz:
            caches.append(dict(npz))  # materialize: the same paths are rewritten below with UMAP appended
    if "umap_sphere_joint" in caches[0]:
        return
    idx_id, idx_ood = caches[0]["idx_id"], caches[0]["idx_ood"]
    m_id, m_ood = len(idx_id), len(idx_ood)
    pools = _pooled_pools(dirs, idx_id, idx_ood)
    cfg_umap = cfg_manif_viz["umap"]
    projs = {}
    for k in _PROJ_KEYS:
        knn = _umap_knn(pools[k], cfg_umap)  # one neighbor search, both fits
        # PCA over the same pool seeds the flat fit, mirroring the pooled t-SNE's init; the spherical
        # fit has no planar init to inherit, so it takes UMAP's own spectral one
        projs[("umap", k)] = compute_umap(pools[k], cfg_umap, _pca_2d(pools[k]), knn=knn)
        projs[("umap_sphere", k)] = compute_umap(pools[k], cfg_umap, None, spherical=True, knn=knn)
    del pools
    for t, (d, cache) in enumerate(zip(dirs, caches)):
        for k, blk in _pooled_block_sizes(m_id, m_ood):
            for m in ("umap", "umap_sphere"):
                cache[f"{m}_{k}"] = projs[(m, k)][t * blk:(t + 1) * blk]
        np.savez(d / "projections_pooled.npz", **cache)

def compute_pooled_projections(dpath_evals, cfg_manif_viz, budget, chunk_elems):
    """COLLECTIVE -- every rank must enter. Fit ONE shared PCA + t-SNE over ALL eval thresholds'
    embeddings pooled (read from each eval's embs.npz) and, on rank 0, write per-threshold masked blocks
    to <eval>/projections_pooled.npz. The pooled UMAP is fit off this path, post-trial, by
    `compute_umap_pooled`, which reuses the subsample recorded here. The geometry is shared, so each threshold is a masked subset of the
    single layout (no orientation needed) and the plots show the eval set migrating through a fixed frame
    as training progresses.

    The t-SNE's per-iteration repulsion compute is O(N^2) (memory is linear and not a constraint), so to
    bound the pooled fit's runtime the pool is class-stratified subsampled to ~budget total points across
    all thresholds (all samples used when the full pool (N_id+N_ood)*n_thresholds <= budget). The subsample
    indices are fixed across thresholds (identical eval order), so a point is the SAME sample in every
    frame. id/ood/joint are fit independently (mirroring the per-eval compute). Rows are laid out in
    per-threshold contiguous blocks ([threshold0, threshold1, ...]; each joint block is its ID rows then
    its OOD rows) so a threshold's block is a single contiguous slice.

    Deterministic subsample + deterministic PCA init => the pool and init are identical on every rank, as
    the collective sharded t-SNE requires."""
    dirs = _ordered_eval_dirs(dpath_evals, "embs.npz")
    if not dirs:
        return
    T = len(dirs)
    with np.load(dirs[0] / "embs.npz") as e0:  # eval set is fixed across thresholds -> any threshold's cids
        cids_id, cids_ood = np.asarray(e0["cids_id"]), np.asarray(e0["cids_ood"])
    n_id, n_ood, n_full = len(cids_id), len(cids_ood), len(cids_id) + len(cids_ood)
    per_thresh = min(n_full * T, budget) / T  # target pooled ID + OOD points per threshold (all samples used when the full pool <= budget)
    idx_id = _stratified_subsample(cids_id, min(n_id, int(round(per_thresh * n_id / n_full))), _POOLED_SEED)
    idx_ood = _stratified_subsample(cids_ood, min(n_ood, int(round(per_thresh * n_ood / n_full))), _POOLED_SEED)
    m_id, m_ood = len(idx_id), len(idx_ood)  # actual per-threshold counts (uniform across thresholds)
    _log(f"pooling {T} thresholds -> t-SNE on {T * (m_id + m_ood)} pts (id {T * m_id}, ood {T * m_ood}) at budget {budget}")

    pools = _pooled_pools(dirs, idx_id, idx_ood)
    cfg_tsne = cfg_manif_viz["tsne"]
    tsne_projs, pca_projs = {}, {}
    for k, pool in pools.items():  # collective sharded t-SNE per subject (id/ood/joint), one at a time
        pca_projs[k] = compute_pca(pool)
        tsne_projs[k] = compute_tsne(pool, perplexity=cfg_tsne["perplexity"], init=_pca_init(pca_projs[k]), chunk_elems=chunk_elems, n_iter=cfg_tsne["n_iter"])
    if dist.is_initialized() and dist.get_rank() != 0:
        return  # non-rank-0 ranks only participate in the collective compute
    # split each pooled projection into per-threshold blocks; cache them (projections.npz schema) per eval
    sub_cids_id, sub_cids_ood = cids_id[idx_id], cids_ood[idx_ood]
    for t, d in enumerate(dirs):
        # idx_* record which rows of embs.npz this subsample took, so the post-trial pooled UMAP
        # (compute_umap_pooled) fits the exact same points in the same row order instead of re-deriving them
        cache = {"cids_id": sub_cids_id, "cids_ood": sub_cids_ood, "idx_id": idx_id, "idx_ood": idx_ood}
        for k, blk in _pooled_block_sizes(m_id, m_ood):
            sl = slice(t * blk, (t + 1) * blk)
            cache[f"pca_{k}"] = pca_projs[k][sl]
            cache[f"tsne_{k}"] = tsne_projs[k][sl]
        np.savez(d / "projections_pooled.npz", **cache)
    _log("pooled projections complete")

def _load_projections(dpath_cache, fname="projections.npz"):
    """Load an eval's cached raw projections from dpath_cache/<fname>:
    ({method: {proj key: (N,2)}}, cids_id, cids_ood) -- one entry per `_METHODS`, each keyed id/ood/joint.
    `fname` selects the per-eval independent cache (projections.npz) or the pooled shared-frame cache
    (projections_pooled.npz), which share this schema."""
    npz = np.load(dpath_cache / fname)
    return ({m: {k: npz[f"{_METHOD_DIR[m]}_{k}"] for k in _PROJ_KEYS} for m in _METHODS},
            list(npz["cids_id"]), list(npz["cids_ood"]))

def _ordered_eval_dirs(dpath_evals, fname="projections.npz"):
    """Eval dirs that hold a cached <fname>, in chronological order (base, eval1..evalN). `fname`
    selects the per-eval cache (projections.npz), the pooled cache (projections_pooled.npz), or the raw
    embedding cache (embs.npz, swept by the pooled compute)."""
    return sorted((d for d in dpath_evals.iterdir() if (d / fname).exists()),
                  key=lambda d: _eval_sort_key(d.name))

def _ema_through(dpath_evals, eval_name, ema_tau):
    """Accumulate the orientation reference over the evals chronologically BEFORE `eval_name`, so it
    seeds `eval_name`'s orientation -- identical to that eval's frame in the evolving GIF (which sweeps
    the same caches in the same order). Recomputed from the on-disk caches each call, so the pipeline
    carries no live/resume orientation state. Returns {key: ref} ({} when `eval_name` is the first eval)."""
    ref = {}  # keyed (method, proj key) -- each method/subject pair has its own orientation reference
    for d in _ordered_eval_dirs(dpath_evals):
        if d.name == eval_name:
            break
        projs_by_method, cids_id, cids_ood = _load_projections(d)
        cids_by = _cids_by(cids_id, cids_ood)
        for m in _METHODS:
            for k, proj in projs_by_method[m].items():
                _, ref[(m, k)] = _orient(m, proj, cids_by[k], ref.get((m, k)), ema_tau)
    return ref

def _save_orient_ref(dpath_eval, ref, ema_tau):
    """Cache this eval's OUTGOING orientation reference (the running per-(method, proj key) {class: CoM} through
    this eval) so the next eval's render reads it in O(1) instead of re-sweeping every prior eval
    (see `_incoming_ref`). ema_tau and the method set are stored alongside so a render under a different
    smoothing factor -- or a reference keyed by a different set of methods -- recomputes rather than
    silently reusing a reference that no longer answers to the keys the render looks up."""
    save_pickle({"ema_tau": ema_tau, "methods": list(_METHODS), "ref": ref}, dpath_eval / "orient_ref.pkl")

def _load_orient_ref(dpath_eval, ema_tau):
    """This eval's cached outgoing orientation reference, or None when absent or written under a different
    ema_tau / method set (forcing a correct recompute rather than reusing a reference the render can no
    longer look up -- the keys are (method, proj key), so a reference built for a different method set
    would resolve to nothing and silently un-orient every eval)."""
    fpath = dpath_eval / "orient_ref.pkl"
    if not fpath.exists():
        return None
    blob = load_pickle(fpath)
    return None if (blob["ema_tau"], blob["methods"]) != (ema_tau, list(_METHODS)) else blob["ref"]

def _incoming_ref(dpath_evals, eval_name, ema_tau):
    """The orientation reference accumulated over the evals chronologically BEFORE `eval_name`, read in
    O(1) from the immediately-preceding eval's cached outgoing reference -- the memo of the same
    chronological orient sweep `_ema_through` would do. Falls back to recomputing from the raw caches
    (`_ema_through`) when that cache is absent or was written under a different ema_tau, so the result is
    always identical to the full sweep. Returns {(method, proj key): {class: CoM}} ({} for the first eval)."""
    ordered = _ordered_eval_dirs(dpath_evals)
    idx = [d.name for d in ordered].index(eval_name)
    if idx == 0:
        return {}
    cached = _load_orient_ref(ordered[idx - 1], ema_tau)
    return cached if cached is not None else _ema_through(dpath_evals, eval_name, ema_tau)

def _no_panels_enabled(cfg_manif_viz):
    """True when every manifold-viz panel group is toggled off (manif_viz.plot_2panel/plot_7panel/
    plot_8panel) -- the render passes then short-circuit, doing no setup/render and writing no (empty)
    viz dirs."""
    return not (cfg_manif_viz["plot_2panel"] or cfg_manif_viz["plot_7panel"] or cfg_manif_viz["plot_8panel"])

def _final_pca_limits(dpath_final, fname):
    """PCA axis limits per proj key frozen to the FINAL eval's pooled projection (id/ood/joint each its
    own bounding box) -- the manif_viz.pooled.pca_bounds='final' frame shared by every pooled PCA
    plot, so the per-threshold plots and the evolving GIF all sit in the converged final layout's box
    (earlier thresholds' points can fall outside it)."""
    projs_by_method, _, _ = _load_projections(dpath_final, fname)
    return {k: _common_limits([projs_by_method["PCA"][k]]) for k in _PROJ_KEYS}

@rank0
def render_eval(dpath_evals, eval_name, cfg_manif_viz, viz_context, orient=True, fname="projections.npz"):
    """Rank-0. Render one eval's plots from its cached projections into <eval_name>/viz(_pooled)/.

    Default (per-eval, `orient=True`, projections.npz): every method's independently-fit projection is
    aligned against the reference accumulated over the prior evals on disk (rigid for t-SNE/UMAP, sign-only
    for PCA), so it matches that eval's frame in the evolving GIF -- and needs no live state.

    Pooled (`orient=False`, fname=projections_pooled.npz): the projection already shares one frame across
    thresholds, so it is plotted as-is (no orientation, no ref cache) into <eval_name>/viz_pooled/. The
    cache holds only this threshold's subsample, so colors are still built from the FULL eval set
    (projections.npz) -- coloring by the subsample would reorder the count-ranked hues and break color
    correspondence with the other plots. `cfg_manif_viz`'s plot_2/4/7panel flags gate which panel groups."""
    if _no_panels_enabled(cfg_manif_viz):
        return
    dpath_eval = dpath_evals / eval_name
    projs_by_method, cids_id, cids_ood = _load_projections(dpath_eval, fname)
    # color maps span the whole dataset so a class is colored identically in every plot; colors are
    # assigned in order of how many plotted (ID+OOD) samples each class/penult-group has -> always the
    # FULL eval set (projections.npz), so pooled's subsample keeps each class's per-eval-plot color
    if fname == "projections.npz":
        cids_id_full, cids_ood_full = cids_id, cids_ood
    else:
        _, cids_id_full, cids_ood_full = _load_projections(dpath_eval)
    color_leaf, color_penult, color_nshot, cid_2_penult, cid_2_nshot, nst_names = \
        _build_color_maps(viz_context, list(cids_id_full) + list(cids_ood_full), cfg_manif_viz["color"])
    penults_id = [cid_2_penult[c] for c in cids_id]
    penults_ood = [cid_2_penult[c] for c in cids_ood]
    nshot_id = [cid_2_nshot[c] for c in cids_id]  # OOD samples are drawn black, not bucketed
    if orient:
        ema_tau = cfg_manif_viz["orient"]["ema_tau"]
        ref = _incoming_ref(dpath_evals, eval_name, ema_tau)  # reference through the prior evals (O(1) cache read)
        cids_by = _cids_by(cids_id, cids_ood)
        render_projs = {}
        for m in _METHODS:
            render_projs[m] = {}
            for k, proj in projs_by_method[m].items():
                render_projs[m][k], ref[(m, k)] = _orient(m, proj, cids_by[k], ref.get((m, k)), ema_tau)
        _save_orient_ref(dpath_eval, ref, ema_tau)  # cache outgoing ref (before plotting) so the next eval reads it in O(1)
    else:  # pooled: shared frame across thresholds -> no orientation
        render_projs = projs_by_method
    tag = eval_name
    pca_limits = (_final_pca_limits(_ordered_eval_dirs(dpath_evals, fname)[-1], fname)
                  if not orient and cfg_manif_viz["pooled"]["pca_bounds"] == "final" else None)
    _render_grids(render_projs, cids_id, cids_ood, penults_id, penults_ood,
                  color_leaf, color_penult, nshot_id, color_nshot, _legend_specs(color_nshot, nst_names),
                  dpath_eval / ("viz" if orient else "viz_pooled"), cfg_manif_viz, viz_context, tag,
                  pca_limits)

def _eval_sort_key(name):
    """Chronological order of eval dirs: base first, then eval1..evalN ascending."""
    return 0 if name == "base" else int(name.removeprefix("eval"))

@rank0
def _evolution_limits(evals, ema_tau, orient=True, fname="projections.npz"):
    """One streaming pass over the caches accumulating the cross-eval axis bounds per (method, proj key)
    while holding only one eval at a time: a bounding box (running min/max) for the bbox methods and a
    square bound (running max |coord|) for the _RIGID ones. The spherical method needs neither -- its
    ball is always unit-radius -- but is still swept so its orientation reference advances in lockstep
    with the render's. The accumulated bounds are taken over ORIENTED projections --
    orientation moves points (rigidly about the origin for t-SNE/UMAP, a sign flip for PCA), so the bound
    must follow the same orientation sweep as the render. Reuses _common_limits/_square_limits on the
    accumulated extremes so the frozen axes are identical to materializing every eval. `orient=False`/
    `fname` (pooled mode) take the bound over the raw pooled projections (already one shared frame).
    Returns {(method, proj key): (xlim, ylim)} for every method x id/ood/joint."""
    pairs = [(m, k) for m in _METHODS for k in _PROJ_KEYS]
    lo = {p: None for p in pairs}       # bbox methods: running per-axis min/max
    hi = {p: None for p in pairs}
    absmax = {p: 0.0 for p in pairs}    # _RIGID methods: running max |coord| over oriented projections
    ref = {}                            # (method, proj key) -> running orientation reference (per-class CoM)
    for d in evals:
        projs_by_method, cids_id, cids_ood = _load_projections(d, fname)
        cids_by = _cids_by(cids_id, cids_ood)
        for m, k in pairs:
            proj = projs_by_method[m][k]
            if orient:  # pooled projections already share one frame -> bound the raw projection
                proj, ref[(m, k)] = _orient(m, proj, cids_by[k], ref.get((m, k)), ema_tau)
            if m in _SPHERICAL:
                continue  # the ball is fixed at unit radius -- nothing to accumulate (but still oriented above)
            if m in _RIGID:
                absmax[(m, k)] = max(absmax[(m, k)], float(np.abs(proj).max()))
            else:
                p_lo, p_hi = proj.min(axis=0), proj.max(axis=0)
                lo[(m, k)] = p_lo if lo[(m, k)] is None else np.minimum(lo[(m, k)], p_lo)
                hi[(m, k)] = p_hi if hi[(m, k)] is None else np.maximum(hi[(m, k)], p_hi)

    def _bound(p):
        if p[0] in _SPHERICAL:
            return _SPHERE_LIMITS
        if p[0] in _RIGID:
            return _square_limits([np.array([[absmax[p], absmax[p]]])])
        return _common_limits([np.stack([lo[p], hi[p]])])

    return {p: _bound(p) for p in pairs}

def render_evolution(dpath_evals, dpath_out, cfg_manif_viz, viz_context, orient=True, fname="projections.npz"):
    """Rank-0. Assemble one GIF per grid (`_GRIDS`) showing the training evolution
    (base -> eval1 -> ... -> evalN): each eval contributes one frame, then the GIF hard-cuts
    to the next eval, axes/gridlines frozen across evals so only the points move. Reads each eval's
    cached <fname> and writes the per-method grids under dpath_out/{2panel,7panel}/<method>/ plus
    the cross-method grids (one column per method) under dpath_out/8panel/.

    Default (per-eval, projections.npz) orients every method by aligning each eval to a running reference
    swept across evals -- the same orientation `render_eval` reproduces per eval. Pooled (`orient=False`,
    projections_pooled.npz, dpath_out=viz_pooled) skips orientation: the pooled projections already share
    one frame, so the GIF just masks the single layout to each threshold's subsample. Colors always come
    from the FULL eval set (projections.npz) so class colors match the other plots. Caches are streamed one
    eval at a time (frozen limits precomputed in a single pass), so peak memory doesn't scale with the number of checkpoints."""
    if _no_panels_enabled(cfg_manif_viz):
        return
    evals = _ordered_eval_dirs(dpath_evals, fname)
    if not evals:
        return
    names = [d.name for d in evals]

    cfg_color = cfg_manif_viz["color"]
    marker_size = DATASET2MARKER_SIZE[viz_context.dataset]
    bg_color = cfg_manif_viz["bg_color"]
    frame_ms = cfg_manif_viz["eval_duration"]  # one frame per eval, so each eval shows for eval_duration
    ema_tau = cfg_manif_viz["orient"]["ema_tau"]
    # eval set is fixed across checkpoints, so any eval's cids give the same (count-ordered) colors; always
    # the FULL eval set (projections.npz, explicit -- NOT the pooled subsample) so a class keeps its color
    _, cids_id, cids_ood = _load_projections(evals[-1], "projections.npz")
    color_leaf, color_penult, color_nshot, cid_2_penult, cid_2_nshot, nst_names = \
        _build_color_maps(viz_context, cids_id + cids_ood, cfg_color)
    cmaps = (color_leaf, color_penult, color_nshot, cid_2_penult, cid_2_nshot)  # shipped to workers (O(classes))
    legends = _legend_specs(color_nshot, nst_names)  # per-color-role coloring legend for every panel
    limits_by = _evolution_limits(evals, ema_tau, orient, fname)  # frozen axes per (method, proj key), single streaming pass
    if not orient and cfg_manif_viz["pooled"]["pca_bounds"] == "final":  # freeze pooled PCA to the final threshold's box
        for k, lim in _final_pca_limits(evals[-1], fname).items():
            limits_by[("PCA", k)] = lim

    jobs = []  # one evolving-GIF job per (render target, grid); fanned out across cores below
    # per-method grids, under <dpath_out>/<group>/<method_dir>/
    for method in _METHODS:
        for out_name, subject, grid in _GRIDS:
            group, stem = _grid_group(out_name)
            if not cfg_manif_viz[f"plot_{group}"]:  # 2panel / 7panel toggles
                continue
            stems = _stems_of(grid)
            col_titles = _COMPOSITE_COL_TITLES if out_name == "joint_panel" else None
            style = RenderStyle(method, marker_size, legends, frame_ms, bg_color, col_titles)
            limits = {s: limits_by[(method, _STEM_PROJKEY[s])] for s in stems}
            fpath = dpath_out / group / _METHOD_DIR[method] / f"{stem}.gif"
            fpath.parent.mkdir(parents=True, exist_ok=True)
            jobs.append((composite_evolving_gif, (grid, subject, viz_context, evals, names, cmaps,
                                                   ema_tau, limits, fpath, style, orient, fname)))
    # cross-method evolving GIFs (one column per method): under <dpath_out>/8panel/
    if cfg_manif_viz["plot_8panel"]:
        style_8panel = RenderStyle(None, marker_size, legends, frame_ms, bg_color)
        for out_name, subject, leaf_stem, penult_stem in _8PANEL_SUBJECTS:
            limits = {(m, s): limits_by[(m, _STEM_PROJKEY[s])] for m in _METHODS for s in (leaf_stem, penult_stem)}
            fpath = dpath_out / "8panel" / f"{out_name}.gif"
            fpath.parent.mkdir(parents=True, exist_ok=True)
            jobs.append((_8panel_evolving_gif, (leaf_stem, penult_stem, subject, viz_context, evals, names,
                                                cmaps, ema_tau, limits, fpath, style_8panel, orient, fname)))
    _parallel_render(jobs)
