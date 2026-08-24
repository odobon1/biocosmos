import math

import numpy as np
import torch

from utils.manif_viz import (_hbeta_search, _knn, _orient, _sparse_joint_p, _tsne_torch, compute_pca,
                            compute_umap, orient_pca, orient_proj, orient_sphere)


def _blob(n=300, d=16, seed=0):
    torch.manual_seed(seed)
    return torch.randn(n, d)


def test_knn_excludes_self_and_sorts_ascending() -> None:
    X = _blob()
    d2, idx = _knn(X, X, k=20, r0=0, chunk=64)  # chunk < n exercises the chunk boundary
    assert (idx != torch.arange(X.shape[0])[:, None]).all()  # self never its own neighbor
    assert (d2.diff(dim=1) >= 0).all()  # nearest-first
    assert (d2 >= 0).all()


def test_hbeta_search_hits_target_perplexity() -> None:
    # the binary search tunes each row's Gaussian so the conditional distribution's Shannon
    # entropy equals log(perplexity)
    perplexity = 25.0
    X = _blob()
    d2, _ = _knn(X, X, k=int(3 * perplexity), r0=0, chunk=1000)
    P = _hbeta_search(d2, perplexity)
    assert torch.allclose(P.sum(1), torch.ones(X.shape[0]), atol=1e-5)  # rows normalized
    H = -(P * P.clamp_min(1e-30).log()).sum(1)
    assert torch.allclose(H, torch.full_like(H, math.log(perplexity)), atol=1e-3)


def test_sparse_joint_p_is_symmetric_and_sums_to_one() -> None:
    n, k = 300, 60
    X = _blob(n)
    d2, idx = _knn(X, X, k=k, r0=0, chunk=1000)
    rowptr, cols, vals = _sparse_joint_p(d2, idx, 20.0, n, 0, [n])
    assert rowptr[-1].item() == len(cols) == len(vals) <= 2 * n * k
    assert (vals > 0).all()
    P = torch.zeros((n, n))
    P[torch.repeat_interleave(torch.arange(n), rowptr.diff()), cols] = vals
    assert torch.equal(P, P.t())  # exact: symmetrization emits each edge with its transpose
    assert abs(P.sum().item() - 1.0) < 1e-5  # joint distribution


def test_tsne_separates_gaussian_clusters() -> None:
    torch.manual_seed(1)
    centers = torch.tensor([[20.0] * 10, [-20.0] * 10, [20.0] * 5 + [-20.0] * 5])
    X = torch.cat([c + torch.randn(60, 10) for c in centers])
    init = (X[:, :2] / X[:, 0].std() * 1e-4).numpy().astype(np.float32)
    Y = _tsne_torch(X, init, perplexity=30.0, chunk_elems=1 << 20, n_iter=300, device="cpu")
    assert Y.shape == (180, 2) and np.isfinite(Y).all()
    lab = np.repeat([0, 1, 2], 60)
    cents = np.stack([Y[lab == i].mean(0) for i in range(3)])
    intra = max(np.linalg.norm(Y[lab == i] - cents[i], axis=1).mean() for i in range(3))
    inter = min(np.linalg.norm(cents[i] - cents[j]) for i in range(3) for j in range(i + 1, 3))
    assert inter > 3 * intra, f"clusters not separated: inter={inter:.2f} intra={intra:.2f}"


_UMAP_CFG = {"n_neighbors": 15, "min_dist": 0.1, "n_iter": 200, "n_iter_sphere": 100}


def test_umap_separates_gaussian_clusters() -> None:
    # UMAP is the one method fit outside the training loop (post-trial, on CPU); this pins the
    # compute_umap contract -- (N, 2) float32 out, and structure actually preserved.
    rng = np.random.default_rng(1)
    centers = rng.normal(scale=8.0, size=(3, 10))
    X = np.concatenate([c + rng.normal(scale=0.4, size=(60, 10)) for c in centers]).astype(np.float32)
    Y = compute_umap(X, _UMAP_CFG, compute_pca(X))
    assert Y.shape == (180, 2) and Y.dtype == np.float32 and np.isfinite(Y).all()
    lab = np.repeat([0, 1, 2], 60)
    cents = np.stack([Y[lab == i].mean(0) for i in range(3)])
    intra = max(np.linalg.norm(Y[lab == i] - cents[i], axis=1).mean() for i in range(3))
    inter = min(np.linalg.norm(cents[i] - cents[j]) for i in range(3) for j in range(i + 1, 3))
    assert inter > 3 * intra, f"clusters not separated: inter={inter:.2f} intra={intra:.2f}"


def test_umap_output_is_origin_centered() -> None:
    # everything downstream assumes an origin-centered layout the way _tsne_torch produces: orient_proj
    # rotates about the origin with no translation term, and _square_limits frames _RIGID methods
    # symmetrically about it. umap-learn does not center its own output, so compute_umap must.
    rng = np.random.default_rng(3)
    X = rng.normal(size=(150, 10)).astype(np.float32)
    Y = compute_umap(X, _UMAP_CFG, compute_pca(X))
    radius = np.linalg.norm(Y, axis=1).mean()
    assert np.allclose(Y.mean(axis=0), 0.0, atol=1e-4 * max(radius, 1.0)), Y.mean(axis=0)


def test_umap_honors_the_supplied_init() -> None:
    # the eval sequence chains each fit's init off the previous eval's layout, so a compute_umap that
    # ignored `init` would silently disable that continuity -- assert the init actually reaches the fit.
    rng = np.random.default_rng(2)
    X = rng.normal(size=(150, 10)).astype(np.float32)
    a = compute_umap(X, _UMAP_CFG, compute_pca(X))
    b = compute_umap(X, _UMAP_CFG, (rng.normal(size=(150, 2)) * 50).astype(np.float32))
    assert not np.allclose(a, b, atol=1e-3)


def test_tsne_is_deterministic() -> None:
    X = _blob(150)
    init = (X[:, :2] / X[:, 0].std() * 1e-4).numpy().astype(np.float32)
    Y1 = _tsne_torch(X, init, perplexity=15.0, chunk_elems=1 << 20, n_iter=60, device="cpu")
    Y2 = _tsne_torch(X, init, perplexity=15.0, chunk_elems=1 << 20, n_iter=60, device="cpu")
    assert np.array_equal(Y1, Y2)


def _labelled_layout(n_per=40, n_cls=5, seed=0):
    """A 2D layout with well-separated per-class clusters -- distinct class CoMs are what both
    orientation routines align on."""
    rng = np.random.default_rng(seed)
    centers = np.array([[np.cos(2 * math.pi * i / n_cls), np.sin(2 * math.pi * i / n_cls)] for i in range(n_cls)]) * 5
    proj = np.concatenate([c + rng.normal(scale=0.2, size=(n_per, 2)) for c in centers])
    labels = [f"c{i}" for i in range(n_cls) for _ in range(n_per)]
    return proj, labels


def _rot(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def test_orient_proj_undoes_rotation_and_reflection() -> None:
    proj, labels = _labelled_layout()
    _, ref = orient_proj(proj, labels)  # first eval bootstraps the canonical frame
    # the same layout, rotated AND mirrored, must come back to the bootstrapped frame
    mangled = (proj @ _rot(1.1)) * np.array([-1.0, 1.0])
    oriented, _ = orient_proj(mangled, labels, ref, tau=1.0)
    base, _ = orient_proj(proj, labels, ref, tau=1.0)
    assert np.allclose(oriented, base, atol=1e-6)


def test_orient_pca_undoes_sign_flips_without_rotating() -> None:
    proj, labels = _labelled_layout()
    _, ref = orient_pca(proj, labels)
    for flip in ([-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]):
        oriented, _ = orient_pca(proj * np.array(flip), labels, ref, tau=1.0)
        assert np.allclose(oriented, proj, atol=1e-6)  # flip removed, layout otherwise untouched


def test_orient_pca_leaves_rotation_alone() -> None:
    # PCA's axes are meaningful: a rotated layout must NOT be rotated back (only signs are in play),
    # which is what separates orient_pca from orient_proj.
    proj, labels = _labelled_layout()
    _, ref = orient_pca(proj, labels)
    rotated = proj @ _rot(0.7)
    oriented, _ = orient_pca(rotated, labels, ref, tau=1.0)
    assert np.allclose(np.abs(oriented), np.abs(rotated), atol=1e-6)
    assert not np.allclose(oriented, proj, atol=1e-2)


def test_orient_dispatches_rigid_for_tsne_umap_and_sign_only_for_pca() -> None:
    proj, labels = _labelled_layout()
    rotated = proj @ _rot(0.9)
    refs = {m: _orient(m, proj, labels, None, 1.0)[1] for m in ("PCA", "t-SNE", "UMAP")}
    for method in ("t-SNE", "UMAP"):  # rigid: the rotation is undone -> lands where unrotated proj lands
        oriented, _ = _orient(method, rotated, labels, refs[method], 1.0)
        expected, _ = _orient(method, proj, labels, refs[method], 1.0)
        assert np.allclose(oriented, expected, atol=1e-6), method
    oriented, _ = _orient("PCA", rotated, labels, refs["PCA"], 1.0)  # sign-only: the rotation survives
    expected, _ = _orient("PCA", proj, labels, refs["PCA"], 1.0)
    assert not np.allclose(oriented, expected, atol=1e-2)


def test_orient_is_idempotent_on_an_already_aligned_layout() -> None:
    proj, labels = _labelled_layout()
    for method in ("PCA", "t-SNE", "UMAP"):
        oriented, ref = _orient(method, proj, labels, None, 1.0)
        again, _ = _orient(method, oriented, labels, ref, 1.0)
        assert np.allclose(again, oriented, atol=1e-6), method


def _sphere_layout(seed=0):
    """Clustered unit vectors with strictly ranked class sizes: c0 is the pole anchor, c1 the meridian one."""
    rng = np.random.default_rng(seed)
    sizes = [60, 40, 25, 15]
    centers = rng.normal(size=(4, 3))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    xyz = np.concatenate([c + rng.normal(scale=0.12, size=(n, 3)) for c, n in zip(centers, sizes)])
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
    return xyz, [f"c{i}" for i, n in enumerate(sizes) for _ in range(n)]


def _sphere_azimuth(o, labels, cls):
    v = o[np.asarray(labels) == cls].mean(axis=0)
    return math.degrees(math.atan2(v[1], v[0]))


def _sphere_polar(o, labels, cls):
    v = o[np.asarray(labels) == cls].mean(axis=0)
    return math.degrees(math.acos(v[2] / np.linalg.norm(v)))


def test_orient_sphere_bootstrap_anchors_pole_and_central_meridian() -> None:
    xyz, labels = _sphere_layout()
    o, _ = orient_sphere(xyz, labels)
    assert _sphere_polar(o, labels, "c0") < 1e-9          # largest class exactly at the north pole
    assert abs(_sphere_azimuth(o, labels, "c1")) < 1e-9   # second exactly on the central meridian


def test_orient_sphere_recovers_rotations_and_reflections() -> None:
    # aligned evals are one SO(3) Procrustes to the reference -- the minimal rigid motion of the whole
    # ball, with nothing pinned after the bootstrap. So an arbitrarily rotated (or mirrored) copy of the
    # SAME layout must come back exactly to the reference frame: zero genuine layout change means zero
    # residual motion.
    xyz, labels = _sphere_layout()
    o0, ref = orient_sphere(xyz, labels)
    rng = np.random.default_rng(3)
    for _ in range(4):
        Q = np.linalg.qr(rng.normal(size=(3, 3)))[0]  # orthogonal; det may be -1 (a mirror) -- both recover
        o, _ = orient_sphere(xyz @ Q, labels, dict(ref), tau=0.5)
        assert np.allclose(o, o0, atol=1e-6)


def test_orient_sphere_moves_only_the_reorganized_cluster() -> None:
    # minimal-rotation gliding: when ONE cluster genuinely moves, the alignment must not spin the whole
    # ball to re-pin an anchor on it. Weighted Procrustes does split the move between the frame and the
    # cluster (the frame chases it in proportion to its weight), so the robust claim is the ordering:
    # the reorganized cluster displaces MORE than every untouched one, and no untouched cluster is
    # dragged more than half the perturbation.
    xyz, labels = _sphere_layout()
    o0, ref = orient_sphere(xyz, labels)
    lab = np.asarray(labels)
    a = math.radians(40.0)
    spin = np.array([[math.cos(a), math.sin(a), 0.0], [-math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
    pert = o0.copy()
    pert[lab == "c1"] = pert[lab == "c1"] @ spin  # one cluster reorganizes by 40 deg
    rng = np.random.default_rng(4)
    o, _ = orient_sphere(pert @ np.linalg.qr(rng.normal(size=(3, 3)))[0], labels, dict(ref), tau=0.5)

    def com_shift(c):
        u = o[lab == c].mean(0); u = u / np.linalg.norm(u)
        v = o0[lab == c].mean(0); v = v / np.linalg.norm(v)
        return math.degrees(math.acos(min(1.0, max(-1.0, float(u @ v)))))

    untouched = {c: com_shift(c) for c in ("c0", "c2", "c3")}
    moved = com_shift("c1")
    assert all(d < 20.0 for d in untouched.values()), untouched  # dragged less than half the 40 deg move
    assert moved > max(untouched.values())                       # the reorganized cluster glides the most
