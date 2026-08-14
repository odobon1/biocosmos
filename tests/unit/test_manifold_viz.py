import math

import numpy as np
import torch

from utils.manifold_viz import _hbeta_search, _knn, _sparse_joint_p, _tsne_torch


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
    Y = _tsne_torch(X, init, perplexity=30.0, n_iter=300, device="cpu")
    assert Y.shape == (180, 2) and np.isfinite(Y).all()
    lab = np.repeat([0, 1, 2], 60)
    cents = np.stack([Y[lab == i].mean(0) for i in range(3)])
    intra = max(np.linalg.norm(Y[lab == i] - cents[i], axis=1).mean() for i in range(3))
    inter = min(np.linalg.norm(cents[i] - cents[j]) for i in range(3) for j in range(i + 1, 3))
    assert inter > 3 * intra, f"clusters not separated: inter={inter:.2f} intra={intra:.2f}"


def test_tsne_is_deterministic() -> None:
    X = _blob(150)
    init = (X[:, :2] / X[:, 0].std() * 1e-4).numpy().astype(np.float32)
    Y1 = _tsne_torch(X, init, perplexity=15.0, n_iter=60, device="cpu")
    Y2 = _tsne_torch(X, init, perplexity=15.0, n_iter=60, device="cpu")
    assert np.array_equal(Y1, Y2)
