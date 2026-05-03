"""
python -m pytest tests/unit/test_eval_metrics.py
"""

import pytest  # type: ignore[import]
import torch

import utils.eval as eval_utils
from utils.eval import compute_map_img2img, compute_map_cross_modal
from utils.head import compute_sim


def _manual_map_img2img(
    embs_q: torch.Tensor,
    class_encs_q: torch.Tensor,
    embs_g: torch.Tensor,
    class_encs_g: torch.Tensor,
) -> float:
    sim = compute_sim(embs_q, embs_g, "cos")
    ap_vals = []

    for i in range(embs_q.size(0)):
        idxs = torch.argsort(sim[i], descending=True)
        idxs = idxs[idxs != i]

        rel = (class_encs_g[idxs] == class_encs_q[i]).float()
        n_pos = int(rel.sum().item())
        if n_pos == 0:
            continue

        ranks = torch.arange(1, rel.numel() + 1, dtype=torch.float32)
        prec = rel.cumsum(0) / ranks
        ap = (prec * rel).sum().item() / n_pos
        ap_vals.append(ap)

    return float(sum(ap_vals) / len(ap_vals))


def _stub_distributed_collectives(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eval_utils.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(eval_utils.dist, "all_reduce", lambda *args, **kwargs: None)


def test_compute_map_img2img_toy_matches_expected_cosine_value(monkeypatch) -> None:
    monkeypatch.setattr(eval_utils.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(eval_utils.dist, "all_reduce", lambda *args, **kwargs: None)

    embs = torch.tensor([
        [1.30, 0.01, 1.00],
        [-1.10, 0.01, 1.00],
        [-1.30, 0.01, 1.00],
        [1.10, 0.01, 1.00],
        [1.00, 0.01, 1.00],
        [-1.00, 0.01, 1.00],
    ], dtype=torch.float32)
    embs = torch.nn.functional.normalize(embs, p=2, dim=1)
    class_encs = torch.tensor([0, 1, 1, 0, 1, 0], dtype=torch.long)

    scores = compute_map_img2img(
        embs_q=embs,
        class_encs_q=class_encs,
        embs_g=embs,
        class_encs_g=class_encs,
    )

    assert scores["map"] == pytest.approx(0.5805555555, abs=1e-6)


def test_compute_map_img2img_expanded_gallery_keeps_only_self_excluded(monkeypatch) -> None:
    monkeypatch.setattr(eval_utils.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(eval_utils.dist, "all_reduce", lambda *args, **kwargs: None)

    embs_q = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ], dtype=torch.float32)
    embs_q = torch.nn.functional.normalize(embs_q, p=2, dim=1)

    embs_g_extra = torch.tensor([
        [0.95, 0.05],
        [0.05, 0.95],
    ], dtype=torch.float32)
    embs_g_extra = torch.nn.functional.normalize(embs_g_extra, p=2, dim=1)

    embs_g = torch.cat([embs_q, embs_g_extra], dim=0)
    class_encs_q = torch.tensor([0, 1, 0], dtype=torch.long)
    class_encs_g = torch.tensor([0, 1, 0, 0, 1], dtype=torch.long)

    scores = compute_map_img2img(
        embs_q=embs_q,
        class_encs_q=class_encs_q,
        embs_g=embs_g,
        class_encs_g=class_encs_g,
    )
    expected = _manual_map_img2img(embs_q, class_encs_q, embs_g, class_encs_g)

    assert scores["map"] == pytest.approx(expected, abs=1e-6)


@pytest.mark.gpu
def test_compute_map_img2img_toy(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_distributed_collectives(monkeypatch)

    device = torch.device("cuda")
    embs = torch.tensor([
        [1.30, 0.01, 1.00],
        [-1.10, 0.01, 1.00],
        [-1.30, 0.01, 1.00],
        [1.10, 0.01, 1.00],
        [1.00, 0.01, 1.00],
        [-1.00, 0.01, 1.00],
    ], dtype=torch.float32, device=device)
    embs = torch.nn.functional.normalize(embs, p=2, dim=1)
    class_encs = torch.tensor([0, 1, 1, 0, 1, 0], dtype=torch.long, device=device)

    scores = compute_map_img2img(
        embs_q=embs,
        class_encs_q=class_encs,
        embs_g=embs,
        class_encs_g=class_encs,
    )

    assert scores["map"] == pytest.approx(0.5805555555, abs=1e-6)
    assert scores["macro_map"] == pytest.approx(0.5805555555, abs=1e-6)


@pytest.mark.gpu
def test_compute_map_img2img_singleton_queries_are_excluded(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_distributed_collectives(monkeypatch)

    device = torch.device("cuda")
    embs = torch.tensor([
        [1.30, 0.01, 1.00],
        [-1.10, 0.01, 1.00],
        [-1.30, 0.01, 1.00],
        [1.10, 0.01, 1.00],
        [1.00, 0.01, 1.00],
        [-1.00, 0.01, 1.00],
        [0.01, 1.00, -1.00],
    ], dtype=torch.float32, device=device)
    embs = torch.nn.functional.normalize(embs, p=2, dim=1)
    class_encs = torch.tensor([0, 1, 1, 0, 1, 0, 2], dtype=torch.long, device=device)

    scores = compute_map_img2img(
        embs_q=embs,
        class_encs_q=class_encs,
        embs_g=embs,
        class_encs_g=class_encs,
    )

    assert scores["map"] == pytest.approx(0.5805555555, abs=1e-6)
    assert scores["macro_map"] == pytest.approx(0.5805555555, abs=1e-6)


@pytest.mark.gpu
def test_compute_map_img2img_extra_gallery_singletons(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_distributed_collectives(monkeypatch)

    device = torch.device("cuda")
    embs_q = torch.tensor([
        [1.30, 0.01, 1.00],
        [-1.10, 0.01, 1.00],
        [-1.30, 0.01, 1.00],
        [1.10, 0.01, 1.00],
        [1.00, 0.01, 1.00],
        [-1.00, 0.01, 1.00],
        [-1.00, 0.11, 1.00],
    ], dtype=torch.float32, device=device)
    embs_q = torch.nn.functional.normalize(embs_q, p=2, dim=1)
    class_encs_q = torch.tensor([0, 1, 1, 0, 1, 0, 2], dtype=torch.long, device=device)

    embs_g = torch.tensor([
        [1.30, 0.01, 1.00],
        [-1.10, 0.01, 1.00],
        [-1.30, 0.01, 1.00],
        [1.10, 0.01, 1.00],
        [1.00, 0.01, 1.00],
        [-1.00, 0.01, 1.00],
        [-1.00, 0.11, 1.00],
        [-1.00, 0.01, 1.20],
        [-1.00, 0.51, 1.00],
        [-1.50, 0.01, 1.00],
        [-1.00, 0.91, 1.00],
    ], dtype=torch.float32, device=device)
    embs_g = torch.nn.functional.normalize(embs_g, p=2, dim=1)
    class_encs_g = torch.tensor([0, 1, 1, 0, 1, 0, 2, 3, 4, 5, 6], dtype=torch.long, device=device)

    scores = compute_map_img2img(
        embs_q=embs_q,
        class_encs_q=class_encs_q,
        embs_g=embs_g,
        class_encs_g=class_encs_g,
    )

    assert scores["map"] == pytest.approx(0.3524801587301587, abs=1e-6)
    assert scores["macro_map"] == pytest.approx(0.3524801587301587, abs=1e-6)


@pytest.mark.gpu
def test_compute_map_cross_modal_extra_gallery_singletons(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_distributed_collectives(monkeypatch)

    device = torch.device("cuda")
    embs_q = torch.tensor([
        [ 1.30, 0.01, 1.00],
        [-1.10, 0.01, 1.00],
        [-1.30, 0.01, 1.00],
        [ 1.10, 0.01, 1.00],
        [ 1.00, 0.01, 1.00],
        [-1.00, 0.01, 1.00],
        [-1.00, 0.11, 1.00],
    ], dtype=torch.float32, device=device)
    embs_q = torch.nn.functional.normalize(embs_q, p=2, dim=1)
    class_encs_q = torch.tensor([0, 1, 1, 0, 1, 0, 2], dtype=torch.long, device=device)

    embs_g = torch.tensor([
        [ 1.30, 0.01, 1.00],
        [-1.10, 0.01, 1.00],
        [-1.30, 0.01, 1.00],
        [ 1.10, 0.01, 1.00],
        [ 1.00, 0.01, 1.00],
        [-1.00, 0.01, 1.00],
        [-1.00, 0.11, 1.00],
        [-1.00, 0.01, 1.20],
        [-1.00, 0.51, 1.00],
        [-1.50, 0.01, 1.00],
        [-1.00, 0.91, 1.00],
    ], dtype=torch.float32, device=device)
    embs_g = torch.nn.functional.normalize(embs_g, p=2, dim=1)
    class_encs_g = torch.tensor([0, 1, 1, 0, 1, 0, 2, 3, 4, 5, 6], dtype=torch.long, device=device)

    scores = compute_map_cross_modal(
        embs_q=embs_q,
        class_encs_q=class_encs_q,
        embs_g=embs_g,
        class_encs_g=class_encs_g,
    )

    assert scores["map"] == pytest.approx(0.6862674362674362, abs=1e-6)
    assert scores["macro_map"] == pytest.approx(0.7559857837635615, abs=1e-6)


@pytest.mark.gpu
def test_compute_acc_macro_acc_cross_modal_extra_gallery_singletons(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_distributed_collectives(monkeypatch)

    device = torch.device("cuda")
    embs_q = torch.tensor([
        [ 1.30, 0.01, 1.00],
        [-1.10, 0.01, 1.00],
        [-1.30, 0.01, 1.00],
        [ 1.10, 0.01, 1.00],
        [ 1.00, 0.01, 1.00],
        [-1.00, 0.01, 1.00],
        [-1.00, 0.11, 1.00],
    ], dtype=torch.float32, device=device)
    embs_q = torch.nn.functional.normalize(embs_q, p=2, dim=1)
    class_encs_q = torch.tensor([0, 1, 1, 0, 1, 0, 2], dtype=torch.long, device=device)

    embs_g = torch.tensor([
        [ 1.30, 0.01, 1.00],
        [-1.10, 0.01, 1.00],
        [-1.30, 0.01, 1.00],
        [ 1.00, 0.01, 1.00],
        [-1.00, 0.01, 1.00],
        [-1.00, 0.11, 1.00],
        [-1.00, 0.01, 1.20],
        [-1.00, 0.51, 1.00],
        [ 1.10, 0.01, 1.00],
        [-1.50, 0.01, 1.00],
        [-1.00, 0.91, 1.00],
    ], dtype=torch.float32, device=device)
    embs_g = torch.nn.functional.normalize(embs_g, p=2, dim=1)
    class_encs_g = torch.tensor([0, 1, 1, 0, 1, 0, 2, 3, 4, 5, 6], dtype=torch.long, device=device)

    scores = compute_map_cross_modal(
        embs_q=embs_q,
        class_encs_q=class_encs_q,
        embs_g=embs_g,
        class_encs_g=class_encs_g,
    )

    print(scores["acc"])
    print(scores["macro_acc"])
    assert scores["acc"] == pytest.approx(0.42857142857142855, abs=1e-6)
    assert scores["macro_acc"] == pytest.approx(0.3333333333333333, abs=1e-6)


@pytest.mark.gpu
def test_compute_map_cross_modal_without_accuracy(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_distributed_collectives(monkeypatch)

    device = torch.device("cuda")
    embs_q = torch.tensor([
        [1.30, 0.01, 1.00],
        [-1.10, 0.01, 1.00],
    ], dtype=torch.float32, device=device)
    embs_q = torch.nn.functional.normalize(embs_q, p=2, dim=1)
    class_encs_q = torch.tensor([0, 1], dtype=torch.long, device=device)

    embs_g = torch.tensor([
        [1.30, 0.01, 1.00],
        [-1.10, 0.01, 1.00],
        [-1.30, 0.01, 1.00],
    ], dtype=torch.float32, device=device)
    embs_g = torch.nn.functional.normalize(embs_g, p=2, dim=1)
    class_encs_g = torch.tensor([0, 1, 1], dtype=torch.long, device=device)

    scores = compute_map_cross_modal(
        embs_q=embs_q,
        class_encs_q=class_encs_q,
        embs_g=embs_g,
        class_encs_g=class_encs_g,
        compute_accuracy=False,
    )

    assert "acc" not in scores
    assert "accs" not in scores
    assert "macro_acc" not in scores
    assert scores["map"] == pytest.approx(1.0, abs=1e-6)