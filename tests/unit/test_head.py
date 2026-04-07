from __future__ import annotations

import pytest  # type: ignore[import]
import torch  # type: ignore[import]

from utils.head import compute_sim


@pytest.fixture
def orthonormal_embs() -> tuple[torch.Tensor, torch.Tensor]:
    embs = torch.eye(2, dtype=torch.float32)
    return embs, embs.clone()


def test_compute_sim_cos_matches_dot_product(orthonormal_embs: tuple[torch.Tensor, torch.Tensor]) -> None:
    embs_img, embs_txt = orthonormal_embs

    sim = compute_sim(embs_img, embs_txt, "cos")

    assert torch.equal(sim, torch.eye(2))


def test_compute_sim_geo_variants_map_to_expected_range(
    orthonormal_embs: tuple[torch.Tensor, torch.Tensor],
) -> None:
    embs_img, embs_txt = orthonormal_embs

    sim_geo1 = compute_sim(embs_img, embs_txt, "geo1")
    sim_geo2 = compute_sim(embs_img, embs_txt, "geo2")

    assert torch.all(sim_geo1 <= 1.0)
    assert torch.all(sim_geo1 >= -1.0)
    assert torch.all(sim_geo2 <= 1.0)
    assert torch.all(sim_geo2 >= -1.0)
    assert torch.allclose(torch.diag(sim_geo1), torch.ones(2))
    assert torch.allclose(torch.diag(sim_geo2), torch.ones(2), atol=1e-5)
    assert torch.allclose(sim_geo1[0, 1], torch.tensor(0.0))
    assert torch.allclose(sim_geo2[0, 1], torch.tensor(0.0), atol=1e-5)
