import pytest
import torch

from utils.head import compute_sim


@pytest.fixture
def orthonormal_embs() -> tuple[torch.Tensor, torch.Tensor]:
    embs = torch.eye(2, dtype=torch.float32)
    return embs, embs.clone()


def test_compute_sim_cos_matches_dot_product(orthonormal_embs: tuple[torch.Tensor, torch.Tensor]) -> None:
    embs_img, embs_txt = orthonormal_embs

    sim = compute_sim(embs_img, embs_txt, "cos")

    assert torch.equal(sim, torch.eye(2))


def test_compute_sim_geo_maps_to_expected_range(
    orthonormal_embs: tuple[torch.Tensor, torch.Tensor],
) -> None:
    embs_img, embs_txt = orthonormal_embs

    sim_geo = compute_sim(embs_img, embs_txt, "geo")

    assert torch.all(sim_geo <= 1.0)
    assert torch.all(sim_geo >= -1.0)
    assert torch.allclose(torch.diag(sim_geo), torch.ones(2), atol=1e-3)  # the eps guard caps the sim at ~0.9991
    assert torch.allclose(sim_geo[0, 1], torch.tensor(0.0), atol=1e-5)