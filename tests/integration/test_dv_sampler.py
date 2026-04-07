from __future__ import annotations

import pytest

from utils.data import DorsalVentralBatchSampler


@pytest.mark.integration
def test_dv_sampler_yields_homogeneous_subbatches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("utils.data.dist.get_world_size", lambda: 2)
    monkeypatch.setattr("utils.data.dist.get_rank", lambda: 0)
    monkeypatch.setattr("utils.data.shuffle_list", lambda values, seed: list(values))

    sampler = DorsalVentralBatchSampler(
        index_pos=["dorsal", "dorsal", "dorsal", "dorsal", "ventral", "ventral", "ventral", "ventral"],
        batch_size=4,
        seed=11,
    )

    batches = list(iter(sampler))

    assert len(sampler) == 2
    assert batches == [[0, 1], [4, 5]]


@pytest.mark.integration
def test_dv_sampler_changes_with_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("utils.data.dist.get_world_size", lambda: 1)
    monkeypatch.setattr("utils.data.dist.get_rank", lambda: 0)
    monkeypatch.setattr(
        "utils.data.shuffle_list",
        lambda values, seed: list(reversed(values)) if seed % 2 else list(values),
    )

    sampler = DorsalVentralBatchSampler(
        index_pos=["dorsal", "dorsal", "ventral", "ventral"],
        batch_size=2,
        seed=5,
    )

    first_epoch = list(iter(sampler))
    sampler.set_epoch(1)
    second_epoch = list(iter(sampler))

    assert first_epoch != second_epoch
