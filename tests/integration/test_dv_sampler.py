import pytest

from utils.data import DorsalVentralBatchSampler, ExactDistributedSampler


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


@pytest.mark.integration
def test_exact_distributed_sampler_assigns_exact_uneven_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("utils.data.dist.get_world_size", lambda: 2)

    dataset = list(range(47_295))

    monkeypatch.setattr("utils.data.dist.get_rank", lambda: 0)
    sampler_rank0 = ExactDistributedSampler(dataset, shuffle=False)
    idxs_rank0 = list(iter(sampler_rank0))

    monkeypatch.setattr("utils.data.dist.get_rank", lambda: 1)
    sampler_rank1 = ExactDistributedSampler(dataset, shuffle=False)
    idxs_rank1 = list(iter(sampler_rank1))

    assert len(sampler_rank0) == 23_648
    assert len(sampler_rank1) == 23_647
    assert len(idxs_rank0) == 23_648
    assert len(idxs_rank1) == 23_647
    assert set(idxs_rank0).isdisjoint(idxs_rank1)
    assert set(idxs_rank0) | set(idxs_rank1) == set(dataset)


@pytest.mark.integration
def test_exact_distributed_sampler_changes_with_epoch_when_shuffling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("utils.data.dist.get_world_size", lambda: 4)
    monkeypatch.setattr("utils.data.dist.get_rank", lambda: 2)
    monkeypatch.setattr(
        "utils.data.shuffle_list",
        lambda values, seed: list(values) if seed % 2 == 0 else list(reversed(values)),
    )

    sampler = ExactDistributedSampler(list(range(9)), shuffle=True, seed=7)

    first_epoch = list(iter(sampler))
    sampler.set_epoch(1)
    second_epoch = list(iter(sampler))

    assert first_epoch != second_epoch