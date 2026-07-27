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

@pytest.mark.integration
def test_dv_sampler_chains_permutations_with_pass_offsets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("utils.data.dist.get_world_size", lambda: 1)
    monkeypatch.setattr("utils.data.dist.get_rank", lambda: 0)
    monkeypatch.setattr("utils.data.shuffle_list", lambda values, seed: list(values))

    sampler = DorsalVentralBatchSampler(
        index_pos=["dorsal", "dorsal", "ventral", "ventral"],
        batch_size=2,
        seed=0,
        n_perms=3,
    )

    batches = list(iter(sampler))

    # 3 chained passes per category (n_samples=4): raw [0,1]/[2,3] with pass offsets 0, 4, 8
    assert len(sampler) == 6
    assert batches == [[0, 1], [4, 5], [8, 9], [2, 3], [6, 7], [10, 11]]


@pytest.mark.integration
def test_dv_sampler_chained_passes_shuffle_independently(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("utils.data.dist.get_world_size", lambda: 1)
    monkeypatch.setattr("utils.data.dist.get_rank", lambda: 0)

    n_d = 50
    index_pos = ["dorsal"] * n_d + ["ventral"] * n_d
    n_perms = 3
    sampler = DorsalVentralBatchSampler(index_pos=index_pos, batch_size=n_d, seed=11, n_perms=n_perms)
    n_samples = len(index_pos)

    batches = list(iter(sampler))

    assert len(batches) == 2 * n_perms
    idxs_d = set(range(n_d))
    blocks_d = []
    for batch in batches:
        raws = [i % n_samples for i in batch]
        # homogeneous position per batch, each batch a full permutation of its category
        assert set(raws) in (idxs_d, {i + n_d for i in idxs_d})
        # one pass per batch: uniform pass offset within the batch
        assert len({i // n_samples for i in batch}) == 1
        if set(raws) == idxs_d:
            blocks_d.append(raws)
    assert len(blocks_d) == n_perms
    assert blocks_d[0] != blocks_d[1] and blocks_d[0] != blocks_d[2] and blocks_d[1] != blocks_d[2]


@pytest.mark.integration
def test_dv_sampler_single_perm_keeps_epoch_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("utils.data.dist.get_world_size", lambda: 1)
    monkeypatch.setattr("utils.data.dist.get_rank", lambda: 0)
    monkeypatch.setattr("utils.data.shuffle_list", lambda values, seed: list(values))

    sampler = DorsalVentralBatchSampler(
        index_pos=["dorsal", "dorsal", "ventral", "ventral"],
        batch_size=2,
        seed=0,
    )
    sampler.set_epoch(2)

    batches = list(iter(sampler))

    assert batches == [[8, 9], [10, 11]]  # epoch * n_samples = 8


@pytest.mark.integration
def test_dv_sampler_rejects_zero_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("utils.data.dist.get_world_size", lambda: 1)
    monkeypatch.setattr("utils.data.dist.get_rank", lambda: 0)

    with pytest.raises(ValueError, match="zero dorsal/ventral batches"):
        DorsalVentralBatchSampler(
            index_pos=["dorsal", "dorsal", "ventral", "ventral"],
            batch_size=8,
            seed=0,
        )


@pytest.mark.integration
def test_dv_sampler_chained_ranks_reassemble_contiguous_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("utils.data.dist.get_world_size", lambda: 2)
    monkeypatch.setattr("utils.data.shuffle_list", lambda values, seed: list(values))

    index_pos = ["dorsal"] * 4
    batches_by_rank = []
    for rank in range(2):
        monkeypatch.setattr("utils.data.dist.get_rank", lambda rank=rank: rank)
        sampler = DorsalVentralBatchSampler(index_pos=index_pos, batch_size=4, seed=0, n_perms=2)
        batches_by_rank.append(list(iter(sampler)))

    # chained dorsal stream is [0,1,2,3, 4,5,6,7]; strided rank slicing means each global batch
    # (rank subbatches side by side) reassembles a contiguous window of the chain
    assert batches_by_rank[0] == [[0, 2], [4, 6]]
    assert batches_by_rank[1] == [[1, 3], [5, 7]]


@pytest.mark.integration
def test_dv_sampler_chained_epoch_advances_pass_offsets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("utils.data.dist.get_world_size", lambda: 1)
    monkeypatch.setattr("utils.data.dist.get_rank", lambda: 0)
    monkeypatch.setattr("utils.data.shuffle_list", lambda values, seed: list(values))

    sampler = DorsalVentralBatchSampler(
        index_pos=["dorsal", "dorsal", "ventral", "ventral"],
        batch_size=2,
        seed=0,
        n_perms=3,
    )
    sampler.set_epoch(2)

    batches = list(iter(sampler))

    # epoch 2 passes are 6, 7, 8 -> offsets 24, 28, 32 (n_samples=4); raw d=[0,1], v=[2,3]
    assert batches == [[24, 25], [28, 29], [32, 33], [26, 27], [30, 31], [34, 35]]
