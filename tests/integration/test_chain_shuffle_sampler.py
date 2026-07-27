import pytest

from utils.data import ChainShuffleDistributedSampler


@pytest.mark.integration
def test_chain_shuffle_blocks_are_full_permutations() -> None:
    n = 50
    n_perms = 3

    sampler = ChainShuffleDistributedSampler(list(range(n)), n_perms=n_perms, num_replicas=1, rank=0)
    idxs = list(iter(sampler))

    assert len(idxs) == len(sampler) == n * n_perms
    blocks = []
    for p in range(n_perms):
        block = idxs[p * n:(p + 1) * n]
        assert all(idx // n == p for idx in block)  # pass-encoded offset
        assert sorted(idx % n for idx in block) == list(range(n))  # each block covers the dataset once
        blocks.append([idx % n for idx in block])
    # permutations within an epoch are independently shuffled, not one shuffle repeated
    assert blocks[0] != blocks[1] and blocks[0] != blocks[2] and blocks[1] != blocks[2]


@pytest.mark.integration
def test_chain_shuffle_ranks_partition_the_chain() -> None:
    n = 10
    n_perms = 4

    r0 = list(iter(ChainShuffleDistributedSampler(list(range(n)), n_perms=n_perms, num_replicas=2, rank=0)))
    r1 = list(iter(ChainShuffleDistributedSampler(list(range(n)), n_perms=n_perms, num_replicas=2, rank=1)))

    assert len(r0) == len(r1) == n * n_perms // 2
    assert not set(r0) & set(r1)
    # together the ranks cover every (pass, item) pair exactly once
    assert sorted(r0 + r1) == sorted(range(n * n_perms))
    # ranks take strided (interleaved) halves of one chained stream, so each global batch of
    # world_size sub-batches reassembles a contiguous window of the chain
    full = list(iter(ChainShuffleDistributedSampler(list(range(n)), n_perms=n_perms, num_replicas=1, rank=0)))
    assert r0 == full[0::2]
    assert r1 == full[1::2]


@pytest.mark.integration
def test_chain_shuffle_drops_tail_on_uneven_split() -> None:
    n = 7
    n_perms = 3  # 21 chained samples across 2 ranks -> 10 per rank, 1 dropped

    r0 = list(iter(ChainShuffleDistributedSampler(list(range(n)), n_perms=n_perms, num_replicas=2, rank=0)))
    r1 = list(iter(ChainShuffleDistributedSampler(list(range(n)), n_perms=n_perms, num_replicas=2, rank=1)))

    assert len(r0) == len(r1) == 10
    assert not set(r0) & set(r1)


@pytest.mark.integration
def test_chain_shuffle_epochs_deterministic_and_distinct() -> None:
    n = 50
    n_perms = 2

    sampler = ChainShuffleDistributedSampler(list(range(n)), n_perms=n_perms, num_replicas=1, rank=0)
    epoch0 = list(iter(sampler))
    epoch0_again = list(iter(sampler))
    sampler.set_epoch(1)
    epoch1 = list(iter(sampler))

    assert epoch0 == epoch0_again
    assert [idx % n for idx in epoch0] != [idx % n for idx in epoch1]
    # the pass counter keeps advancing across epochs so augmentation seeds never repeat
    assert all(idx // n in (2, 3) for idx in epoch1)
