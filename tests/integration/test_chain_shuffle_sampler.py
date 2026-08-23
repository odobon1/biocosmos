import pytest

from utils.data import ChainShuffleDistributedSampler


@pytest.mark.integration
def test_chain_shuffle_windows_tile_one_continuous_stream() -> None:
    n = 50
    samps_per_pass = 140  # not a multiple of n: windows open and close mid-permutation

    sampler = ChainShuffleDistributedSampler(list(range(n)), samps_per_pass=samps_per_pass, num_replicas=1, rank=0)
    epoch0 = list(iter(sampler))
    sampler.set_epoch(1)
    epoch1 = list(iter(sampler))

    assert len(epoch0) == len(epoch1) == len(sampler) == samps_per_pass
    # the 10 samples of perm 2 chopped off by epoch 0's window lead epoch 1 instead of being dropped
    assert [idx // n for idx in epoch0[-40:]] == [2] * 40
    assert [idx // n for idx in epoch1[:10]] == [2] * 10
    stream = epoch0 + epoch1  # consecutive epochs tile the stream with nothing dropped between
    perms = []
    for b in range(len(stream) // n):
        block = stream[b * n:(b + 1) * n]
        assert all(idx // n == b for idx in block)  # stream-encoded offset
        assert sorted(idx % n for idx in block) == list(range(n))  # each aligned block covers the dataset once
        perms.append(tuple(idx % n for idx in block))
    assert len(set(perms)) == len(perms)  # permutations independently shuffled, not one shuffle repeated


@pytest.mark.integration
def test_chain_shuffle_ranks_partition_the_window() -> None:
    n = 10
    samps_per_pass = 40

    r0 = list(iter(ChainShuffleDistributedSampler(list(range(n)), samps_per_pass=samps_per_pass, num_replicas=2, rank=0)))
    r1 = list(iter(ChainShuffleDistributedSampler(list(range(n)), samps_per_pass=samps_per_pass, num_replicas=2, rank=1)))

    assert len(r0) == len(r1) == samps_per_pass // 2
    assert not set(r0) & set(r1)
    # together the ranks cover every (perm, item) pair of the window exactly once
    assert sorted(r0 + r1) == list(range(samps_per_pass))
    # ranks take strided (interleaved) halves of the stream window, so each global batch of
    # world_size sub-batches reassembles a contiguous window of the stream
    full = list(iter(ChainShuffleDistributedSampler(list(range(n)), samps_per_pass=samps_per_pass, num_replicas=1, rank=0)))
    assert r0 == full[0::2]
    assert r1 == full[1::2]


@pytest.mark.integration
def test_chain_shuffle_wraps_whole_permutations_when_batches_exceed_them() -> None:
    # batch_size > len(dataset): windows span several permutations, and an epoch boundary can wrap
    # more than a permutation's worth of samples to the front of the next window
    n = 10
    samps_per_pass = 25

    sampler = ChainShuffleDistributedSampler(list(range(n)), samps_per_pass=samps_per_pass, num_replicas=1, rank=0)
    epoch0 = list(iter(sampler))
    sampler.set_epoch(1)
    epoch1 = list(iter(sampler))

    assert sorted(epoch0 + epoch1) == list(range(50))  # perms 0-4 consumed exactly once, nothing dropped
    assert [idx // n for idx in epoch1] == [2] * 5 + [3] * 10 + [4] * 10  # perm 2 finishes before perm 3 starts


@pytest.mark.integration
def test_chain_shuffle_epochs_deterministic_and_disjoint() -> None:
    n = 50
    samps_per_pass = 100

    sampler = ChainShuffleDistributedSampler(list(range(n)), samps_per_pass=samps_per_pass, num_replicas=1, rank=0)
    epoch0 = list(iter(sampler))
    epoch0_again = list(iter(sampler))
    sampler.set_epoch(1)
    epoch1 = list(iter(sampler))

    assert epoch0 == epoch0_again
    # disjoint stream windows: encoded indexes never repeat across epochs, so augmentation seeds don't either
    assert not set(epoch0) & set(epoch1)
    assert [idx % n for idx in epoch0] != [idx % n for idx in epoch1]
