import time

import pytest

from utils import hardware
from utils.hardware import _auto_n_workers, _tres_mem_gib, read_cgroup_ram


def test_ram_peak_tracker_retains_spike(monkeypatch):
    try:
        used_base, _ = read_cgroup_ram()
    except RuntimeError:
        pytest.skip("no memory-limited cgroup (not running under a SLURM job)")

    tracker = hardware._RamPeakTracker(interval=0.02)
    tracker.start()
    monkeypatch.setattr(hardware, "_ram_peak_tracker", tracker)

    spike = b"\x01" * (300 * 2**20)  # writes every byte, so all pages are charged to the cgroup
    time.sleep(0.3)  # several polls while the spike is live
    del spike
    time.sleep(0.1)

    # the polling thread saw the spike, and read_cgroup_ram folds it in after the free
    assert tracker.peak >= used_base + 200 * 2**20
    used_peak, _ = read_cgroup_ram()
    assert used_peak >= tracker.peak


def test_tres_mem_gib_suffixes():
    assert _tres_mem_gib("128G") == 128.0
    assert _tres_mem_gib("512000M") == 512000 / 2**10
    assert _tres_mem_gib("2T") == 2048.0
    assert _tres_mem_gib("1024") == 1.0  # bare number is MiB
    assert _tres_mem_gib("0") == 0.0


def test_auto_n_workers_cpu_bound_at_small_batch():
    # tiny batches: RAM bound is huge, the CPU reserve rule decides
    assert _auto_n_workers(n_cpus=16, n_gpus=2, ram_gib=128, batch_size=512, prefetch_factor=2, model_type="siglip_vitb16") == 6
    assert _auto_n_workers(n_cpus=32, n_gpus=4, ram_gib=512, batch_size=512, prefetch_factor=2, model_type="siglip_vitb16") == 6


def test_auto_n_workers_ram_bound_at_large_batch():
    # 64k global on 2 GPUs, 128 GiB, prefetch 1: sub-batch = 32768 uint8 samples (~4.9 GB),
    # worker cost = 3 sub-batches (~14.8 GB) x2 for the train+eval loader overlap; the 0.85
    # budget less the main processes' 3 sub-batches each fits 1 worker/GPU -- the CPU
    # bound (6) would OOM
    assert _auto_n_workers(n_cpus=16, n_gpus=2, ram_gib=128, batch_size=65_536, prefetch_factor=1, model_type="siglip_vitb16") == 1
    # doubling the alloc's RAM lifts the RAM bound
    assert _auto_n_workers(n_cpus=16, n_gpus=2, ram_gib=256, batch_size=65_536, prefetch_factor=1, model_type="siglip_vitb16") == 3
    # the verified 16k config stays CPU-bound: RAM bound (7) > CPU bound (6)
    assert _auto_n_workers(n_cpus=16, n_gpus=2, ram_gib=128, batch_size=16_384, prefetch_factor=1, model_type="siglip_vitb16") == 6


def test_auto_n_workers_scales_with_model_resolution():
    # 384px samples are ~2.94x the 224px bytes: the same 16k config drops from 6 workers to 2
    assert _auto_n_workers(n_cpus=16, n_gpus=2, ram_gib=128, batch_size=16_384, prefetch_factor=1, model_type="siglip_vitb16_384") == 2
    # trailing _<res> is the marker; other digits in the name don't parse as a resolution
    assert _auto_n_workers(n_cpus=16, n_gpus=2, ram_gib=128, batch_size=16_384, prefetch_factor=1, model_type="siglip_vitso400m14") == 6


def test_auto_n_workers_floors_at_one():
    assert _auto_n_workers(n_cpus=16, n_gpus=2, ram_gib=128, batch_size=1_048_576, prefetch_factor=1, model_type="siglip_vitb16") == 1
