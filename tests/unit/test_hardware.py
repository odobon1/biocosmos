import time

import pytest

from utils import hardware
from utils.hardware import read_cgroup_ram


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
