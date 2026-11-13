import errno
import json
import multiprocessing
import os
import platform

import numpy as np
import pytest

from utils.utils import atomic_write, load_pickle, save_json, save_npz, save_pickle, shuffle_list, tmp_fpath, update_json
from utils.log import PrintLog
from utils.timing import RunningMean
from utils.text import get_text_template


def test_running_mean_tracks_average() -> None:
    mean = RunningMean()

    for value in [2.0, 4.0, 6.0, 8.0]:
        mean.update(value)

    assert mean.n == 4
    assert mean.value() == 5.0


def test_shuffle_list_is_seeded_and_non_mutating() -> None:
    values = [1, 2, 3, 4, 5]

    shuffled_a = shuffle_list(values, seed=17)
    shuffled_b = shuffle_list(values, seed=17)

    assert shuffled_a == shuffled_b
    assert values == [1, 2, 3, 4, 5]
    assert shuffled_a != values


def test_get_text_template_returns_known_templates() -> None:
    train_template = get_text_template("train")
    lepid_template = get_text_template("train", dataset="lepid")
    bioclip_template = get_text_template("sci", dataset="cub")

    assert isinstance(train_template, list)
    assert train_template[0] == ["", "a photo of "]
    assert lepid_template[-1] == ["", " butterfly"]
    assert bioclip_template == [["a photo of $SCI$"]]


def test_printlog_eval_handles_missing_loss_key() -> None:
    class _EvalPipe:
        partitions = ["id"]

    eval_metrics = {
        "scores": {
            "native": {
                "id": {"map": {"i2t": 0.1, "i2i": 0.2, "t2i": 0.3}, "acc": {"i2t": 0.4}},
                "comp": {"map": {"all": 0.2, "i2i": 0.2, "id": 0.2}},
            },
            "joint_macro": {
                "id": {"map": {"i2t": 0.13, "i2i": 0.23, "t2i": 0.33}, "acc": {"i2t": 0.43}},
                "comp": {"map": {"all": 0.23, "i2i": 0.23, "id": 0.23}},
            },
        },
        "loss_raw": {"id": None},
        "sim_targ": {"id": None, "comb": None},
    }

    PrintLog.eval(eval_metrics, _EvalPipe(), {"native": "Standard", "joint_macro": "GZSL"})


def test_printlog_eval_prints_the_combined_sim_targ_point_stats(capsys) -> None:
    # the Sim/Targ block reads the point stats over every partition's chunks together (sim_targ's comb)
    class _EvalPipe:
        partitions = ["id"]

    comb = {f"{prefix}_{stat}": val for prefix in ("sim", "targ")
            for stat, val in (("min", -0.5), ("max", 0.9), ("median", 0.1), ("mean", 0.12))}
    eval_metrics = {
        "scores": {"native": {"comp": {"map": {"all": 0.2, "i2i": 0.2, "id": 0.2}}}},
        "loss_raw": {"id": 0.7},
        "sim_targ": {"id": comb, "comb": comb},
    }

    PrintLog.eval(eval_metrics, _EvalPipe(), {"native": "Standard"})

    assert "min=-0.5000 max= 0.9000 med= 0.1000 mean= 0.1200" in capsys.readouterr().out


def test_printlog_eval_blocks_follow_the_eval_groups_in_play(capsys) -> None:
    # one composite block per group in play, under its reported name; a group switched off has no
    # scores subtree at all, so a stale block would KeyError rather than print blank
    class _EvalPipe:
        partitions = ["id"]

    eval_metrics = {
        "scores": {
            "native": {"comp": {"map": {"all": 0.2, "i2i": 0.2, "id": 0.2}}},
            "joint_macro": {"comp": {"map": {"all": 0.23, "i2i": 0.23, "id": 0.23}}},
        },
        "loss_raw": {"id": None},
        "sim_targ": {"id": None, "comb": None},
    }

    PrintLog.eval(eval_metrics, _EvalPipe(), {"native": "Standard", "joint_macro": "GZSL"})

    printout = capsys.readouterr().out
    assert "Standard Composite mAP" in printout
    assert "GZSL Composite mAP" in printout
    assert "Native" not in printout and "Joint" not in printout


def test_save_npz_round_trips_and_leaves_no_tmp(tmp_path) -> None:
    fpath = tmp_path / "projections.npz"

    save_npz(fpath, a=np.arange(3), b=np.ones((2, 2)))

    with np.load(fpath) as npz:
        assert npz["a"].tolist() == [0, 1, 2]
        assert npz["b"].shape == (2, 2)
    assert list(tmp_path.iterdir()) == [fpath]  # the tmp file was renamed away, not left behind


def test_save_npz_killed_mid_write_leaves_the_existing_file_intact(tmp_path, monkeypatch) -> None:
    # the atomicity save_npz exists for: a render worker terminated mid-rewrite (campaign runner replan,
    # watchdog cancellation) must leave the previously-good cache in place, never a torn npz
    fpath = tmp_path / "projections.npz"
    save_npz(fpath, a=np.arange(3))

    def torn_savez(f, **arrays):
        f.write(b"PK\x03\x04 torn")  # a few bytes land, then the kill
        raise KeyboardInterrupt

    monkeypatch.setattr("utils.utils.np.savez", torn_savez)
    with pytest.raises(KeyboardInterrupt):
        save_npz(fpath, a=np.arange(5))

    with np.load(fpath) as npz:
        assert npz["a"].tolist() == [0, 1, 2]  # the pre-rewrite cache, readable


def test_tmp_fpath_is_unique_across_nodes_and_processes(tmp_path) -> None:
    # the temp name an atomic write goes through carries the node as well as the pid: two writers of one target on
    # different nodes of a shared filesystem can share a pid, and must not share a temp file
    fpath = tmp_path / "phase_metadata.json"
    assert tmp_fpath(fpath) == tmp_path / f"phase_metadata.json.{platform.node()}.{os.getpid()}.tmp"


def test_atomic_write_publishes_only_a_finished_file(tmp_path) -> None:
    fpath = tmp_path / "state.bin"
    fpath.write_bytes(b"old")

    with atomic_write(fpath) as f:
        f.write(b"new")
        assert fpath.read_bytes() == b"old"  # nothing is published until the write is whole

    assert fpath.read_bytes() == b"new"
    assert list(tmp_path.iterdir()) == [fpath]  # the temp file was renamed into place


@pytest.mark.parametrize("fail_at", ["write", "fsync", "replace"])
def test_atomic_write_that_fails_leaves_the_old_file_and_no_temp(tmp_path, monkeypatch, fail_at) -> None:
    # a full disk can surface at any step of the write (on NFS, as late as the fsync): whichever it is, the target
    # is still the old file -- the only valid copy is never the one being written -- and the partial temp is gone
    fpath = tmp_path / "state.bin"
    fpath.write_bytes(b"old")

    def enospc(*args, **kwargs):
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

    if fail_at != "write":
        monkeypatch.setattr(f"utils.utils.os.{fail_at}", enospc)
    with pytest.raises(OSError) as exc:
        with atomic_write(fpath) as f:
            f.write(b"new, but only part of it")
            if fail_at == "write":
                enospc()

    assert exc.value.errno == errno.ENOSPC
    monkeypatch.undo()
    assert fpath.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [fpath]


def test_save_pickle_killed_mid_write_leaves_the_existing_file_intact(tmp_path, monkeypatch) -> None:
    # save_pickle publishes atomically too: a trial killed while rewriting data_trial.pkl leaves the last whole one
    fpath = tmp_path / "data_trial.pkl"
    save_pickle({"n": 1}, fpath)

    def torn_dump(obj, f):
        f.write(b"\x80\x04 torn")
        raise KeyboardInterrupt

    monkeypatch.setattr("utils.utils.pickle.dump", torn_dump)
    with pytest.raises(KeyboardInterrupt):
        save_pickle({"n": 2}, fpath)

    monkeypatch.undo()
    assert load_pickle(fpath) == {"n": 1}
    assert list(tmp_path.iterdir()) == [fpath]


def test_update_json_writes_back_only_on_a_clean_exit(tmp_path) -> None:
    fpath = tmp_path / "phase_metadata.json"
    save_json({"n_crashes": {"other": 0}, "duration": "0-00:00:00"}, fpath)

    with update_json(fpath) as metadata:
        metadata["n_crashes"]["other"] += 1
    assert json.loads(fpath.read_text()) == {"n_crashes": {"other": 1}, "duration": "0-00:00:00"}

    with pytest.raises(RuntimeError):
        with update_json(fpath) as metadata:
            metadata["n_crashes"]["other"] += 1
            raise RuntimeError("the update did not finish")
    assert json.loads(fpath.read_text())["n_crashes"] == {"other": 1}  # a half-made update is not written


def _bump(fpath, key, n) -> None:
    for _ in range(n):
        with update_json(fpath) as metadata:
            metadata[key] += 1


def test_update_json_serializes_concurrent_writers(tmp_path) -> None:
    # the whole read-modify-write is one critical section: several processes updating one file at once -- two
    # of them the same counter, the others their own keys -- lose nothing. Without the lock, a writer's save
    # lands over the file another has read and not yet written back, and that other's update is gone.
    fpath = tmp_path / "phase_metadata.json"
    save_json({"shared": 0, "a": 0, "b": 0}, fpath)
    n = 150
    ctx = multiprocessing.get_context("fork")
    procs = [ctx.Process(target=_bump, args=(fpath, key, n)) for key in ("shared", "shared", "a", "b")]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()

    assert [proc.exitcode for proc in procs] == [0, 0, 0, 0]
    assert json.loads(fpath.read_text()) == {"shared": 2 * n, "a": n, "b": n}
    assert sorted(p.name for p in tmp_path.iterdir()) == [".lock", "phase_metadata.json"]  # no temp file left behind
