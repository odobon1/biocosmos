import errno
import json
import os

import pytest

from utils import chkpt
from utils.utils import atomic_write


FNAMES = ["train_state.pt", "trial_state.pkl", "rng_state_rank0.pt", "rng_state_rank1.pt"]  # a 2-rank trial's generation


def _write_gen(dpath_chkpt, gen, fnames=FNAMES):
    """Write generation `gen`'s files (each tagged with its generation), uncommitted."""
    dpath_gen = chkpt.dpath_gen(dpath_chkpt, gen)
    dpath_gen.mkdir(parents=True, exist_ok=True)
    for fname in fnames:
        with atomic_write(dpath_gen / fname, "w") as f:
            f.write(f"{fname}@{gen}")
    return dpath_gen


def _commit_gen(dpath_chkpt, gen, n_samps_seen=None):
    dpath_gen = _write_gen(dpath_chkpt, gen)
    chkpt.commit(dpath_chkpt, gen, FNAMES, n_samps_seen=100 * gen if n_samps_seen is None else n_samps_seen)
    return dpath_gen


def _loaded(dpath_resume):
    """What a resume from `dpath_resume` reads: every state file's content."""
    return {fname: (dpath_resume / fname).read_text() for fname in FNAMES}


def test_no_checkpoint(tmp_path) -> None:
    assert chkpt.resume_point(tmp_path / "chkpts" / "in_progress") is None  # no dir at all
    (tmp_path / "chkpts" / "in_progress").mkdir(parents=True)
    assert chkpt.resume_point(tmp_path / "chkpts" / "in_progress") is None  # an empty one
    assert chkpt.progress(tmp_path / "chkpts" / "in_progress") == -1


def test_commit_publishes_the_generation_and_drops_the_last(tmp_path) -> None:
    _commit_gen(tmp_path, 1)
    assert chkpt.resume_point(tmp_path) == (tmp_path / "gen-000001", 1)

    dpath_gen = _commit_gen(tmp_path, 2)

    assert chkpt.resume_point(tmp_path) == (dpath_gen, 2)
    assert _loaded(dpath_gen) == {fname: f"{fname}@2" for fname in FNAMES}
    assert [p.name for p in tmp_path.iterdir()] == ["gen-000002"]  # retention is bounded: one committed checkpoint
    manifest = json.loads((dpath_gen / "manifest.json").read_text())
    assert manifest == {"gen": 2, "n_samps_seen": 200, "files": {fname: len(f"{fname}@2") for fname in FNAMES}}


def test_interrupted_write_leaves_the_committed_checkpoint_as_the_resume_point(tmp_path) -> None:
    # killed partway through the next checkpoint -- rank 0's state down, a rank's RNG state not yet, a temp file
    # mid-write, no manifest: the generation is not a checkpoint, and the resume is the last committed one, whole
    dpath_committed = _commit_gen(tmp_path, 1)
    dpath_partial = _write_gen(tmp_path, 2, fnames=["train_state.pt", "trial_state.pkl", "rng_state_rank0.pt"])
    (dpath_partial / "rng_state_rank1.pt.node.123.tmp").write_text("half a fi")

    assert chkpt.resume_point(tmp_path) == (dpath_committed, 1)
    assert _loaded(dpath_committed) == {fname: f"{fname}@1" for fname in FNAMES}  # one save point, never a mixture
    assert chkpt.progress(tmp_path) == 100  # the abandoned write is no progress


def test_interrupted_write_is_cleared_by_the_next_commit(tmp_path) -> None:
    # the resumed trial reaches the same generation number again: its files replace the abandoned ones, the
    # manifest covers exactly the files of this attempt, and the commit sweeps out what the dead one left
    _commit_gen(tmp_path, 1)
    dpath_partial = _write_gen(tmp_path, 2, fnames=["train_state.pt"])
    (dpath_partial / "train_state.pt").write_text("from the attempt that died")
    (dpath_partial / "trial_state.pkl.node.123.tmp").write_text("half a fi")
    _write_gen(tmp_path, 3, fnames=["train_state.pt"])  # ...and one it left further on

    dpath_gen = _commit_gen(tmp_path, 2)

    assert chkpt.resume_point(tmp_path) == (dpath_gen, 2)
    assert _loaded(dpath_gen) == {fname: f"{fname}@2" for fname in FNAMES}
    assert [p.name for p in tmp_path.iterdir()] == ["gen-000002"]


def test_commit_refuses_a_generation_missing_a_rank(tmp_path) -> None:
    # a generation is published only once EVERY component is on disk -- each rank's RNG state included: with one
    # missing there is no commit, nothing is deleted, and the last committed checkpoint stands
    dpath_committed = _commit_gen(tmp_path, 1)
    _write_gen(tmp_path, 2, fnames=["train_state.pt", "trial_state.pkl", "rng_state_rank0.pt"])

    with pytest.raises(FileNotFoundError):
        chkpt.commit(tmp_path, 2, FNAMES, n_samps_seen=200)

    assert not (tmp_path / "gen-000002" / "manifest.json").exists()
    assert chkpt.resume_point(tmp_path) == (dpath_committed, 1)


def test_failed_component_write_never_reaches_the_committed_checkpoint(tmp_path, monkeypatch) -> None:
    # the disk fills while a component of the next generation is written: the write raises with the errno intact,
    # leaves no partial file, and the committed generation -- written elsewhere -- is untouched
    dpath_committed = _commit_gen(tmp_path, 1)
    dpath_gen = chkpt.dpath_gen(tmp_path, 2)
    dpath_gen.mkdir()

    def enospc(fd):
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

    monkeypatch.setattr("utils.utils.os.fsync", enospc)
    with pytest.raises(OSError) as exc:
        with atomic_write(dpath_gen / "train_state.pt", "w") as f:
            f.write("train_state.pt@2")
    monkeypatch.undo()

    assert exc.value.errno == errno.ENOSPC
    assert list(dpath_gen.iterdir()) == []
    assert chkpt.resume_point(tmp_path) == (dpath_committed, 1)
    assert _loaded(dpath_committed) == {fname: f"{fname}@1" for fname in FNAMES}


def test_committed_generation_that_fails_its_manifest_is_reported_not_skipped(tmp_path) -> None:
    # a committed checkpoint with a file missing or cut short is uncertain state: it raises -- no fall back to an
    # older generation, no "no checkpoint" (which would restart the trial) -- and nothing is deleted
    dpath_gen = _commit_gen(tmp_path, 2)
    dpath_older = _write_gen(tmp_path, 1)  # an older, whole generation beside it (a commit never leaves one, but even so)
    (dpath_older / "manifest.json").write_text(json.dumps({"gen": 1, "n_samps_seen": 100, "files": {}}))

    (dpath_gen / "rng_state_rank1.pt").write_text("cut sh")
    with pytest.raises(chkpt.ChkptError, match="rng_state_rank1.pt"):
        chkpt.resume_point(tmp_path)

    (dpath_gen / "rng_state_rank1.pt").unlink()
    with pytest.raises(chkpt.ChkptError, match="rng_state_rank1.pt"):
        chkpt.resume_point(tmp_path)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["gen-000001", "gen-000002"]  # both left exactly as found
    assert sorted(p.name for p in dpath_gen.iterdir()) == ["manifest.json", "rng_state_rank0.pt", "train_state.pt", "trial_state.pkl"]


def test_unreadable_checkpoint_dir_is_an_error_not_an_absent_checkpoint(tmp_path, monkeypatch) -> None:
    # a listing that fails (I/O error, stale NFS handle) says nothing about whether a checkpoint exists: it must
    # raise, never read as "no checkpoint"
    _commit_gen(tmp_path, 1)

    def eio(path):
        raise OSError(errno.EIO, os.strerror(errno.EIO))

    monkeypatch.setattr("utils.chkpt.os.listdir", eio)
    with pytest.raises(OSError) as exc:
        chkpt.resume_point(tmp_path)
    assert exc.value.errno == errno.EIO
    with pytest.raises(OSError):
        chkpt.progress(tmp_path)


def test_progress_is_the_committed_training_position(tmp_path) -> None:
    # the runner's retry loop counts an attempt as progress when the committed checkpoint moved FORWARD: a
    # checkpoint re-committed at the position it was at (a trial that keeps dying after its last one) is none
    _commit_gen(tmp_path, 1, n_samps_seen=0)  # the base eval's checkpoint
    assert chkpt.progress(tmp_path) == 0
    _commit_gen(tmp_path, 2, n_samps_seen=4_096)
    assert chkpt.progress(tmp_path) == 4_096
    _commit_gen(tmp_path, 3, n_samps_seen=4_096)
    assert chkpt.progress(tmp_path) == 4_096
