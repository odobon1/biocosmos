"""
A trial's resume checkpoints, <trial>/chkpts/in_progress/, as GENERATIONS: every checkpoint is written into a
directory of its own beside the last one and becomes the resume point only when a manifest is published into it.

    chkpts/in_progress/
        gen-000007/                 the committed generation: the resume point
            train_state.pt          model / optimizer / LR schedule / counters   (rank 0)
            trial_state.pkl         the trial's recorded series + eval count + timer   (rank 0)
            rng_state_rank<k>.pt    one per rank, each written by its own rank
            manifest.json           written LAST (commit): what the generation holds, each file with its size
        gen-000008/                 the next one, mid-write: no manifest yet, so not a checkpoint

So a checkpoint never overwrites the only valid copy: a kill (or a full disk) anywhere in a write leaves the last
committed generation untouched, and what a resume loads is always ONE generation -- never one save point's weights
beside another's RNG state. Every file lands by atomic rename (utils.utils.atomic_write), so a file under its own
name is whole, and the manifest's rename is the commit. Retention is bounded: committing a generation deletes
everything else in the dir (commit), so the dir holds one checkpoint, two while the next is being written.

Generations are found by listing the dir, never by stat-ing a path: on a shared filesystem (NFS) a client can keep
answering "no such file" for a path created since it last looked, while a listing reflects the server.
"""

from pathlib import Path
import os
import re
import shutil

from utils.utils import load_json, save_json


MANIFEST = "manifest.json"


class ChkptError(RuntimeError):
    """A trial's checkpoint state can't be trusted as it stands: nothing is deleted, the trial is not restarted."""


def dpath_gen(dpath_chkpt: Path, gen: int) -> Path:
    return dpath_chkpt / f"gen-{gen:06d}"

def _entries(dpath_chkpt: Path) -> list[str]:
    """The names in `dpath_chkpt`; [] when the dir is not there."""
    try:
        return os.listdir(dpath_chkpt)
    except FileNotFoundError:
        return []

def _committed(dpath_chkpt: Path) -> list[int]:
    """The committed generations (those holding a manifest), ascending."""
    gens = [int(m.group(1)) for name in _entries(dpath_chkpt) if (m := re.fullmatch(r"gen-(\d+)", name))]
    return sorted(gen for gen in gens if MANIFEST in os.listdir(dpath_gen(dpath_chkpt, gen)))

def commit(dpath_chkpt: Path, gen: int, fnames: list[str], n_samps_seen: int) -> None:
    """Publish generation `gen` as the trial's resume point, then delete everything else in `dpath_chkpt` (older
    generations, an abandoned half-written one). `fnames` are the files that make the
    generation up -- every one must already be on disk (a missing one raises: an incomplete generation is never
    published) -- and the manifest records each with its size, for resume_point to check them against. It also
    carries the generation's `n_samps_seen`, the training position the runner's progress check reads (progress).
    Called once every writer is done (train.TrainPipeline._checkpoint: after the barrier)."""
    dpath = dpath_gen(dpath_chkpt, gen)
    manifest = {"gen": gen, "n_samps_seen": n_samps_seen,
                "files": {fname: os.stat(dpath / fname).st_size for fname in fnames}}
    save_json(manifest, dpath / MANIFEST)
    for name in os.listdir(dpath_chkpt):
        if name != dpath.name:
            path = dpath_chkpt / name
            shutil.rmtree(path) if path.is_dir() else path.unlink()

def resume_point(dpath_chkpt: Path) -> tuple[Path, int] | None:
    """(the dir a resume loads from, its generation number), or None when `dpath_chkpt` holds no checkpoint: the
    newest committed generation, checked against its manifest. A committed generation whose files don't match
    its manifest raises ChkptError rather than being skipped: falling back from it (to an older generation, or to
    "no checkpoint") would quietly discard training that may only be unreadable right now. A listing that fails
    (I/O error, stale handle) raises as it is, for the same reason."""
    gens = _committed(dpath_chkpt)
    if not gens:
        return None
    dpath = dpath_gen(dpath_chkpt, gens[-1])
    present = {entry.name: entry.stat().st_size for entry in os.scandir(dpath)}
    for fname, size in load_json(dpath / MANIFEST)["files"].items():
        if present.get(fname) != size:
            raise ChkptError(
                f"checkpoint {dpath} does not match its manifest ({fname}: expected {size} bytes, found "
                f"{present.get(fname)}) -- left as it is; the trial is not restarted."
            )
    return dpath, gens[-1]

def progress(dpath_chkpt: Path) -> int:
    """The training position (n_samps_seen) of the trial's newest committed generation, -1 with none. What the
    runner compares across a trial attempt to tell whether it got anywhere (campaign_runner._run_phase)."""
    gens = _committed(dpath_chkpt)
    return load_json(dpath_gen(dpath_chkpt, gens[-1]) / MANIFEST)["n_samps_seen"] if gens else -1
