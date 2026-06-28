"""
python -m campaign_runner
"""

from pathlib import Path
from copy import deepcopy
import ctypes
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import traceback
import time
import psutil
import torch

from utils.config import apply_overrides, apply_train_debug_overrides, load_train_config_dict, load_manifold_viz_config_dict
from utils.utils import paths, save_pickle, save_json, load_json


CAMPAIGN = "dev2"

SEED0 = 42
NUM_SEEDS = 3

# DATASETS = ("bryo", "cub", "lepid", "nymph")
DATASETS = ("nymph", "lepid", "cub", "bryo")
# DATASETS = ("cub", "bryo", "lepid", "nymph")
# DATASETS = ("cub", "nymph")
# DATASETS = ("lepid",)

BASELINE_OVERRIDES = [
    # {"batch_size": 32_000, "name": "way-too-big-bs"},
    {"loss2.mix": 0.3, "loss2.targ": "phylo", "name": "hp"},
    # {"loss.targ": "aligned", "name": "iw"},
    # {"loss.targ": "multipos", "name": "sw"},
]


def _relevant_stderr(stderr: str) -> str:
    """Drop pre-crash noise (eval progress bars, warnings) by keeping the
    captured stderr from the first Python traceback onward. tqdm and warnings
    are emitted while the trial runs, so the first 'Traceback (most recent call
    last):' marks where the relevant error content begins. Falls back to the
    full text when no traceback is present (e.g. a bare SIGKILL)."""
    marker = "Traceback (most recent call last):"
    idx = stderr.find(marker)
    return stderr if idx == -1 else stderr[idx:]

def _log_trial_error(dpath_trial: Path, idx_trial: int, n_trials: int, seed: int, dataset: str, setting: str, exc: Exception) -> None:
    """Log trial error to stdout and to error.log in the trial-seed's directory."""
    dpath_trial.mkdir(parents=True, exist_ok=True)
    fpath_error = dpath_trial / "error.log"
    # Format error message with context
    error_msg = (
        f"\n[{idx_trial}/{n_trials}] TRIAL FAILED\n"
        f"  seed={seed}, dataset={dataset}, setting={setting}"
    )
    stderr_body = None
    if isinstance(exc, subprocess.CalledProcessError):
        stderr = getattr(exc, "stderr", None)
        if stderr:
            stderr_body = _relevant_stderr(stderr)
    # Print to stdout
    print(error_msg, flush=True)
    # Write to error log file
    with open(fpath_error, "w") as f:
        f.write(error_msg + "\n")
        if stderr_body is not None:
            f.write("--- stderr ---\n")
            f.write(stderr_body + "\n")
        else:
            f.write(traceback.format_exc())

def _dpath_campaign() -> Path:
    return paths["artifacts"] / CAMPAIGN

def _load_or_create_baseline_config() -> dict:
    fpath = _dpath_campaign() / "cfg_baseline.json"
    if fpath.exists():
        return load_json(fpath)

    cfg_baseline = load_train_config_dict()
    cfg_baseline = apply_train_debug_overrides(cfg_baseline)
    for key in ("setting", "seed", "dataset"):
        cfg_baseline.pop(key, None)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    save_json(cfg_baseline, fpath)
    return cfg_baseline

def _load_or_create_manifold_viz_config() -> dict:
    fpath = _dpath_campaign() / "cfg_manifold_viz.json"
    if fpath.exists():
        return load_json(fpath)

    cfg_manifold_viz = load_manifold_viz_config_dict()
    fpath.parent.mkdir(parents=True, exist_ok=True)
    save_json(cfg_manifold_viz, fpath)
    return cfg_manifold_viz

def _expand_settings(settings_raw: list[dict]) -> list[tuple[str, dict]]:
    settings = []
    seen_names: set[str] = set()
    for item in settings_raw:
        if "name" not in item:
            raise ValueError("Each BASELINE_OVERRIDES item must include a 'name' field.")

        name = item["name"]
        if name in seen_names:
            raise ValueError(f"Duplicate BASELINE_OVERRIDES name: {name}")
        seen_names.add(name)
        payload = {k: deepcopy(v) for k, v in item.items() if k != "name"}
        settings.append((name, payload))
    return settings

def _write_setting_overrides(setting: str, normalized_overrides: dict) -> None:
    fpath = _dpath_campaign() / setting / "overrides.json"
    fpath.parent.mkdir(parents=True, exist_ok=True)
    with open(fpath, "w") as f:
        json.dump(normalized_overrides, f, indent=2, sort_keys=True)

def _iter_seeds() -> list[int]:
    return list(range(SEED0, SEED0 + NUM_SEEDS))

def _enable_child_subreaper() -> None:
    """Become the reaper for orphaned descendants. torch elastic starts each
    rank in its own session, so when a rank is SIGKILLed (e.g. OOM) its
    DataLoader workers orphan to init and escape any process-group kill from
    here. As a subreaper we inherit them instead, so _reap_subtree can find
    and kill them."""
    PR_SET_CHILD_SUBREAPER = 36
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)
    except OSError:
        pass

def _reap_subtree(grace: float = 10.0, spare_root: int | None = None) -> None:
    """SIGKILL and reap every descendant process (torchrun, ranks, DataLoader
    workers). Relies on _enable_child_subreaper so orphaned workers reparent
    here and show up as descendants. A background render worker (spare_root and
    its subtree) is left alone so it can keep rendering through the next trial."""
    parent = psutil.Process()
    spare: set[int] = set()
    if spare_root is not None:
        try:
            rp = psutil.Process(spare_root)
            spare = {rp.pid, *(c.pid for c in rp.children(recursive=True))}
        except psutil.NoSuchProcess:
            pass
    deadline = time.time() + grace
    while True:
        procs = [p for p in parent.children(recursive=True) if p.pid not in spare]
        if not procs:
            return
        for p in procs:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(procs, timeout=2)
        if time.time() >= deadline:
            return

def _run_trial_subprocess(cfg_dict: dict, spare_render_pid: int | None = None) -> None:
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc-per-node=auto",
        "-m",
        "campaign_trial_runner",
        "--cfg-json",
        json.dumps(cfg_dict),
    ]

    # start_new_session isolates torchrun from the terminal's Ctrl-C so the
    # campaign drives teardown itself (via _reap_subtree) rather than racing
    # torchrun's own signal handling.
    proc = subprocess.Popen(
        cmd,
        stdout=None,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    # Drain stderr in a thread: when a rank is SIGKILLed (e.g. OOM), its
    # DataLoader workers are orphaned but keep the stderr pipe's write end open,
    # so a read-until-EOF loop in this process would hang forever even after
    # torchrun itself exits.
    stderr_chunks: list[bytes] = []
    def _drain() -> None:
        assert proc.stderr is not None
        while chunk := proc.stderr.read1(4096):
            sys.stderr.buffer.write(chunk)
            sys.stderr.buffer.flush()
            stderr_chunks.append(chunk)

    reader = threading.Thread(target=_drain, daemon=True)
    reader.start()

    try:
        return_code = proc.wait()
    finally:
        # Tear down the entire descendant subtree on any exit from the wait.
        # Covers normal crash recovery (orphaned DataLoader workers left by a
        # SIGKILLed rank, which would otherwise keep leaking into the cgroup
        # memory budget of later trials) and Ctrl-C / SIGTERM of the campaign
        # (the detached trial would otherwise keep running). A process-group
        # kill is insufficient: elastic puts each rank in its own session.
        _reap_subtree(spare_root=spare_render_pid)

    # Bounded: don't re-hang if a worker is wedged in uninterruptible sleep and
    # still holding the pipe; the daemon thread is torn down at interpreter exit.
    reader.join(timeout=30)

    if return_code != 0:
        stderr_data = b"".join(stderr_chunks)
        stderr_body = "\n".join(stderr_data.decode(errors="replace").splitlines())
        raise subprocess.CalledProcessError(return_code, cmd, stderr=stderr_body)

def _check_trial_completion(dpath_trial: Path) -> bool:
    fpath_metadata_trial = dpath_trial / "trial_metadata.json"
    if not fpath_metadata_trial.exists():
        complete = False
    else:
        metadata_trial = load_json(fpath_metadata_trial)
        complete = metadata_trial["complete"]
    return complete

def _mark_trial_complete(dpath_trial: Path) -> None:
    # marked only after a clean subprocess exit + cleanup, so a mid-finalization crash never leaves a
    # trial falsely flagged complete (which would make a re-run skip it instead of resuming)
    fpath_metadata_trial = dpath_trial / "trial_metadata.json"
    metadata_trial = load_json(fpath_metadata_trial)
    metadata_trial["complete"] = True
    save_json(metadata_trial, fpath_metadata_trial)

def _spawn_render(trial_rel: str, render_evo: bool) -> subprocess.Popen:
    """Spawn the post-trial manifold-viz render as a detached, CPU-only process so it overlaps the next
    trial's training. It renders purely from the trial's cached projections.npz (no GPU/DDP), using the
    campaign's frozen config snapshot. CUDA_VISIBLE_DEVICES is cleared so it never contends for the GPUs
    the training subprocess needs."""
    cmd = [sys.executable, "-m", "tools.manifold_viz", trial_rel, "snapshot"]
    if not render_evo:
        cmd.append("no_evo")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    return subprocess.Popen(cmd, env=env, start_new_session=True)

def _raise_interrupt(signum, frame) -> None:
    raise KeyboardInterrupt

def run_campaign() -> None:
    _enable_child_subreaper()
    # Route SIGTERM (e.g. `kill`, SLURM scancel) through the same path as Ctrl-C
    # so the trial's subtree is torn down before the campaign exits.
    signal.signal(signal.SIGTERM, _raise_interrupt)

    time_data = {
        "last_updated": time.time(),
        "elapsed": 0.0,
    }
    dpath_campaign = _dpath_campaign()
    dpath_campaign.mkdir(parents=True, exist_ok=True)
    save_pickle(time_data, dpath_campaign / "time.pkl")

    n_gpus = torch.cuda.device_count()
    fpath_meta = dpath_campaign / "campaign_metadata.json"
    if fpath_meta.exists():
        existing_meta = load_json(fpath_meta)
        if existing_meta["n_gpus"] != n_gpus:
            raise RuntimeError(
                f"GPU count mismatch: campaign '{CAMPAIGN}' was run with "
                f"{existing_meta['n_gpus']} GPUs but current environment has {n_gpus}."
            )
    else:
        metadata_camp = {
            "duration": "0-00:00:00",
            "n_gpus": n_gpus,
        }
        save_json(metadata_camp, dpath_campaign / "campaign_metadata.json")

    cfg_baseline = _load_or_create_baseline_config()
    cfg_manifold_viz = _load_or_create_manifold_viz_config()
    settings = _expand_settings(BASELINE_OVERRIDES)
    seeds = _iter_seeds()

    for setting, setting_payload in settings:
        _write_setting_overrides(setting, setting_payload)

    n_trials = len(seeds) * len(DATASETS) * len(settings)
    print(f"Campaign: '{CAMPAIGN}' ({n_trials} trials)")

    render_proc: subprocess.Popen | None = None
    render_evo = cfg_baseline["dev"]["traintime_evals"]  # evolution GIFs need mid-evals to evolve across

    idx_trial = 0
    for idx_seed, seed in enumerate(seeds):
        for dataset in DATASETS:
            for setting, setting_payload in settings:
                idx_trial += 1

                dpath_trial = _dpath_campaign() / setting / dataset / str(seed)
                if _check_trial_completion(dpath_trial):
                    print(f"[{idx_trial}/{n_trials}] SKIP (completed): seed={seed} dataset={dataset} setting={setting}")
                    continue

                cfg_dict = deepcopy(cfg_baseline)
                cfg_dict["campaign"] = CAMPAIGN
                cfg_dict["setting"] = setting
                cfg_dict["seed"] = seed
                cfg_dict["dataset"] = dataset
                cfg_dict["standalone"] = False
                cfg_dict["idx_seed"] = idx_seed
                cfg_dict["manifold_viz"] = cfg_manifold_viz
                cfg_dict["_setting_overrides"] = setting_payload
                cfg_dict = apply_overrides(cfg_dict, setting_payload)

                if dpath_trial.exists():
                    print(f"[{idx_trial}/{n_trials}] RESUME: seed={seed} dataset={dataset} setting={setting}")
                else:
                    print(f"[{idx_trial}/{n_trials}] seed={seed} dataset={dataset} setting={setting}")

                spare_pid = render_proc.pid if render_proc is not None and render_proc.poll() is None else None
                try:
                    _run_trial_subprocess(cfg_dict, spare_render_pid=spare_pid)
                    shutil.rmtree(dpath_trial / "chkpts/in_progress")
                    _mark_trial_complete(dpath_trial)
                except KeyboardInterrupt:
                    print(
                        f"\n[{idx_trial}/{n_trials}] INTERRUPTED — terminated trial process group; exiting campaign.",
                        flush=True,
                    )
                    if render_proc is not None and render_proc.poll() is None:
                        render_proc.terminate()
                    return
                except Exception as e:
                    _log_trial_error(
                        dpath_trial=dpath_trial,
                        idx_trial=idx_trial,
                        n_trials=n_trials,
                        seed=seed,
                        dataset=dataset,
                        setting=setting,
                        exc=e,
                    )
                    continue

                # Render this trial's manifold viz off-process (CPU-only), overlapping the next trial's
                # training. At most one render in flight: wait on the prior one first (near-instant in
                # practice, since a trial far outlasts a render).
                if render_proc is not None and render_proc.poll() is None:
                    render_proc.wait()
                render_proc = _spawn_render(f"{CAMPAIGN}/{setting}/{dataset}/{seed}", render_evo)

    # let the last trial's render finish before the campaign exits
    if render_proc is not None:
        try:
            render_proc.wait()
        except KeyboardInterrupt:
            render_proc.terminate()


if __name__ == "__main__":
    run_campaign()