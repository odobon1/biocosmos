"""
python -m campaign_runner
"""

from pathlib import Path
from copy import deepcopy
import ctypes
import json
import shutil
import signal
import subprocess
import sys
import threading
import traceback
import time
import psutil
import torch

from utils.config import apply_overrides, apply_train_debug_overrides, load_train_config_dict
from utils.utils import paths, save_pickle, save_json, load_json


CAMPAIGN = "dev"

SEED0 = 42
NUM_SEEDS = 1

# DATASETS = ("bryo", "cub", "lepid", "nymph")
# DATASETS = ("nymph", "lepid", "cub", "bryo")
DATASETS = ("nymph",)

BASELINE_OVERRIDES = [
    {"batch_size": 32_000, "name": "way-too-big-bs"},
    {"loss2.mix": 0.3, "loss2.targ": "phylo", "name": "hp"},
    {"loss.targ": "aligned", "name": "iw"},
    {"loss.targ": "multipos", "name": "sw"},
]


def _log_trial_error(dpath_campaign: Path, idx_trial: int, n_trials: int, seed: int, dataset: str, setting: str, exc: Exception) -> None:
    """Log trial error to both stdout and campaign error log file."""
    dpath_campaign.mkdir(parents=True, exist_ok=True)
    fpath_errors = dpath_campaign / "errors.log"
    # Format error message with context
    error_msg = (
        f"\n[{idx_trial}/{n_trials}] TRIAL FAILED\n"
        f"  seed={seed}, dataset={dataset}, setting={setting}"
    )
    stderr_body = None
    if isinstance(exc, subprocess.CalledProcessError):
        stderr = getattr(exc, "stderr", None)
        if stderr:
            stderr_lines = stderr.splitlines()
            stderr_body = "\n".join(stderr_lines)
    # Print to stdout
    print(error_msg, flush=True)
    # Write to error log file
    with open(fpath_errors, "a") as f:
        f.write(error_msg + "\n")
        if stderr_body is not None:
            f.write("--- stderr ---\n")
            f.write(stderr_body + "\n")
        else:
            f.write(traceback.format_exc())
        f.write("\n" + "#"*80 + "\n")

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

def _reap_subtree(grace: float = 10.0) -> None:
    """SIGKILL and reap every descendant process (torchrun, ranks, DataLoader
    workers). Relies on _enable_child_subreaper so orphaned workers reparent
    here and show up as descendants."""
    parent = psutil.Process()
    deadline = time.time() + grace
    while True:
        procs = parent.children(recursive=True)
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

def _run_trial_subprocess(cfg_dict: dict) -> None:
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
        _reap_subtree()

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
    settings = _expand_settings(BASELINE_OVERRIDES)
    seeds = _iter_seeds()

    for setting, setting_payload in settings:
        _write_setting_overrides(setting, setting_payload)

    n_trials = len(seeds) * len(DATASETS) * len(settings)
    print(f"Campaign: '{CAMPAIGN}' ({n_trials} trials)")

    idx_trial = 0
    for dataset in DATASETS:
        for seed in seeds:
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
                cfg_dict["_setting_overrides"] = setting_payload
                cfg_dict = apply_overrides(cfg_dict, setting_payload)

                if dpath_trial.exists():
                    print(f"[{idx_trial}/{n_trials}] RESUME: seed={seed} dataset={dataset} setting={setting}")
                else:
                    print(f"[{idx_trial}/{n_trials}] seed={seed} dataset={dataset} setting={setting}")

                try:
                    _run_trial_subprocess(cfg_dict)
                    shutil.rmtree(dpath_trial / "chkpts/in_progress")
                except KeyboardInterrupt:
                    print(
                        f"\n[{idx_trial}/{n_trials}] INTERRUPTED — terminated trial process group; exiting campaign.",
                        flush=True,
                    )
                    return
                except Exception as e:
                    _log_trial_error(
                        dpath_campaign=_dpath_campaign(),
                        idx_trial=idx_trial,
                        n_trials=n_trials,
                        seed=seed,
                        dataset=dataset,
                        setting=setting,
                        exc=e,
                    )


if __name__ == "__main__":
    run_campaign()