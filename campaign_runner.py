"""
python -m campaign_runner
"""

from pathlib import Path
from copy import deepcopy
import json
import shutil
import subprocess
import sys
import traceback
import time
import torch

from utils.config import apply_overrides, apply_train_debug_overrides, load_train_config_dict
from utils.utils import paths, save_pickle, save_json, load_json


CAMPAIGN_NAME = "dev2"

SEED0 = 42
NUM_SEEDS = 3

DATASETS = ("nymph",)

BASELINE_OVERRIDES = [
    # {"batch_size": 32_000, "name": "way-too-big-bs"},
    {"loss2": {"mix": 0.3, "targ": "phylo"}, "name": "hp"},
    {"loss": {"targ": "aligned"}, "name": "iw"},
    {"loss": {"targ": "multipos"}, "name": "sw"},
]


def _log_trial_error(dpath_campaign: Path, idx_trial: int, n_trials: int, seed: int, dataset_name: str, setting_name: str, exc: Exception) -> None:
    """Log trial error to both stdout and campaign error log file."""
    dpath_campaign.mkdir(parents=True, exist_ok=True)
    fpath_errors = dpath_campaign / "errors.log"
    # Format error message with context
    error_msg = (
        f"\n[{idx_trial}/{n_trials}] TRIAL FAILED\n"
        f"  seed={seed}, dataset={dataset_name}, setting={setting_name}"
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
    return paths["artifacts"] / CAMPAIGN_NAME

def _load_or_create_baseline_config() -> dict:
    fpath = _dpath_campaign() / "cfg_baseline.json"
    if fpath.exists():
        return load_json(fpath)

    cfg_baseline = load_train_config_dict()
    cfg_baseline = apply_train_debug_overrides(cfg_baseline)
    for key in ("setting_name", "seed", "dataset_name"):
        cfg_baseline.pop(key, None)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    save_json(cfg_baseline, fpath)
    return cfg_baseline

def _normalize_setting_overrides(overrides: dict) -> dict:
    # Keep nested keys as primary representation and only consume slash keys as fallback.
    return apply_overrides({}, overrides)

def _expand_settings(settings_raw: list[dict]) -> list[tuple[str, dict, dict]]:
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
        normalized_payload = _normalize_setting_overrides(payload)
        settings.append((name, payload, normalized_payload))
    return settings

def _write_setting_overrides(setting_name: str, normalized_overrides: dict) -> None:
    fpath = _dpath_campaign() / setting_name / "overrides.json"
    fpath.parent.mkdir(parents=True, exist_ok=True)
    with open(fpath, "w") as f:
        json.dump(normalized_overrides, f, indent=2, sort_keys=True)

def _iter_seeds() -> list[int]:
    return list(range(SEED0, SEED0 + NUM_SEEDS))

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

    proc = subprocess.Popen(
        cmd,
        stdout=None,
        stderr=subprocess.PIPE,
    )

    stderr_data = b""
    assert proc.stderr is not None
    while chunk := proc.stderr.read1(4096):
        sys.stderr.buffer.write(chunk)
        sys.stderr.buffer.flush()
        stderr_data += chunk

    return_code = proc.wait()
    if return_code != 0:
        stderr_body = "\n".join(stderr_data.decode(errors="replace").splitlines())
        raise subprocess.CalledProcessError(return_code, cmd, stderr=stderr_body)

def _check_trial_completion(dpath_trial: Path) -> bool:
    fpath_metadata_trial = dpath_trial / "metadata_trial.json"
    if not fpath_metadata_trial.exists():
        complete = False
    else:
        metadata_trial = load_json(fpath_metadata_trial)
        complete = metadata_trial["complete"]
    return complete

def run_campaign() -> None:
    time_data = {
        "last_updated": time.time(),
        "elapsed": 0.0,
    }
    dpath_campaign = _dpath_campaign()
    dpath_campaign.mkdir(parents=True, exist_ok=True)
    save_pickle(time_data, dpath_campaign / "time.pkl")

    n_gpus = torch.cuda.device_count()
    fpath_meta = dpath_campaign / "metadata_campaign.json"
    if fpath_meta.exists():
        existing_meta = load_json(fpath_meta)
        if existing_meta["n_gpus"] != n_gpus:
            raise RuntimeError(
                f"GPU count mismatch: campaign '{CAMPAIGN_NAME}' was run with "
                f"{existing_meta['n_gpus']} GPUs but current environment has {n_gpus}."
            )
    else:
        metadata_camp = {
            "duration": "0-00:00:00",
            "n_gpus": n_gpus,
        }
        save_json(metadata_camp, dpath_campaign / "metadata_campaign.json")

    cfg_baseline = _load_or_create_baseline_config()
    settings = _expand_settings(BASELINE_OVERRIDES)
    seeds = _iter_seeds()

    for setting_name, _setting_payload_raw, setting_payload_norm in settings:
        _write_setting_overrides(setting_name, setting_payload_norm)

    n_trials = len(seeds) * len(DATASETS) * len(settings)
    print(f"Campaign: '{CAMPAIGN_NAME}' ({n_trials} trials)")

    idx_trial = 0
    for dataset_name in DATASETS:
        for seed in seeds:
            for setting_name, _setting_payload_raw, setting_payload_norm in settings:
                idx_trial += 1

                dpath_trial = _dpath_campaign() / setting_name / dataset_name / str(seed)
                if _check_trial_completion(dpath_trial):
                    print(f"[{idx_trial}/{n_trials}] SKIP (completed): seed={seed} dataset={dataset_name} setting={setting_name}")
                    continue

                cfg_dict = deepcopy(cfg_baseline)
                cfg_dict["campaign_name"] = CAMPAIGN_NAME
                cfg_dict["setting_name"] = setting_name
                cfg_dict["seed"] = seed
                cfg_dict["dataset_name"] = dataset_name
                cfg_dict["standalone"] = False
                cfg_dict["_setting_overrides"] = setting_payload_norm
                cfg_dict = apply_overrides(cfg_dict, setting_payload_norm)

                if dpath_trial.exists():
                    print(f"[{idx_trial}/{n_trials}] RESUME: seed={seed} dataset={dataset_name} setting={setting_name}")
                else:
                    print(f"[{idx_trial}/{n_trials}] seed={seed} dataset={dataset_name} setting={setting_name}")

                try:
                    _run_trial_subprocess(cfg_dict)
                    shutil.rmtree(dpath_trial / "chkpts/in_progress")
                except Exception as e:
                    _log_trial_error(
                        dpath_campaign=_dpath_campaign(),
                        idx_trial=idx_trial,
                        n_trials=n_trials,
                        seed=seed,
                        dataset_name=dataset_name,
                        setting_name=setting_name,
                        exc=e,
                    )


if __name__ == "__main__":
    run_campaign()