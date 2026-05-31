"""
python -m campaign_runner
"""

from pathlib import Path
from copy import deepcopy
import json
import subprocess
import sys
import yaml
import traceback
import time
import torch

from utils.config import apply_overrides, load_train_config_dict
from utils.utils import paths, save_pickle, save_json, load_json


CAMPAIGN_NAME = "dev"

SEED0 = 42
NUM_SEEDS = 1

DATASETS = ("lepid",)

BASELINE_OVERRIDES = [
    {"loss": {"targ": "aligned"}, "name": "iw"},
    {"loss": {"targ": "multipos"}, "name": "sw"},
    {"loss2": {"mix": 0.3, "targ": "phylo"}, "name": "hp"},
]


def _save_yaml(data: dict, fpath: Path) -> None:
    fpath.parent.mkdir(parents=True, exist_ok=True)
    with open(fpath, "w") as f:
        yaml.safe_dump(data, f, sort_keys=True)

def _load_yaml(fpath: Path) -> dict:
    with open(fpath) as f:
        return yaml.safe_load(f)

def _log_trial_error(dpath_campaign: Path, idx: int, n_trials: int, seed: int, dataset: str, setting_name: str, exc: Exception) -> None:
    """Log trial error to both stdout and campaign error log file."""
    error_log_path = dpath_campaign / "campaign_errors.log"
    dpath_campaign.mkdir(parents=True, exist_ok=True)
    trial_cfg_fpath = dpath_campaign / "trial_cfgs" / f"trial_{idx:05d}.json"
    
    # Format error message with context
    error_msg = (
        f"\n[{idx}/{n_trials}] TRIAL FAILED\n"
        f"  seed={seed}, dataset={dataset}, setting={setting_name}\n"
        f"  cfg={trial_cfg_fpath}\n"
        f"  {type(exc).__name__}: {str(exc)}"
    )

    stderr_tail = None
    if isinstance(exc, subprocess.CalledProcessError):
        stderr = getattr(exc, "stderr", None)
        if stderr:
            stderr_lines = stderr.splitlines()
            stderr_tail = "\n".join(stderr_lines[-200:])
    
    # Print to stdout
    print(error_msg, flush=True)
    
    # Write to error log file
    with open(error_log_path, "a") as f:
        f.write(error_msg + "\n")
        if stderr_tail is not None:
            f.write("--- stderr (tail) ---\n")
            f.write(stderr_tail + "\n")
        else:
            f.write(traceback.format_exc())
        f.write("\n" + "="*80 + "\n")

def _dpath_campaign() -> Path:
    return paths["artifacts"] / CAMPAIGN_NAME

def _baseline_path() -> Path:
    return _dpath_campaign() / "baseline.yaml"

def _load_or_create_baseline() -> dict:
    fpath = _baseline_path()
    if fpath.exists():
        return _load_yaml(fpath)

    baseline = load_train_config_dict()
    _save_yaml(baseline, fpath)
    return baseline

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

def _write_trial_cfg(dpath_campaign: Path, idx: int, cfg_dict: dict) -> Path:
    cfg_dir = dpath_campaign / "trial_cfgs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_fpath = cfg_dir / f"trial_{idx:05d}.json"
    with open(cfg_fpath, "w") as f:
        json.dump(cfg_dict, f, indent=2, sort_keys=True)
    return cfg_fpath

def _run_trial_subprocess(cfg_fpath: Path) -> None:
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc-per-node=auto",
        "-m",
        "campaign_trial_runner",
        "--cfg-path",
        str(cfg_fpath),
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
        stderr_tail = "\n".join(stderr_data.decode(errors="replace").splitlines()[-200:])
        raise subprocess.CalledProcessError(return_code, cmd, stderr=stderr_tail)


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

    baseline = _load_or_create_baseline()
    settings = _expand_settings(BASELINE_OVERRIDES)
    seeds = _iter_seeds()

    for setting_name, _setting_payload_raw, setting_payload_norm in settings:
        _write_setting_overrides(setting_name, setting_payload_norm)

    n_trials = len(seeds) * len(DATASETS) * len(settings)
    print(f"Campaign: '{CAMPAIGN_NAME}' ({n_trials} trials)")

    idx = 0
    for dataset in DATASETS:
        for seed in seeds:
            for setting_name, setting_payload_raw, _setting_payload_norm in settings:
                idx += 1

                dpath_trial = _dpath_campaign() / setting_name / dataset / str(seed)
                if (dpath_trial / "completed").exists():
                    print(f"[{idx}/{n_trials}] SKIP (completed): seed={seed} dataset={dataset} setting={setting_name}")
                    continue

                cfg_dict = apply_overrides(baseline, setting_payload_raw)
                cfg_dict["campaign_name"] = CAMPAIGN_NAME
                cfg_dict["setting_name"] = setting_name
                cfg_dict["seed"] = seed
                cfg_dict["dataset"] = dataset
                cfg_dict["standalone"] = False

                if dpath_trial.exists():
                    print(f"[{idx}/{n_trials}] RESUME: seed={seed} dataset={dataset} setting={setting_name}")
                else:
                    print(f"[{idx}/{n_trials}] seed={seed} dataset={dataset} setting={setting_name}")

                try:
                    cfg_fpath = _write_trial_cfg(_dpath_campaign(), idx, cfg_dict)
                    _run_trial_subprocess(cfg_fpath)
                except Exception as e:
                    _log_trial_error(
                        dpath_campaign=_dpath_campaign(),
                        idx=idx,
                        n_trials=n_trials,
                        seed=seed,
                        dataset=dataset,
                        setting_name=setting_name,
                        exc=e,
                    )


if __name__ == "__main__":
    run_campaign()