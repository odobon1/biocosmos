"""
torchrun --standalone --nproc-per-node=auto -m campaign_runner
"""

from pathlib import Path
from copy import deepcopy
import json
import subprocess
import sys
import yaml
import traceback

from utils.config import apply_overrides, load_train_config_dict
from utils.utils import paths


CAMPAIGN_NAME = "loss_ablation"

SEED0 = 42
NUM_SEEDS = 2

DATASETS = ("lepid", "bryo", "cub")

BASELINE_OVERRIDES = [
    {"loss": {"targ": "aligned"}, "name": "iw"},
    {"loss": {"targ": "multipos"}, "name": "sw"},
    {"loss2": {"mix": 0.3, "targ": "phylo"}, "name": "hp"},
    {"batch_size": 32_000, "name": "way-too-big-bs"},
]


def _save_yaml(data: dict, fpath: Path) -> None:
    fpath.parent.mkdir(parents=True, exist_ok=True)
    with open(fpath, "w") as f:
        yaml.safe_dump(data, f, sort_keys=True)

def _load_yaml(fpath: Path) -> dict:
    with open(fpath) as f:
        return yaml.safe_load(f)

def _log_trial_error(campaign_dir: Path, idx: int, total: int, seed: int, dataset: str, setting_name: str, exc: Exception) -> None:
    """Log trial error to both stdout and campaign error log file."""
    error_log_path = campaign_dir / "campaign_errors.log"
    campaign_dir.mkdir(parents=True, exist_ok=True)
    trial_cfg_fpath = campaign_dir / "trial_cfgs" / f"trial_{idx:05d}.json"
    
    # Format error message with context
    error_msg = (
        f"\n[{idx}/{total}] TRIAL FAILED\n"
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

def _campaign_dir() -> Path:
    return paths["artifacts"] / CAMPAIGN_NAME

def _baseline_path() -> Path:
    return _campaign_dir() / "baseline.yaml"

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
    fpath = _campaign_dir() / setting_name / "overrides.json"
    fpath.parent.mkdir(parents=True, exist_ok=True)
    with open(fpath, "w") as f:
        json.dump(normalized_overrides, f, indent=2, sort_keys=True)

def _iter_seeds() -> list[int]:
    return list(range(SEED0, SEED0 + NUM_SEEDS))


def _write_trial_cfg(campaign_dir: Path, idx: int, cfg_dict: dict) -> Path:
    cfg_dir = campaign_dir / "trial_cfgs"
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
        text=True,
        stdout=None,
        stderr=subprocess.PIPE,
    )

    stderr_lines = []
    assert proc.stderr is not None
    for line in proc.stderr:
        sys.stderr.write(line)
        stderr_lines.append(line.rstrip("\n"))

    return_code = proc.wait()
    if return_code != 0:
        stderr_tail = "\n".join(stderr_lines[-200:])
        raise subprocess.CalledProcessError(return_code, cmd, stderr=stderr_tail)


def run_campaign() -> None:
    baseline = _load_or_create_baseline()
    settings = _expand_settings(BASELINE_OVERRIDES)
    seeds = _iter_seeds()

    for setting_name, _setting_payload_raw, setting_payload_norm in settings:
        _write_setting_overrides(setting_name, setting_payload_norm)

    total = len(seeds) * len(DATASETS) * len(settings)
    print(f"Campaign '{CAMPAIGN_NAME}' -> {total} total trials")

    idx = 0
    for dataset in DATASETS:
        for seed in seeds:
            for setting_name, setting_payload_raw, _setting_payload_norm in settings:
                idx += 1

                cfg_dict = apply_overrides(baseline, setting_payload_raw)
                cfg_dict["campaign_name"] = CAMPAIGN_NAME
                cfg_dict["setting_name"] = setting_name
                cfg_dict["seed"] = seed
                cfg_dict["dataset"] = dataset

                print(
                    f"[{idx}/{total}] seed={seed} dataset={dataset} setting={setting_name}"
                )
                try:
                    cfg_fpath = _write_trial_cfg(_campaign_dir(), idx, cfg_dict)
                    _run_trial_subprocess(cfg_fpath)
                except Exception as e:
                    _log_trial_error(
                        campaign_dir=_campaign_dir(),
                        idx=idx,
                        total=total,
                        seed=seed,
                        dataset=dataset,
                        setting_name=setting_name,
                        exc=e,
                    )


if __name__ == "__main__":
    run_campaign()