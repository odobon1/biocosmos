"""
torchrun --standalone --nproc-per-node=auto -m campaign_runner
"""

from pathlib import Path
from copy import deepcopy
import json
import yaml  # type: ignore[import]
import traceback

from train import run_training
from utils.config import apply_overrides, get_config_train, load_train_config_dict
from utils.utils import paths


CAMPAIGN_NAME = "loss_ablation"

SEED0 = 42
NUM_SEEDS = 2

DATASETS = ("bryo", "cub", "lepid")

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
    
    # Format error message with context
    error_msg = (
        f"\n[{idx}/{total}] TRIAL FAILED\n"
        f"  seed={seed}, dataset={dataset}, setting={setting_name}\n"
        f"  {type(exc).__name__}: {str(exc)}"
    )
    
    # Print to stdout
    print(error_msg, flush=True)
    
    # Write to error log file
    with open(error_log_path, "a") as f:
        f.write(error_msg + "\n")
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
                    cfg = get_config_train(cfg_dict=cfg_dict)
                    run_training(cfg)
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