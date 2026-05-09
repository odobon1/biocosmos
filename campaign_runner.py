"""
torchrun --standalone --nproc-per-node=auto -m campaign_runner
"""

from pathlib import Path
from copy import deepcopy
import gc
import json
import yaml  # type: ignore[import]
import traceback

from train import run_training
from utils.config import apply_overrides, get_config_hardware, get_config_train, load_train_config_dict
from utils.data import build_img_cache, spawn_partition_data
from utils.eval import list_eval_partition_names
from utils.utils import load_split, paths


campaign_name = "loss_ablation4"

seed0 = 42
num_seeds = 2

DATASETS = ("bryo", "cub", "lepid")

baseline_overrides = [
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
    return paths["artifacts"] / campaign_name

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
            raise ValueError("Each baseline_overrides item must include a 'name' field.")

        name = item["name"]
        if name in seen_names:
            raise ValueError(f"Duplicate baseline_overrides name: {name}")
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
    return list(range(seed0, seed0 + num_seeds))


def _build_img_cache(dataset: str, cfg_dict: dict) -> dict | None:
    """Build one image cache dict covering train + all eval partitions.
    Keys are relative file paths; values are PIL images (pl) or tensors (pp).
    Returns None when caching is disabled in hardware.yaml.
    """
    hw_cfg = get_config_hardware()
    if hw_cfg.cached_imgs not in ("pl", "pp"):
        return None

    cfg = get_config_train(cfg_dict=cfg_dict)

    # Collect unique relpaths across train + all eval partitions.
    split = load_split(cfg.split_name, dataset=dataset)
    partition_names = ["train"] + list_eval_partition_names(split, cfg.eval_type)
    seen: set[str] = set()
    rfpaths: list[str] = []
    for pname in partition_names:
        index_data, _ = spawn_partition_data(config=cfg, partition_name=pname)
        for datum in index_data:
            rfpath = datum["rfpath"]
            if rfpath not in seen:
                seen.add(rfpath)
                rfpaths.append(rfpath)

    if hw_cfg.cached_imgs == "pp":
        from models import VLMWrapper  # avoid heavy top-level import
        img_pp = VLMWrapper.build(cfg, verbose=False).img_pp_train
    else:
        img_pp = None  # unused for "pl"

    return build_img_cache(
        rfpaths=rfpaths,
        dataset=dataset,
        img_pp=img_pp,
        n_workers=cfg.n_workers,
        cached_imgs=hw_cfg.cached_imgs,
        desc=f"Caching images ({hw_cfg.cached_imgs}) [{dataset}]",
    )


def run_campaign() -> None:
    baseline = _load_or_create_baseline()
    settings = _expand_settings(baseline_overrides)
    seeds = _iter_seeds()

    for setting_name, _setting_payload_raw, setting_payload_norm in settings:
        _write_setting_overrides(setting_name, setting_payload_norm)

    total = len(seeds) * len(DATASETS) * len(settings)
    print(f"Campaign '{campaign_name}' -> {total} total trials")

    idx = 0
    for dataset in DATASETS:
        cfg_dict_cache = apply_overrides(baseline, {})
        cfg_dict_cache["campaign_name"] = campaign_name
        cfg_dict_cache["setting_name"] = settings[0][0]
        cfg_dict_cache["seed"] = seeds[0]
        cfg_dict_cache["dataset"] = dataset
        imgs_mem_dataset = _build_img_cache(dataset, cfg_dict_cache)

        try:
            for seed in seeds:
                for setting_name, setting_payload_raw, _setting_payload_norm in settings:
                    idx += 1

                    cfg_dict = apply_overrides(baseline, setting_payload_raw)
                    cfg_dict["campaign_name"] = campaign_name
                    cfg_dict["setting_name"] = setting_name
                    cfg_dict["seed"] = seed
                    cfg_dict["dataset"] = dataset

                    print(
                        f"[{idx}/{total}] seed={seed} dataset={dataset} setting={setting_name}"
                    )
                    try:
                        cfg = get_config_train(cfg_dict=cfg_dict)
                        run_training(cfg, imgs_mem=imgs_mem_dataset)
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
        finally:
            del imgs_mem_dataset
            gc.collect()


if __name__ == "__main__":
    run_campaign()