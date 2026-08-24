"""
Qualified campaigns are defined in config/quals/<name>.yaml and launched through the campaign
queue: a qual.<name> entry in config/camp_queue.yaml (see campaign_runner).

A qualified campaign tops up a subset of a completed base campaign's settings (the ones that
"qualified") to n_trials_qual trials per setting. It creates campaign <base_campaign>_qual seeded
from the base campaign -- the frozen config snapshot plus each qualified setting's completed trials
are copied over, as if the qualified campaign had run them -- then the standard campaign machinery
runs the missing seeds (the ones above the base campaign's). Datasets are the base campaign's;
per-setting overrides come from the base campaign's persisted overrides.json.
"""

import shutil
import torch
import yaml

from campaign_runner import _check_trial_completion, _dedupe_campaign_name, _dpath_campaign, run_campaign
from utils.config import load_train_config_dict
from utils.utils import load_json, paths


def _load_qual_config(name: str) -> dict:
    fpath = paths["config"] / "quals" / f"{name}.yaml"
    if not fpath.exists():
        avail = ", ".join(sorted(p.stem for p in (paths["config"] / "quals").glob("*.yaml")))
        raise SystemExit(f"Qual config not found: {fpath}\nAvailable qual configs: {avail}")
    with open(fpath) as f:
        return yaml.safe_load(f)

def _check_base_complete(base_campaign: str, metadata_base: dict) -> None:
    """Every trial in the base campaign's recorded matrix must be complete: qualification copies the
    base's results wholesale and extends its seed sweeps, so a partially-run base has nothing settled
    to qualify from."""
    dpath_base = _dpath_campaign(base_campaign)
    incomplete = [
        f"{setting}/{dataset}/{seed}"
        for setting in metadata_base["settings"]
        for dataset in metadata_base["datasets"]
        for seed in metadata_base["seeds"]
        if not _check_trial_completion(dpath_base / "settings" / setting / dataset / str(seed))
    ]
    if incomplete:
        raise RuntimeError(
            f"base_campaign '{base_campaign}' is not complete -- incomplete trials:\n  "
            + "\n  ".join(incomplete)
        )

def _copy_base_artifacts(base_campaign: str, campaign: str, qualified_settings: list[str]) -> None:
    """Seed the qual campaign from the base campaign: the frozen config snapshot (cfg_baseline.json,
    so qual trials train against the base campaign's frozen config, not the current yamls) plus each
    qualified setting's whole directory (trials, metadata, per-dataset stats) -- as if the qual
    campaign had run those trials itself. Idempotent per item (whatever already exists is left
    alone), so a relaunch copies nothing and a later extension of qualified_settings copies only the
    newly-qualified settings. A setting is copied to a temp name and renamed into place, so an
    interrupted copy never leaves a half-copied setting dir that a relaunch would take as done."""
    dpath_base = _dpath_campaign(base_campaign)
    dpath_qual = _dpath_campaign(campaign)
    fpath_snapshot = dpath_qual / "cfg_baseline.json"
    if not fpath_snapshot.exists():
        dpath_qual.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(dpath_base / "cfg_baseline.json", fpath_snapshot)
    for setting in qualified_settings:
        dpath_dst = dpath_qual / "settings" / setting
        if dpath_dst.exists():
            continue
        dpath_tmp = dpath_dst.parent / f".copying_{setting}"
        if dpath_tmp.exists():
            shutil.rmtree(dpath_tmp)
        shutil.copytree(dpath_base / "settings" / setting, dpath_tmp)
        dpath_tmp.rename(dpath_dst)
        print(f"copied {base_campaign}/settings/{setting} -> {campaign}/settings/{setting}", flush=True)

def run_qual_campaign(n_trials_qual: int, base_campaign: str, qualified_settings: list[str]) -> bool:
    fpath_meta_base = _dpath_campaign(base_campaign) / "campaign_metadata.json"
    if not fpath_meta_base.exists():
        raise FileNotFoundError(f"base_campaign '{base_campaign}' not found: {fpath_meta_base}")
    metadata_base = load_json(fpath_meta_base)

    if not qualified_settings:
        raise ValueError("qualified_settings is empty")
    if len(set(qualified_settings)) != len(qualified_settings):
        raise ValueError(f"qualified_settings has duplicates: {qualified_settings}")
    unknown = [s for s in qualified_settings if s not in metadata_base["settings"]]
    if unknown:
        raise ValueError(
            f"qualified_settings not in base_campaign '{base_campaign}': {unknown} "
            f"(base settings: {metadata_base['settings']})"
        )
    n_trials_base = len(metadata_base["seeds"])
    if n_trials_qual < n_trials_base:
        raise ValueError(
            f"n_trials_qual ({n_trials_qual}) is below base_campaign '{base_campaign}'s "
            f"{n_trials_base} trials per setting; a qual campaign only extends the base's seed sweep."
        )
    n_gpus = torch.cuda.device_count()
    if n_gpus != metadata_base["n_gpus"]:
        raise RuntimeError(
            f"GPU count mismatch: base_campaign '{base_campaign}' was run with "
            f"{metadata_base['n_gpus']} GPUs but current environment has {n_gpus}; qual trials "
            f"extend the base's seed sweeps and must run under the same world size."
        )
    _check_base_complete(base_campaign, metadata_base)

    campaign = f"{base_campaign}_qual"
    if not load_train_config_dict()["dev"]["continue_campaign"]:
        deduped = _dedupe_campaign_name(campaign)
        if deduped != campaign:
            print(f"campaign '{campaign}' already exists -- starting '{deduped}' (dev.continue_campaign: false)", flush=True)
            campaign = deduped

    _copy_base_artifacts(base_campaign, campaign, qualified_settings)

    dpath_base_settings = _dpath_campaign(base_campaign) / "settings"
    settings = [
        (setting, load_json(dpath_base_settings / setting / "overrides.json"))
        for setting in qualified_settings
    ]
    return run_campaign(
        campaign=campaign,
        n_trials=n_trials_qual,
        datasets=metadata_base["datasets"],
        settings=settings,
    )

def launch(name: str) -> bool:
    """Run the qual campaign defined by config/quals/<name>.yaml (queued as 'qual.<name>' in
    config/camp_queue.yaml); returns run_campaign's completed flag."""
    cfg = _load_qual_config(name)
    return run_qual_campaign(
        n_trials_qual=cfg["n_trials_qual"],
        base_campaign=cfg["base_campaign"],
        qualified_settings=cfg["qualified_settings"],
    )
