"""
python -m tools.regen_learning_curves <campaign>

Re-render every trial's learning-curve plots (learning_curves/{native,native_macro,joint,joint_macro}.png)
from its persisted data_trial.pkl using the CURRENT utils/report.py plotting code -- no train/eval rerun --
so styling/layout edits take effect for an already-run campaign. Each trial's config is rebuilt exactly as
on the campaign launch path (frozen cfg_baseline.json snapshot + the setting's overrides.json) to recover
samps_per_epoch (the plots' epoch axis) and the split whose n-shot bucket names label the n-shot panels.
Trials without a data_trial.pkl (never checkpointed) are skipped.
"""

import pickle
import sys
from copy import deepcopy
from types import SimpleNamespace

from utils.config import get_config_train
from utils.report import plot_metrics
from utils.train import ArtifactManager
from utils.utils import load_json, load_split, paths


def regen_learning_curves(campaign):
    ArtifactManager.dpath_campaign = paths["artifacts"] / campaign
    cfg_snapshot = load_json(ArtifactManager.dpath_campaign / "cfg_baseline.json")
    metadata = load_json(ArtifactManager.dpath_campaign / "campaign_metadata.json")

    for setting in metadata["settings"]:
        ArtifactManager.dpath_setting = ArtifactManager.dpath_campaign / "settings" / setting
        overrides = load_json(ArtifactManager.dpath_setting / "overrides.json")
        for dataset in metadata["datasets"]:
            for seed in metadata["seeds"]:
                dpath_trial = ArtifactManager.dpath_setting / dataset / str(seed)
                fpath_data = dpath_trial / "data_trial.pkl"
                if not fpath_data.exists():
                    continue

                # effective trial config as on the campaign launch path (_build_trial_cfg_dict):
                # frozen snapshot + trial identity + setting overrides (applied by get_config_train)
                cfg_dict = deepcopy(cfg_snapshot["train"])
                cfg_dict["campaign"] = campaign
                cfg_dict["setting"] = setting
                cfg_dict["seed"] = seed
                cfg_dict["dataset"] = dataset
                cfg_dict["manifold_viz"] = cfg_snapshot["manifold_viz"]
                cfg_dict["model_specific"] = cfg_snapshot["model_specific"]
                cfg_dict["hw"] = cfg_snapshot["hardware"]
                cfg_dict["_setting_overrides"] = overrides
                cfg = get_config_train(cfg_dict)

                ArtifactManager.dataset = dataset
                with open(fpath_data, "rb") as f:
                    data_tracker = SimpleNamespace(data=pickle.load(f))
                # mirrors train.py's call: bucket names from the split when eval ran, else []
                nshot_bucket_names = (
                    list(load_split(cfg.dataset, cfg.split).nshot["names"])
                    if cfg.train_pt != "trainval"
                    else []
                )
                plot_metrics(data_tracker, dpath_trial, nshot_bucket_names, cfg.samps_per_epoch)
                print(f"regenerated: {dpath_trial / 'learning_curves'}")


def main():
    args = sys.argv[1:]
    if len(args) != 1:
        sys.exit("usage: python -m tools.regen_learning_curves <campaign>")
    regen_learning_curves(args[0])


if __name__ == "__main__":
    main()
