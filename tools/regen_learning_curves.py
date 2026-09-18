"""
python -m tools.regen_learning_curves <campaign>

Re-render every trial's learning-curve plots (learning_curves/scores/<group>.png, one per eval group the campaign has in play + learning_curves/{general,alpha,logalpha,KL}.png; the trainval phase has no scores/), in
every phase of the campaign (_screen/, and qual/ + trainval/ when they exist), from its persisted data_trial.pkl using the
CURRENT utils/report.py plotting code -- no train/eval rerun -- so styling/layout edits take effect for an
already-run campaign. Each trial's config is rebuilt exactly as on the campaign launch path (the phase's frozen
cfg_baseline.json snapshot + the coord's overrides.json, arm + coord overrides merged, against the SLURM alloc
recorded in phase_metadata.json rather than a live one -- so no SLURM job is needed) to recover samps_per_epoch
(the plots' epoch axis) and the split whose n-shot bucket names label the n-shot panels. Trials without a
data_trial.pkl (never checkpointed) are skipped.
"""

import pickle
import sys
from copy import deepcopy
from types import SimpleNamespace

import utils.hardware
from utils.config import eval_groups, get_config_train, inject_snapshots
from utils.report import plot_metrics
from utils.train import ArtifactManager
from utils.utils import load_json, load_split, paths


def regen_learning_curves(campaign):
    for phase in ("_screen", "qual", "trainval"):
        ArtifactManager.dpath_phase = paths["artifacts"] / campaign / phase
        if not ArtifactManager.dpath_phase.exists():  # phase not configured (n_trials_qual null / trainval false), or not reached yet
            continue
        cfg_snapshot = load_json(ArtifactManager.dpath_phase / "cfg_baseline.json")
        metadata = load_json(ArtifactManager.dpath_phase / "phase_metadata.json")
        # TrainConfig sizes its dataloader workers against the live SLURM alloc; hand it the alloc the phase
        # recorded at launch instead, so the rebuild matches the launch path and the tool also runs outside a job
        utils.hardware.get_slurm_alloc = lambda: {key: metadata[key] for key in ("n_gpus", "n_cpus", "ram")}

        for dataset, arms in metadata["matrix"].items():
            for arm, coords in arms.items():
                for coord in coords:
                    ArtifactManager.dpath_coord = (ArtifactManager.dpath_phase / "_datasets" / dataset / "_arms" / arm
                                                   / "_coords" / coord)
                    for seed in metadata["seeds"]:
                        dpath_trial = ArtifactManager.dpath_coord / "_seeds" / str(seed)
                        fpath_data = dpath_trial / "data_trial.pkl"
                        if not fpath_data.exists():
                            continue
                        overrides = load_json(ArtifactManager.dpath_coord / "overrides.json")

                        # effective trial config as on the campaign launch path (_build_trial_cfg_dict):
                        # frozen snapshot + trial identity + the merged arm/coord overrides (applied by get_config_train)
                        cfg_dict = deepcopy(cfg_snapshot["train"])
                        cfg_dict["campaign"] = campaign
                        cfg_dict["phase"] = phase
                        cfg_dict["arm"] = arm
                        cfg_dict["coord"] = coord
                        cfg_dict["seed"] = seed
                        cfg_dict["dataset"] = dataset
                        inject_snapshots(cfg_dict, cfg_snapshot)
                        cfg_dict["_overrides"] = {**overrides["baseline"], **overrides["arm"], **overrides["coord"]}
                        if phase == "trainval":  # the runner's phase-level injection; chkpt_stop doesn't touch the epoch axis
                            cfg_dict["train_pt"] = "trainval"
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
                        plot_metrics(data_tracker, dpath_trial, nshot_bucket_names, cfg.samps_per_epoch,
                                     cfg.reporting["learning_curves"]["hpsm"], eval_groups(cfg.reporting))
                        print(f"regenerated: {dpath_trial / 'learning_curves'}")


def main():
    args = sys.argv[1:]
    if len(args) != 1:
        sys.exit("usage: python -m tools.regen_learning_curves <campaign>")
    regen_learning_curves(args[0])


if __name__ == "__main__":
    main()
