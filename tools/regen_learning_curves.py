"""
python -m tools.regen_learning_curves <campaign>

Re-render every trial's learning-curve plots (learning_curves/scores.png for the native eval group + learning_curves/secondary/scores_<group>.png, one per other eval group the campaign has in play + learning_curves/{optim,sim_targ_train,sim_targ_val_{id,ood,comb},scale,logscale,KL,bias,unitless_loss_blend}.png, each where the trial records its series; the trainval phase has no scores.png / secondary/), in
every phase of the campaign (screen/, and refine/ + trainval/ when they exist), from its persisted data_trial.pkl using the
CURRENT utils/report.py plotting code and config/render/learning_curves.yaml settings (dpi) -- no train/eval rerun -- so
styling/layout edits take effect for an
already-run campaign. Each trial's config is rebuilt exactly as on the campaign launch path (the phase's frozen
cfg_baseline.json snapshot + the coord's overrides.json, arm + coord overrides merged, against the SLURM alloc
recorded in phase_metadata.json rather than a live one -- so no SLURM job is needed) to recover samps_per_epoch
(the plots' epoch axis) and the split whose n-shot bucket names label the n-shot panels. Trials without a
data_trial.pkl (never checkpointed) are skipped; a tqdm bar per phase tracks the rest. Each complete arm's arm_summary/best_coord/ mirror (screen/ and refine/) is
re-copied from the re-rendered originals, so it never lags them.
"""

import pickle
import sys
from copy import deepcopy
from types import SimpleNamespace

from tqdm import tqdm

import utils.hardware
from utils.config import eval_groups, get_config_train, inject_snapshots, load_learning_curves_config_dict
from utils.report import plot_metrics, update_best_coord
from utils.train import ArtifactManager
from utils.utils import load_json, load_split, paths


def regen_learning_curves(campaign):
    dpi = load_learning_curves_config_dict()["dpi"]
    for phase in ("screen", "refine", "trainval"):
        ArtifactManager.dpath_phase = paths["artifacts"] / campaign / "_phase" / phase
        if not ArtifactManager.dpath_phase.exists():  # phase not configured (n_trials_refine null / trainval false), or not reached yet
            continue
        cfg_snapshot = load_json(ArtifactManager.dpath_phase / "cfg_baseline.json")
        metadata = load_json(ArtifactManager.dpath_phase / "phase_metadata.json")
        # TrainConfig sizes its dataloader workers against the live SLURM alloc; hand it the alloc the phase
        # recorded at launch instead, so the rebuild matches the launch path and the tool also runs outside a job
        utils.hardware.get_slurm_alloc = lambda: {key: metadata[key] for key in ("n_gpus", "n_cpus", "ram")}

        # every trial of the phase's matrix with a data_trial.pkl to re-render from
        trials = [
            (dataset, arm, coord, trial_num, seed)
            for dataset, arms in metadata["matrix"].items()
            for arm, coords in arms.items()
            for coord in coords
            for trial_num, seed in enumerate(metadata["seeds"], 1)
            if (ArtifactManager.dpath_phase / "_dataset" / dataset / "_arm" / arm / "_coord" / coord / "_trial"
                / str(trial_num) / "data_trial.pkl").exists()
        ]
        for dataset, arm, coord, trial_num, seed in tqdm(trials, desc=phase, unit="trial"):
            ArtifactManager.dpath_coord = (ArtifactManager.dpath_phase / "_dataset" / dataset / "_arm" / arm
                                           / "_coord" / coord)
            dpath_trial = ArtifactManager.dpath_coord / "_trial" / str(trial_num)
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
                cfg_dict["split"]["train_pt"] = "trainval"
            cfg = get_config_train(cfg_dict)

            ArtifactManager.dataset = dataset
            with open(dpath_trial / "data_trial.pkl", "rb") as f:
                data_tracker = SimpleNamespace(data=pickle.load(f))
            # mirrors train.py's call: bucket names from the split when eval ran, else []
            nshot_bucket_names = (
                list(load_split(cfg.dataset, cfg.split["split"]).nshot["names"])
                if cfg.split["train_pt"] != "trainval"
                else []
            )
            plot_metrics(data_tracker, dpath_trial, nshot_bucket_names, cfg.samps_per_epoch,
                         cfg.reporting["learning_curves"]["hpsm"], eval_groups(cfg.reporting), dpi)
        # each arm's arm_summary/best_coord/ mirror is a copy of curves just re-rendered: re-copy it rather than leave
        # it on the old renders
        for dataset, arms in metadata["matrix"].items():
            for arm in arms:
                update_best_coord(dataset, arm)

def main():
    args = sys.argv[1:]
    if len(args) != 1:
        sys.exit("usage: python -m tools.regen_learning_curves <campaign>")
    regen_learning_curves(args[0])


if __name__ == "__main__":
    main()
