"""
python -m tools.regen_stats <campaign>

Regenerate a campaign's stats artifacts, phase by phase (screen/, and refine/ when it exists), from its completed
trials -- no train/eval rerun. Reselects, per run (dataset, arm, coord) of the phase's matrix, the checkpoint its
trials are scored at (argmax of the across-trial mean Native curve) and rewrites every completed trial's
eval/sel/ ({scores,secondary/scores_<group>}.json + viz/) to that checkpoint and its
eval/best/ to its own best, plus
_dataset/<dataset>/_arm/<arm>/_coord/<coord>/coord_summary/{scores_agg.json, scores_trials.json, chkpt_means.pkl,
chkpt_means.png} (+ secondary/{scores_agg,scores_trials,chkpt_means}_<group>.*) and coord_metadata.json's sel_chkpt, and re-renders every
cross-coord level: _dataset/<dataset>/_arm/<arm>/arm_summary/performance/
({scores,convergence}.png + secondary/{scores,convergence}_<group>.png; in screen/, a two-param sweep's
{scores_hpo2d}.png too) and
_dataset/<dataset>/dataset_summary/performance/ ({scores,convergence}.png + secondary/{scores,convergence}_<group>.png) and
phase_summary/{armcoords,arms}/<campaign>_<phase>-<kind>.xlsx (+ secondary/<campaign>_<phase>-<kind>_<group>.xlsx; arms/ only in refine/).
(Native, the primary eval group, takes the plain name; every other group sits under secondary/.) Each
arm's _arm/<arm>/arm_summary/coord_strips/ (its coords' logit-scale and pair-entropy panels stacked), its
arm_summary/{scale,logscale}/{trials/<n>,agg}/convergence.png (the coords' logit scales overlaid, coloured by lr.init; a
lr.init x loss.logits.scale.init sweep only) and, when the arm is
complete, its arm_summary/best_coord/ copy of the best coord's learning curves + eval/{best,sel}/ are rebuilt with its arm_summary/. All using
the CURRENT config/render/stats.yaml settings (spread_type/bold_high/ordered/heatmap/supp_scores/overrides/dpi), so edits to any
of them take effect for an already-run campaign,  Each trial's cached per-checkpoint
eval/all/<k>/ metrics files are reused and re-aggregated exactly as on the trial-completion path in train.py
(update_chkpt_selection -> update_metric_stats -> update_arm_metrics -> update_dataset_metrics -> update_phase_metrics) --
except that the phase workbooks render unconditionally here, rather than only at the end of a full seed sweep.
A tqdm bar per phase tracks its (dataset, arm, coord) runs.
"""

import sys

from tqdm import tqdm

from utils.config import eval_groups, get_config_stats
from utils.report import (
    update_metric_stats,
    update_chkpt_selection,
    update_arm_metrics,
    update_dataset_metrics,
    update_phase_metrics,
)
from utils.train import ArtifactManager
from utils.utils import load_json, paths


def regen_campaign(campaign, cfg_stats):
    # the eval groups the campaign has in play, off its frozen snapshot -- the set its trials actually
    # scored, so the live reporting.yaml having moved on can't ask for a group that was never written
    groups = eval_groups(load_json(paths["artifacts"] / campaign / "_phase" / "screen" / "cfg_baseline.json")["reporting"])
    for phase in ("screen", "refine"):  # the trainval phase runs no evals: nothing to select, aggregate or render
        ArtifactManager.dpath_phase = paths["artifacts"] / campaign / "_phase" / phase
        if not ArtifactManager.dpath_phase.exists():  # no refine phase: n_trials_refine null, or not reached yet
            continue
        matrix = load_json(ArtifactManager.dpath_phase / "phase_metadata.json")["matrix"]
        n_coords = sum(len(coords) for arms in matrix.values() for coords in arms.values())
        with tqdm(total=n_coords, desc=phase, unit="coord") as pbar:
            for dataset, arms in matrix.items():
                ArtifactManager.dataset = dataset
                for arm, coords in arms.items():
                    # per (dataset, arm, coord) reselection + aggregations; skip combos with no trial dir (they iterdir() it)
                    for coord in coords:
                        ArtifactManager.dpath_coord = (ArtifactManager.dpath_phase / "_dataset" / dataset / "_arm" / arm
                                                       / "_coord" / coord)
                        if ArtifactManager.dpath_coord.exists():
                            update_chkpt_selection(groups, cfg_stats)
                            update_metric_stats(groups, cfg_stats)
                        pbar.update(1)
                    update_arm_metrics(dataset, arm, groups, cfg_stats)
                update_dataset_metrics(dataset, groups, cfg_stats)
        update_phase_metrics(groups, cfg_stats)


def main():
    args = sys.argv[1:]
    if len(args) != 1:
        sys.exit("usage: python -m tools.regen_stats <campaign>")
    regen_campaign(args[0], get_config_stats())


if __name__ == "__main__":
    main()
