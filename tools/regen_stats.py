"""
python -m tools.regen_stats <campaign>

Regenerate a campaign's stats artifacts, phase by phase (screen/, and refine/ when it exists), from its completed
trials -- no train/eval rerun. Reselects, per run (dataset, arm, coord) of the phase's matrix, the checkpoint its
trials are scored at (argmax of the across-trial mean Native curve) and rewrites every completed trial's
evals/sel/ ({scores,secondary/scores-<group>}.json + viz/) to that checkpoint and its
evals/best/ to its own best, plus
_dataset/<dataset>/_arm/<arm>/_coord/<coord>/coord_metrics/{scores_agg.json, scores_trials.json, chkpt_means.pkl,
chkpt_means.png} (+ secondary/{scores_agg,scores_trials,chkpt_means}-<group>.*) and coord_metadata.json's best_chkpt, and re-renders every
cross-coord level: _dataset/<dataset>/_arm/<arm>/arm_metrics/performance/
({scores,convergence}.png + secondary/{scores,convergence}-<group>.png) and
_dataset/<dataset>/dataset_metrics/{armcoords,arms}/performance/ (arms/ only in refine/; each
{scores,convergence}.png + secondary/{scores,convergence}-<group>.png) and
phase_metrics/{armcoords,arms}/<campaign>_<phase>-<kind>.xlsx (+ secondary/<campaign>_<phase>-<kind>_<group>.xlsx; arms/ only in refine/).
(Native, the primary eval group, takes the plain name; every other group sits under secondary/.) Each
arm's _arm/<arm>/arm_metrics/coord_strips/ (its coords' logit-scale panels stacked) and, when the arm is
complete, its arm_metrics/best_coords/ copy of the best coord's learning curves are rebuilt with its arm_metrics/. All using
the CURRENT config/render/stats.yaml settings (spread_type/bold_high/ordered/heatmap/supp_scores/overrides), so edits to any
of them take effect for an already-run campaign,  Each trial's cached per-checkpoint
evals/evals/<k>/ metrics files are reused and re-aggregated exactly as on the trial-completion path in train.py
(update_chkpt_selection -> update_metric_stats -> update_arm_metrics -> update_dataset_metrics -> update_phase_metrics) --
except that every level renders unconditionally here, rather than only at the end of its seed cycle.
"""

import sys

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
    style = (cfg_stats.spread_type, cfg_stats.bold_high, cfg_stats.ordered, cfg_stats.heatmap, cfg_stats.supp_scores)
    # the eval groups the campaign has in play, off its frozen snapshot -- the set its trials actually
    # scored, so the live reporting.yaml having moved on can't ask for a group that was never written
    groups = eval_groups(load_json(paths["artifacts"] / campaign / "_phase" / "screen" / "cfg_baseline.json")["reporting"])
    for phase in ("screen", "refine"):  # the trainval phase runs no evals: nothing to select, aggregate or render
        ArtifactManager.dpath_phase = paths["artifacts"] / campaign / "_phase" / phase
        if not ArtifactManager.dpath_phase.exists():  # no refine phase: n_trials_refine null, or not reached yet
            continue
        matrix = load_json(ArtifactManager.dpath_phase / "phase_metadata.json")["matrix"]
        for dataset, arms in matrix.items():
            ArtifactManager.dataset = dataset
            for arm, coords in arms.items():
                # per (dataset, arm, coord) reselection + aggregations; skip combos with no trial dir (they iterdir() it)
                for coord in coords:
                    ArtifactManager.dpath_coord = (ArtifactManager.dpath_phase / "_dataset" / dataset / "_arm" / arm
                                                   / "_coord" / coord)
                    if ArtifactManager.dpath_coord.exists():
                        update_chkpt_selection(groups, cfg_stats.spread_type)
                        update_metric_stats(groups, cfg_stats.spread_type)
                update_arm_metrics(dataset, arm, groups, *style)
            update_dataset_metrics(dataset, groups, *style)
        update_phase_metrics(groups, *style, cfg_stats.overrides)


def main():
    args = sys.argv[1:]
    if len(args) != 1:
        sys.exit("usage: python -m tools.regen_stats <campaign>")
    regen_campaign(args[0], get_config_stats())


if __name__ == "__main__":
    main()
