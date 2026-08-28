"""
python -m tools.regen_stats <campaign>

Regenerate a campaign's stats artifacts, phase by phase (screening/, and qual/ when it exists), from its completed
trials -- no train/eval rerun. Reselects, per run (dataset, arm, coord) of the phase's matrix, the checkpoint its
trials are scored at (argmax of the across-trial mean curve) and rewrites every completed trial's
evals/_best/{map,acc}/<group>.json to that checkpoint, plus
datasets/<dataset>/arms/<arm>/coords/<coord>/coord_stats/{map,acc}/<group>/{metrics.json, metrics_listview.json,
chkpt_means.pkl, chkpt_means.png} and coord_metadata.json's best_chkpt, and re-renders every cross-coord level:
datasets/<dataset>/arms/<arm>/arm_stats/ and datasets/<dataset>/dataset_stats/{arm_coords,arms}/ (each
{map,acc}/<group>/{metrics,convergence}.png) and campaign_stats/{arm_coords,arms}/{map,acc}/<group>.xlsx, all using
the CURRENT config/stats.yaml settings (spread_type/bold_high/ordered/heatmap/supp_scores/overrides), so edits to any
of them take effect for an already-run campaign. Each trial's cached per-checkpoint evals/{base,eval*}/ metrics
files are reused and re-aggregated exactly as on the trial-completion path in train.py (update_chkpt_selection ->
update_metric_stats -> update_arm_stats -> update_dataset_stats -> update_campaign_stats) -- except that every
level renders unconditionally here, rather than only at the end of its seed cycle.
"""

import sys

from utils.config import get_config_stats
from utils.report import (
    update_metric_stats,
    update_chkpt_selection,
    update_arm_stats,
    update_dataset_stats,
    update_campaign_stats,
)
from utils.train import ArtifactManager
from utils.utils import load_json, paths


def regen_campaign(campaign, cfg_stats):
    style = (cfg_stats.spread_type, cfg_stats.bold_high, cfg_stats.ordered, cfg_stats.heatmap, cfg_stats.supp_scores)
    for phase in ("screening", "qual"):
        ArtifactManager.dpath_phase = paths["artifacts"] / campaign / phase
        if not ArtifactManager.dpath_phase.exists():  # no qual phase: n_trials_qual null, or not reached yet
            continue
        matrix = load_json(ArtifactManager.dpath_phase / "campaign_metadata.json")["matrix"]
        for dataset, arms in matrix.items():
            ArtifactManager.dataset = dataset
            for arm, coords in arms.items():
                # per (dataset, arm, coord) reselection + aggregations; skip combos with no trial dir (they iterdir() it)
                for coord in coords:
                    ArtifactManager.dpath_coord = (ArtifactManager.dpath_phase / "datasets" / dataset / "arms" / arm
                                                   / "coords" / coord)
                    if ArtifactManager.dpath_coord.exists():
                        update_chkpt_selection(cfg_stats.spread_type)
                        update_metric_stats(cfg_stats.spread_type)
                update_arm_stats(dataset, arm, *style)
            update_dataset_stats(dataset, *style)
        update_campaign_stats(*style, cfg_stats.overrides)


def main():
    args = sys.argv[1:]
    if len(args) != 1:
        sys.exit("usage: python -m tools.regen_stats <campaign>")
    regen_campaign(args[0], get_config_stats())


if __name__ == "__main__":
    main()
