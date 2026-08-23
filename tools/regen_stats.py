"""
python -m tools.regen_stats <campaign>

Regenerate a campaign's stats artifacts from its completed trials -- no train/eval rerun. Reselects, per run
(setting, dataset), the checkpoint its trials are scored at (argmax of the across-trial mean curve) and rewrites
every completed trial's evals/_best/{map,acc}/<group>.json to that checkpoint, plus
settings/<setting>/<dataset>/stats/{map,acc}/<group>/{metrics.json, metrics_listview.json, chkpt_means.pkl,
chkpt_means.png} and setting_metadata.json's best_chkpt,
and re-renders artifacts/<campaign>/stats/<dataset>/{map,acc}/<group>.png and
artifacts/<campaign>/stats/metrics/{map,acc}/<group>.xlsx (one per selection criterion x eval group), all using the
CURRENT config/stats.yaml settings (spread_type/bold_high/ordered/heatmap), so
edits to any of them take effect for an already-run campaign. Each trial's cached per-checkpoint
evals/{base,eval*}/ metrics files are reused and re-aggregated exactly as on the trial-completion path in
train.py (update_chkpt_selection -> update_metric_stats -> update_stats_tables -> update_metrics_xlsx) -- except
that the workbooks render unconditionally here, rather than only at a seed's full sweep of the matrix.
"""

import sys

from utils.config import get_config_stats
from utils.report import update_metric_stats, update_chkpt_selection, update_stats_tables, update_metrics_xlsx
from utils.train import ArtifactManager
from utils.utils import load_json, paths


def regen_campaign(campaign, cfg_stats):
    ArtifactManager.dpath_campaign = paths["artifacts"] / campaign
    metadata = load_json(ArtifactManager.dpath_campaign / "campaign_metadata.json")
    settings, datasets = metadata["settings"], metadata["datasets"]

    # per (setting, dataset) reselection + aggregations; skip combos with no trial dir (they iterdir() it)
    for setting in settings:
        ArtifactManager.dpath_setting = ArtifactManager.dpath_campaign / "settings" / setting
        for dataset in datasets:
            if (ArtifactManager.dpath_setting / dataset).exists():
                ArtifactManager.dataset = dataset
                update_chkpt_selection(cfg_stats.spread_type)
                update_metric_stats(cfg_stats.spread_type)

    for dataset in datasets:
        ArtifactManager.dataset = dataset
        update_stats_tables(
            cfg_stats.spread_type,
            cfg_stats.bold_high,
            cfg_stats.ordered,
            cfg_stats.heatmap,
            cfg_stats.supp_scores,
        )
    update_metrics_xlsx(
        cfg_stats.spread_type,
        cfg_stats.bold_high,
        cfg_stats.ordered,
        cfg_stats.heatmap,
        cfg_stats.supp_scores,
        cfg_stats.baseline_overrides,
    )


def main():
    args = sys.argv[1:]
    if len(args) != 1:
        sys.exit("usage: python -m tools.regen_stats <campaign>")
    regen_campaign(args[0], get_config_stats())


if __name__ == "__main__":
    main()
