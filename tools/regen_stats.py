"""
python -m tools.regen_stats <campaign>

Regenerate a campaign's stats artifacts from its completed trials -- no train/eval rerun. Rewrites, per run
(setting, dataset), settings/<setting>/<dataset>/stats/{metrics,metrics_listview}.json, and re-renders
artifacts/<campaign>/stats/<dataset>/{map,acc}.png and artifacts/<campaign>/stats/metrics.xlsx, all using the
CURRENT config/train.yaml stats settings (stats.spread_type/table_eval_group/xlsx.*), so edits to the spread
type, eval group, or xlsx bold_high/ordered/heatmap take effect for an already-run campaign. Each trial's
cached evals/final/metrics.json is reused and re-aggregated exactly as on the trial-completion path in
train.py (update_metric_stats -> update_stats_tables -> update_metrics_xlsx).
"""

import sys

from utils.config import load_train_config_dict
from utils.train import ArtifactManager
from utils.utils import load_json, paths


def regen_campaign(campaign, cfg_stats):
    ArtifactManager.dpath_campaign = paths["artifacts"] / campaign
    metadata = load_json(ArtifactManager.dpath_campaign / "campaign_metadata.json")
    settings, datasets = metadata["settings"], metadata["datasets"]

    # per (setting, dataset) aggregations; skip combos with no trial dir (update_metric_stats iterdir()s it)
    for setting in settings:
        ArtifactManager.dpath_setting = ArtifactManager.dpath_campaign / "settings" / setting
        for dataset in datasets:
            if (ArtifactManager.dpath_setting / dataset).exists():
                ArtifactManager.dataset = dataset
                ArtifactManager.update_metric_stats(cfg_stats["spread_type"])

    for dataset in datasets:
        ArtifactManager.dataset = dataset
        ArtifactManager.update_stats_tables(cfg_stats["table_eval_group"], cfg_stats["spread_type"])
    ArtifactManager.update_metrics_xlsx(
        cfg_stats["table_eval_group"],
        cfg_stats["spread_type"],
        cfg_stats["xlsx"]["bold_high"],
        cfg_stats["xlsx"]["ordered"],
        cfg_stats["xlsx"]["heatmap"],
    )


def main():
    args = sys.argv[1:]
    if len(args) != 1:
        sys.exit("usage: python -m tools.regen_stats <campaign>")
    regen_campaign(args[0], load_train_config_dict()["stats"])


if __name__ == "__main__":
    main()
