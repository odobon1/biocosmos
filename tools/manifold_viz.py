"""
Re-render a trial's manifold-viz plots from its cached projections -- no train/eval rerun.
Reads each eval's projections.npz under <trial_dir>/evals/ and regenerates the plots using the
CURRENT config/manifold_viz.yaml (so edits to colors/bg_color/eval_duration/n_stoch_layers/ema_tau
take effect; the cached t-SNE/PCA coords are reused, so perplexity/n_iter cannot change here). Which
panel groups are generated follows train.yaml's dev.manifold_viz.plot_2panel/plot_4panel/plot_7panel.

python -m tools.manifold_viz <campaign>/settings/<setting>/<dataset>/<seed>
python -m tools.manifold_viz <campaign>/settings/<setting>/<dataset>/<seed> evo_only

<campaign>/settings/<setting>/<dataset>/<seed> e.g. dev/settings/hp/lepid/43 (resolved under artifacts/)
evo_only    re-render only the cross-eval evolution GIFs (per-eval plots left as-is)
no_evo      render only the per-eval plots, skip the cross-eval evolution GIFs
snapshot    use the campaign's frozen config snapshot (cfg_baseline.json under artifacts/<campaign>/,
            reading its manifold_viz + train.dev.manifold_viz fields) instead of the live config/*.yaml --
            used by the campaign render worker
"""

from pathlib import Path
import sys

from utils.config import load_manifold_viz_config_dict, load_train_config_dict
from utils.manifold_viz import render_eval, render_evolution, VizContext, _ordered_eval_dirs
from utils.utils import load_json, paths


def _viz_context(dpath_trial):
    # dataset/split from the trial metadata, setting from the path (<campaign>/settings/<setting>/<dataset>/<seed>).
    # Training manifold viz is only produced for eval-enabled trials (train_pt="train" -> eval_type="val").
    meta = load_json(dpath_trial / "trial_metadata.json")
    return VizContext(
        setting=dpath_trial.parent.parent.name,
        dataset=meta["dataset"],
        split=meta["split"],
        eval_type="val",
    )

def render_trial(dpath_trial, evo_only=False, skip_evo=False, cfg_manifold_viz=None, plot_flags=None):
    dpath_evals = dpath_trial / "evals"
    if cfg_manifold_viz is None:
        cfg_manifold_viz = load_manifold_viz_config_dict()
    if plot_flags is None:
        plot_flags = load_train_config_dict()["dev"]["manifold_viz"]
    viz_context = _viz_context(dpath_trial)

    if not evo_only:
        for d in _ordered_eval_dirs(dpath_evals):
            render_eval(dpath_evals, d.name, cfg_manifold_viz, viz_context, plot_flags)
    if not skip_evo:
        render_evolution(dpath_evals, dpath_trial / "viz", cfg_manifold_viz, viz_context, plot_flags)

    # pooled shared-frame plots: one t-SNE/PCA fit over all thresholds pooled, each threshold rendered as
    # a masked subset of the single layout (no orientation), under viz_pooled/. Present only when the
    # end-of-trial pooled compute wrote projections_pooled.npz (dev.manifold_viz.pooled.enabled).
    if plot_flags["pooled"]["enabled"]:
        if not evo_only:
            for d in _ordered_eval_dirs(dpath_evals, "projections_pooled.npz"):
                render_eval(dpath_evals, d.name, cfg_manifold_viz, viz_context, plot_flags, orient=False, fname="projections_pooled.npz")
        if not skip_evo:
            render_evolution(dpath_evals, dpath_trial / "viz_pooled", cfg_manifold_viz, viz_context, plot_flags, orient=False, fname="projections_pooled.npz")

def main():
    args = sys.argv[1:]
    evo_only = "evo_only" in args
    if evo_only:
        args.remove("evo_only")
    skip_evo = "no_evo" in args
    if skip_evo:
        args.remove("no_evo")
    snapshot = "snapshot" in args
    if snapshot:
        args.remove("snapshot")
    if len(args) != 1:
        sys.exit("usage: python -m tools.manifold_viz <campaign>/settings/<setting>/<dataset>/<seed> [evo_only|no_evo] [snapshot]")
    trial_rel = args[0]
    cfg_manifold_viz = plot_flags = None
    if snapshot:
        dpath_campaign = paths["artifacts"] / Path(trial_rel).parts[0]
        cfg_snapshot = load_json(dpath_campaign / "cfg_baseline.json")
        cfg_manifold_viz = cfg_snapshot["manifold_viz"]
        plot_flags = cfg_snapshot["train"]["dev"]["manifold_viz"]
    render_trial(paths["artifacts"] / trial_rel, evo_only, skip_evo, cfg_manifold_viz, plot_flags)


if __name__ == "__main__":
    main()
