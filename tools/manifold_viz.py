"""
Re-render a trial's manifold-viz plots from its cached projections -- no train/eval rerun.
Reads each eval's projections.npz under <trial_dir>/evals/ and regenerates the plots using the
CURRENT config/manifold_viz.yaml (so edits to colors/bg_color/eval_duration/n_stoch_layers/ema_tau
take effect; the cached t-SNE/PCA coords are reused, so perplexity/n_iter cannot change here). Which
panel groups are generated follows train.yaml's dev.manifold_viz.plot_2panel/plot_4panel/plot_7panel.

python -m tools.manifold_viz <campaign/setting/dataset/seed>
python -m tools.manifold_viz <campaign/setting/dataset/seed> evo_only

<campaign/setting/dataset/seed> e.g. dev/hp/lepid/43 (resolved under artifacts/)
evo_only    re-render only the cross-eval evolution GIFs (per-eval plots left as-is)
"""

import sys

from utils.config import load_manifold_viz_config_dict, load_train_config_dict
from utils.manifold_viz import render_eval, render_evolution, VizContext, _ordered_eval_dirs
from utils.utils import load_json, paths


def _viz_context(dpath_trial):
    # dataset/split from the trial metadata, setting from the path (campaign/setting/dataset/seed).
    # Training manifold viz is only produced for eval-enabled trials (train_pt="train" -> eval_type="val").
    meta = load_json(dpath_trial / "trial_metadata.json")
    return VizContext(
        setting=dpath_trial.parent.parent.name,
        dataset=meta["dataset"],
        split=meta["split"],
        eval_type="val",
    )

def render_trial(dpath_trial, evo_only=False):
    dpath_evals = dpath_trial / "evals"
    cfg_manifold_viz = load_manifold_viz_config_dict()
    plot_flags = load_train_config_dict()["dev"]["manifold_viz"]
    viz_context = _viz_context(dpath_trial)

    if not evo_only:
        for d in _ordered_eval_dirs(dpath_evals):
            render_eval(dpath_evals, d.name, cfg_manifold_viz, viz_context, plot_flags)
    render_evolution(dpath_evals, dpath_trial / "viz", cfg_manifold_viz, viz_context, plot_flags)

def main():
    args = sys.argv[1:]
    evo_only = "evo_only" in args
    if evo_only:
        args.remove("evo_only")
    if len(args) != 1:
        sys.exit("usage: python -m tools.manifold_viz <campaign/setting/dataset/seed> [evo_only]")
    render_trial(paths["artifacts"] / args[0], evo_only)


if __name__ == "__main__":
    main()
