"""
Re-render a campaign's (or one trial's) manifold-viz plots from cached projections -- no train/eval rerun.
Reads each eval's projections.npz under <trial_dir>/evals/ and regenerates the plots using the
CURRENT config/manif_viz.yaml (so edits to colors/bg_color/eval_duration/orient.ema_tau
and to the plot_2panel/plot_7panel/plot_8panel group toggles take effect; the cached PCA/t-SNE/UMAP/UMAP-sphere
coords are reused, so tsne.* / umap.* cannot change here -- delete the cached coords to refit).

This is also where BOTH UMAP variants are FIT (flat + spherical, from each eval's cached embs.npz) on
the first pass, since they have no sharded GPU implementation and this process is CPU-only; PCA and
t-SNE are computed in the training loop. The fits are skipped once their coords are cached.

python -m tools.regen_manif_viz <campaign>
python -m tools.regen_manif_viz <campaign>/datasets/<dataset>/settings/<setting>/<seed> [evo_only|no_evo] [snapshot]

<campaign>  e.g. dev40 -- re-render every trial in the campaign, from its campaign_metadata.json matrix
            (settings x datasets x seeds); trials that never ran are skipped
<campaign>/datasets/<dataset>/settings/<setting>/<seed>  e.g. dev40/datasets/cub/settings/mp/42 -- one trial. This is the
            form the campaign render worker spawns per completed trial.
evo_only    re-render only the cross-eval evolving GIFs (per-eval plots left as-is)
no_evo      render only the per-eval plots, skip the cross-eval evolving GIFs
snapshot    use the campaign's frozen config snapshot (cfg_baseline.json under artifacts/<campaign>/,
            reading its manif_viz field) instead of the live config/manif_viz.yaml -- used by the
            campaign render worker. Omit it to pick up edits to config/manif_viz.yaml, which is the
            point of re-rendering by hand.
"""

from pathlib import Path
import sys

from utils.config import load_manif_viz_config_dict
from utils.manif_viz import (compute_umap_projections, compute_umap_pooled, render_eval,
                             render_evolution, VizContext, _ordered_eval_dirs)
from utils.utils import load_json, paths


def _viz_context(dpath_trial):
    # dataset/split from the trial metadata, setting from the path (<campaign>/datasets/<dataset>/settings/<setting>/<seed>).
    # Training manifold viz is only produced for eval-enabled trials (train_pt="train").
    meta = load_json(dpath_trial / "trial_metadata.json")
    return VizContext(
        setting=dpath_trial.parent.name,
        dataset=meta["dataset"],
        split=meta["split"],
    )

def render_trial(dpath_trial, evo_only=False, skip_evo=False, cfg_manif_viz=None):
    dpath_evals = dpath_trial / "evals"
    if cfg_manif_viz is None:
        cfg_manif_viz = load_manif_viz_config_dict()
    viz_context = _viz_context(dpath_trial)

    # Both UMAP variants are fit here, on CPU, rather than in the training loop: neither has a sharded
    # GPU implementation, and this process is CUDA-free (detached, when the render worker spawns it).
    # Idempotent, so a re-render reuses the cached layouts instead of refitting.
    compute_umap_projections(dpath_evals, cfg_manif_viz)
    if cfg_manif_viz["pooled"]["enabled"]:
        compute_umap_pooled(dpath_evals, cfg_manif_viz)

    if not evo_only:
        for d in _ordered_eval_dirs(dpath_evals):
            render_eval(dpath_evals, d.name, cfg_manif_viz, viz_context)
    if not skip_evo:
        render_evolution(dpath_evals, dpath_trial / "viz", cfg_manif_viz, viz_context)

    # pooled shared-frame plots: one t-SNE/PCA fit over all thresholds pooled, each threshold rendered as
    # a masked subset of the single layout (no orientation), under viz_pooled/. Present only when the
    # end-of-trial pooled compute wrote projections_pooled.npz (manif_viz.pooled.enabled).
    if cfg_manif_viz["pooled"]["enabled"]:
        if not evo_only:
            for d in _ordered_eval_dirs(dpath_evals, "projections_pooled.npz"):
                render_eval(dpath_evals, d.name, cfg_manif_viz, viz_context, orient=False, fname="projections_pooled.npz")
        if not skip_evo:
            render_evolution(dpath_evals, dpath_trial / "viz_pooled", cfg_manif_viz, viz_context, orient=False, fname="projections_pooled.npz")

def render_campaign(campaign, evo_only=False, skip_evo=False, cfg_manif_viz=None):
    """Re-render every trial in a campaign, sweeping its planned matrix from campaign_metadata.json the
    way the other regen_* tools do. Trials that never ran (or never reached an eval) have no
    trial_metadata.json and are skipped rather than erroring, so this works on a partially-run campaign."""
    dpath_campaign = paths["artifacts"] / campaign
    metadata = load_json(dpath_campaign / "campaign_metadata.json")
    for setting in metadata["settings"]:
        for dataset in metadata["datasets"]:
            for seed in metadata["seeds"]:
                dpath_trial = dpath_campaign / "datasets" / dataset / "settings" / setting / str(seed)
                if not (dpath_trial / "trial_metadata.json").exists():
                    continue
                # a campaign sweep is a long foreground job, so it reports per trial -- unlike the
                # single-trial invocation, which the render worker runs detached alongside the next
                # trial's progress bar and so stays silent
                print(f"{setting}/{dataset}/{seed}", flush=True)
                render_trial(dpath_trial, evo_only, skip_evo, cfg_manif_viz)

def main():
    args = sys.argv[1:]
    flags = {a for a in args if a in ("evo_only", "no_evo", "snapshot")}
    targets = [a for a in args if a not in flags]
    if len(targets) != 1:
        sys.exit("usage: python -m tools.regen_manif_viz <campaign>[/datasets/<dataset>/settings/<setting>/<seed>] "
                 "[evo_only|no_evo] [snapshot]")
    target = targets[0].strip("/")
    evo_only, skip_evo = "evo_only" in flags, "no_evo" in flags
    cfg_manif_viz = None
    if "snapshot" in flags:
        cfg_manif_viz = load_json(paths["artifacts"] / Path(target).parts[0] / "cfg_baseline.json")["manif_viz"]
    if "/" in target:  # a trial path; a bare name is the whole campaign
        render_trial(paths["artifacts"] / target, evo_only, skip_evo, cfg_manif_viz)
    else:
        render_campaign(target, evo_only, skip_evo, cfg_manif_viz)


if __name__ == "__main__":
    main()
