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
python -m tools.regen_manif_viz <campaign>/<phase>/_datasets/<dataset>/_arms/<arm>/_coords/<coord>/_seeds/<seed> [evo_only|no_evo] [snapshot]

<campaign>  e.g. dev40 -- re-render every trial in the campaign, phase by phase (_screen/, and qual/ when it
            exists) from each phase's phase_metadata.json matrix x seeds; trials that never ran are skipped
<campaign>/<phase>/_datasets/<dataset>/_arms/<arm>/_coords/<coord>/_seeds/<seed>  e.g.
            dev40/_screen/_datasets/cub/_arms/mp/_coords/LR-1.0e-5/_seeds/42 -- one trial. This is the form the campaign
            render worker spawns per completed trial.
evo_only    re-render only the cross-eval evolving GIFs (per-eval plots left as-is)
no_evo      render only the per-eval plots, skip the cross-eval evolving GIFs
snapshot    use the campaign's frozen config snapshot (cfg_baseline.json under artifacts/<campaign>/<phase>/ --
            each phase carries a copy), reading its manif_viz field, instead of the live config/manif_viz.yaml --
            used by the campaign render worker. Omit it to pick up edits to config/manif_viz.yaml, which is the
            point of re-rendering by hand.
"""

from pathlib import Path
import sys

from utils.config import load_manif_viz_config_dict
from utils.manif_viz import (compute_umap_projections, compute_umap_pooled, render_eval,
                             render_evolution, VizContext, _ordered_eval_dirs)
from utils.utils import load_json, paths


def _viz_context(dpath_trial):
    # dataset/split from the trial metadata, arm/coord from the path
    # (<campaign>/<phase>/_datasets/<dataset>/_arms/<arm>/_coords/<coord>/_seeds/<seed>).
    # Training manifold viz is only produced for eval-enabled trials (train_pt="train").
    meta = load_json(dpath_trial / "trial_metadata.json")
    return VizContext(
        arm=dpath_trial.parents[3].name,
        coord=dpath_trial.parents[1].name,
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
    """Re-render every trial in a campaign, sweeping each phase's planned matrix from its phase_metadata.json
    the way the other regen_* tools do. Trials that never ran (or never reached an eval) have no
    trial_metadata.json and are skipped rather than erroring, so this works on a partially-run campaign."""
    for phase in ("_screen", "qual"):  # the trainval phase runs no evals: nothing to select, aggregate or render
        dpath_phase = paths["artifacts"] / campaign / phase
        if not dpath_phase.exists():  # no qual phase: n_trials_qual null, or not reached yet
            continue
        metadata = load_json(dpath_phase / "phase_metadata.json")
        for dataset, arms in metadata["matrix"].items():
            for arm, coords in arms.items():
                for coord in coords:
                    for seed in metadata["seeds"]:
                        dpath_trial = dpath_phase / "_datasets" / dataset / "_arms" / arm / "_coords" / coord / "_seeds" / str(seed)
                        if not (dpath_trial / "trial_metadata.json").exists():
                            continue
                        # a campaign sweep is a long foreground job, so it reports per trial -- unlike the
                        # single-trial invocation, which the render worker runs detached alongside the next
                        # trial's progress bar and so stays silent
                        print(f"{phase}/{dataset}/{arm}/{coord}/{seed}", flush=True)
                        render_trial(dpath_trial, evo_only, skip_evo, cfg_manif_viz)

def main():
    args = sys.argv[1:]
    flags = {a for a in args if a in ("evo_only", "no_evo", "snapshot")}
    targets = [a for a in args if a not in flags]
    if len(targets) != 1:
        sys.exit("usage: python -m tools.regen_manif_viz <campaign>[/<phase>/_datasets/<dataset>/_arms/<arm>/_coords/<coord>/_seeds/<seed>] "
                 "[evo_only|no_evo] [snapshot]")
    target = targets[0].strip("/")
    evo_only, skip_evo = "evo_only" in flags, "no_evo" in flags
    cfg_manif_viz = None
    if "/" in target:  # a trial path; a bare name is the whole campaign
        if "snapshot" in flags:
            campaign, phase = Path(target).parts[:2]
            cfg_manif_viz = load_json(paths["artifacts"] / campaign / phase / "cfg_baseline.json")["manif_viz"]
        render_trial(paths["artifacts"] / target, evo_only, skip_evo, cfg_manif_viz)
    else:
        if "snapshot" in flags:
            cfg_manif_viz = load_json(paths["artifacts"] / target / "_screen" / "cfg_baseline.json")["manif_viz"]
        render_campaign(target, evo_only, skip_evo, cfg_manif_viz)


if __name__ == "__main__":
    main()
