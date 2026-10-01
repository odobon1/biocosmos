"""
Re-render a campaign's (or one trial's) manifold-viz plots from cached projections -- no train/eval rerun.
Reads each eval's projections.npz under <trial_dir>/eval/all/ and regenerates the plots using the
CURRENT config/{trial,render}/manifold_viz.yaml (so edits to colors/bg_color/gif.frame_dur/orient.ema_tau
and to the plot_2panel/plot_7panel/plot_8panel group toggles take effect; the cached PCA/t-SNE/UMAP/UMAP-sphere
coords are reused, so tsne.* / umap.* cannot change here -- delete the cached coords to refit).

This is also where BOTH UMAP variants are FIT (flat + spherical, from each eval's cached embs.npz) on
the first pass, since they have no sharded GPU implementation and this process is CPU-only; PCA and
t-SNE are computed in the training loop. The fits are skipped once their coords are cached.

And it is where `manifold_viz.store_cache: false` takes effect: a FULL render's last step deletes the
trial's viz/cache/ dirs (a partial render -- evo_only/no_evo -- leaves them for the half it skipped),
so such a trial cannot be re-rendered or refit afterwards -- the caches it would read are gone.

A test trial (test.py's, under _phase/test/) has one eval, its eval/sel/, cached under eval/sel/viz/cache/: its UMAPs
are fit and its stills drawn there (render_test_trial) -- no evolving GIFs and no pooled frame, a single eval having
no sequence -- and the arm's test best_coord/ mirror, which copies the stills, is rebuilt. test.py runs this itself at
the end of a run for the trials it scored; re-running it by hand re-renders under edited render settings.

python -m tools.regen_manifold_viz <campaign>
python -m tools.regen_manifold_viz <campaign>/_phase/<phase>/_dataset/<dataset>/_arm/<arm>/_coord/<coord>/_trial/<n> [evo_only|no_evo] [snapshot]

<campaign>  e.g. dev40 -- re-render every trial in the campaign, phase by phase (screen/, and refine/ and test/ when
            they exist) from each phase's phase_metadata.json matrix x seeds; trials that never ran (or, in test/,
            aren't scored yet) are skipped
<campaign>/_phase/<phase>/_dataset/<dataset>/_arm/<arm>/_coord/<coord>/_trial/<n>  e.g.
            dev40/_phase/screen/_dataset/cub/_arm/mp/_coord/LR-1e-5/_trial/1 -- one trial. This is the form the campaign
            render worker spawns per completed trial.
evo_only    re-render only the cross-eval evolving GIFs (per-eval plots left as-is; test trials, which have none, skipped)
no_evo      render only the per-eval plots, skip the cross-eval evolving GIFs
snapshot    use the campaign's frozen config snapshot (cfg_baseline.json under artifacts/<campaign>/_phase/<phase>/ --
            each phase carries a copy), reading its manifold_viz field, instead of the live config/{trial,render}/manifold_viz.yaml --
            used by the campaign render worker. Omit it to pick up edits to config/{trial,render}/manifold_viz.yaml, which is the
            point of re-rendering by hand.
"""

from pathlib import Path
import shutil
import sys

from utils.config import load_manifold_viz_config_dict, load_manifold_viz_render_config_dict
from utils.manifold_viz import (compute_umap_eval, compute_umap_projections, compute_umap_pooled, render_eval,
                             render_evolution, render_test_eval, VizContext, _ordered_eval_dirs)
from utils.report import coord_complete, update_best_coord, update_test_best_coord
from utils.train import ArtifactManager, arm_lock, copy_viz, dpath_eval_seq, dpath_viz_cache, dpath_viz_dyn, dpath_viz_plots
from utils.utils import load_json, paths


def _viz_context(dpath_trial):
    # dataset/split from the trial metadata, arm/coord from the path
    # (<campaign>/_phase/<phase>/_dataset/<dataset>/_arm/<arm>/_coord/<coord>/_trial/<n>).
    # Training manifold viz is only produced for eval-enabled trials (train_pt="train").
    meta = load_json(dpath_trial / "trial_metadata.json")
    return VizContext(
        arm=dpath_trial.parents[3].name,
        coord=dpath_trial.parents[1].name,
        dataset=meta["dataset"],
        split=meta["split"],
        eval_pt="val",
    )

def trial_viz_cached(dpath_trial):
    """Whether a test trial ran: test.py caches every trial's projections under eval/sel/viz/cache/ (before its score
    files, so a scored trial's cache is complete). A store_cache: false trial's cache is deleted after its render, so
    it reads as un-run here and the campaign sweep skips it -- there is nothing left to re-render from anyway."""
    return (dpath_viz_cache(dpath_trial / "eval" / "sel") / "projections.npz").exists()

def render_test_trial(dpath_trial, cfg_manifold_viz=None):
    """A test trial (<campaign>/_phase/test/_dataset/<dataset>/_arm/<arm>/_coord/<coord>/_trial/<n>): fit the UMAPs
    of its one eval (eval/sel/, from its cached embs.npz -- idempotent, as render_trial's fits) and draw its stills
    under eval/sel/viz/vanilla/, titled by the trainval checkpoint the model was saved at (the coord's chkpt_stop).
    No evolving GIFs or pooled frame: a single eval has no sequence. The arm's test best_coord/ mirror copies the
    stills, so it is rebuilt after."""
    if cfg_manifold_viz is None:
        cfg_manifold_viz = load_manifold_viz_config_dict()
    dpath_eval = dpath_trial / "eval" / "sel"
    dpath_phase = dpath_trial.parents[7]
    # the split from the phase's frozen config snapshot: the coord's config.json copy prunes split.split (an inert
    # param there, train.ArtifactManager.save_metadata_coord), and its chkpt_stop names the eval in the titles
    split = load_json(dpath_phase / "cfg_baseline.json")["train"]["split"]["split"]
    chkpt_stop = load_json(dpath_trial.parents[1] / "config.json")["chkpt_stop"]
    viz_context = VizContext(
        arm=dpath_trial.parents[3].name,
        coord=dpath_trial.parents[1].name,
        dataset=dpath_trial.parents[5].name,
        split=split,
        eval_pt="test",
    )
    compute_umap_eval(dpath_eval, cfg_manifold_viz["umap"])
    render_test_eval(dpath_eval, cfg_manifold_viz, viz_context, f"Chkpt {chkpt_stop}")
    ArtifactManager.dpath_phase = dpath_phase
    update_test_best_coord(dpath_trial.parents[5].name, dpath_trial.parents[3].name)

    # delete-after-use, as in render_trial: the stills are drawn and the mirror rebuilt, so the lone
    # eval's cache has no consumer left (a test trial has no partial-render modes)
    dpath_cache = dpath_viz_cache(dpath_eval)
    if not cfg_manifold_viz["store_cache"] and dpath_cache.exists():
        shutil.rmtree(dpath_cache)

def render_trial(dpath_trial, evo_only=False, skip_evo=False, cfg_manifold_viz=None):
    dpath_evals = dpath_eval_seq(dpath_trial)
    if cfg_manifold_viz is None:
        cfg_manifold_viz = load_manifold_viz_config_dict()
    viz_context = _viz_context(dpath_trial)
    # (<campaign>/_phase/<phase>/_dataset/<dataset>/_arm/<arm>/_coord/<coord>/_trial/<n>)
    ArtifactManager.dpath_phase = dpath_trial.parents[7]
    # the titles mark the trial's own-best eval at the chkpt recorded now (None until the coord has selected: a hand
    # render mid-trial) and its selected eval only once the coord's pick is final (every planned trial complete) --
    # the pick moves as trials land, and a mark baked earlier would go stale; the earlier trials pick the mark up on a
    # re-render (the campaign sweep)
    chkpt = load_json(dpath_trial / "trial_metadata.json")["chkpt"]
    if not coord_complete(dpath_trial.parents[5].name, dpath_trial.parents[3].name, dpath_trial.parents[1].name):
        chkpt = {**chkpt, "sel": None}

    # Both UMAP variants are fit here, on CPU, rather than in the training loop: neither has a sharded
    # GPU implementation, and this process is CUDA-free (detached, when the render worker spawns it).
    # Idempotent, so a re-render reuses the cached layouts instead of refitting.
    compute_umap_projections(dpath_evals, cfg_manifold_viz)
    if cfg_manifold_viz["pooled"]["enabled"]:
        compute_umap_pooled(dpath_evals, cfg_manifold_viz)

    if not evo_only:
        for d in _ordered_eval_dirs(dpath_evals):
            render_eval(dpath_evals, d.name, cfg_manifold_viz, viz_context, chkpt)
    if not skip_evo:
        render_evolution(dpath_evals, dpath_viz_dyn(dpath_trial, pooled=False), cfg_manifold_viz, viz_context, chkpt)

    # pooled shared-frame plots: one t-SNE/PCA fit over all thresholds pooled, each threshold rendered as
    # a masked subset of the single layout (no orientation), under viz/pooled/ (GIFs: viz_dyn/pooled/). Present only when the
    # end-of-trial pooled compute wrote projections_pooled.npz (manifold_viz.pooled.enabled).
    if cfg_manifold_viz["pooled"]["enabled"]:
        if not evo_only:
            for d in _ordered_eval_dirs(dpath_evals, "projections_pooled.npz"):
                render_eval(dpath_evals, d.name, cfg_manifold_viz, viz_context, chkpt, orient=False, fname="projections_pooled.npz")
        if not skip_evo:
            render_evolution(dpath_evals, dpath_viz_dyn(dpath_trial, pooled=True), cfg_manifold_viz, viz_context, chkpt, orient=False, fname="projections_pooled.npz")

    # the trial's selected / own-best evals carry copies of their eval's viz/ stills (eval/{sel,best}/viz/); the
    # selection is written at trial end, before this (detached) render has drawn them, so they're re-copied here, at
    # the evals trial_metadata.json's chkpt records (None until the coord has selected: a hand render mid-trial) --
    # report.update_chkpt_selection re-copies them whenever a later trial of the coord moves the selection. Under the
    # arm's lock, which that reselection (in the next trial's process, overlapping this render) also holds, so the
    # chkpt read here is never one it is midway through moving
    # (<phase>/_dataset/<dataset>/_arm/<arm>/_coord/<coord>/_trial/<n>)
    dpath_arm = dpath_trial.parents[3]
    with arm_lock(dpath_arm):
        chkpt = load_json(dpath_trial / "trial_metadata.json")["chkpt"]
        for name in ("sel", "best"):
            if chkpt[name] is not None:
                copy_viz(dpath_evals / str(chkpt[name]), dpath_trial / "eval" / name)
    # the arm's best_coord/ mirror copies those eval/{sel,best}/ dirs, and the trial end's rebuild of it predates
    # the stills: rebuilt now (a no-op unless the arm is complete)
    update_best_coord(dpath_trial.parents[5].name, dpath_arm.name)

    # delete-after-use: this render is the caches' last consumer, so a campaign that doesn't keep them
    # (store_cache false) drops them here -- only after a FULL render, since a partial one (evo_only /
    # no_evo) still needs them for the half it skipped
    if not (evo_only or skip_evo) and not cfg_manifold_viz["store_cache"]:
        for d in dpath_evals.iterdir():
            dpath_cache = dpath_viz_cache(d)
            if dpath_cache.exists():
                shutil.rmtree(dpath_cache)

def render_campaign(campaign, evo_only=False, skip_evo=False, cfg_manifold_viz=None):
    """Re-render every trial in a campaign, sweeping each phase's planned matrix from its phase_metadata.json
    the way the other regen_* tools do. Trials that never ran (or never reached an eval) have no
    trial_metadata.json and are skipped rather than erroring, so this works on a partially-run campaign. The test
    tree's trials (render_test_trial) are swept the same way, those not scored yet (no cache) skipped -- and all of
    them under evo_only, having no evolving GIFs."""
    for phase in ("screen", "refine", "test"):  # the trainval phase runs no evals: nothing to select, aggregate or render
        dpath_phase = paths["artifacts"] / campaign / "_phase" / phase
        if not dpath_phase.exists() or (phase == "test" and evo_only):  # no refine phase: n_trials_refine null, or not reached yet; no test run yet
            continue
        metadata = load_json(dpath_phase / "phase_metadata.json")
        for dataset, arms in metadata["matrix"].items():
            for arm, coords in arms.items():
                for coord in coords:
                    for trial_num in range(1, len(metadata["seeds"]) + 1):
                        dpath_trial = dpath_phase / "_dataset" / dataset / "_arm" / arm / "_coord" / coord / "_trial" / str(trial_num)
                        if not (trial_viz_cached(dpath_trial) if phase == "test" else (dpath_trial / "trial_metadata.json").exists()):
                            continue
                        # a campaign sweep is a long foreground job, so it reports per trial -- unlike the
                        # single-trial invocation, which the render worker runs detached alongside the next
                        # trial's progress bar and so stays silent
                        print(f"{phase}/{dataset}/{arm}/{coord}/{trial_num}", flush=True)
                        if phase == "test":
                            render_test_trial(dpath_trial, cfg_manifold_viz)
                        else:
                            render_trial(dpath_trial, evo_only, skip_evo, cfg_manifold_viz)

def main():
    args = sys.argv[1:]
    flags = {a for a in args if a in ("evo_only", "no_evo", "snapshot")}
    targets = [a for a in args if a not in flags]
    if len(targets) != 1:
        sys.exit("usage: python -m tools.regen_manifold_viz <campaign>[/_phase/<phase>/_dataset/<dataset>/_arm/<arm>/_coord/<coord>/_trial/<n>] "
                 "[evo_only|no_evo] [snapshot]")
    target = targets[0].strip("/")
    evo_only, skip_evo = "evo_only" in flags, "no_evo" in flags
    cfg_manifold_viz = None
    if "/" in target:  # a trial path; a bare name is the whole campaign
        campaign, _, phase = Path(target).parts[:3]
        if "snapshot" in flags:
            cfg_manifold_viz = load_json(paths["artifacts"] / campaign / "_phase" / phase / "cfg_baseline.json")["manifold_viz"]
            cfg_manifold_viz = {**cfg_manifold_viz, **load_manifold_viz_render_config_dict()}
        if phase == "test":
            render_test_trial(paths["artifacts"] / target, cfg_manifold_viz)
        else:
            render_trial(paths["artifacts"] / target, evo_only, skip_evo, cfg_manifold_viz)
    else:
        if "snapshot" in flags:
            cfg_manifold_viz = load_json(paths["artifacts"] / target / "_phase" / "screen" / "cfg_baseline.json")["manifold_viz"]
            cfg_manifold_viz = {**cfg_manifold_viz, **load_manifold_viz_render_config_dict()}
        render_campaign(target, evo_only, skip_evo, cfg_manifold_viz)


if __name__ == "__main__":
    main()
