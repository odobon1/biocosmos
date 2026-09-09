"""
torchrun --standalone --nproc-per-node=auto -m test <campaign>

Test-partition evaluation of a campaign's trainval models -- the campaign's final scores. Requires the
campaign to have a trainval phase on disk (artifacts/<campaign>/trainval/); errors otherwise. Every model
saved there (the phase's matrix x seeds, each trial's model.pt) is rebuilt from the campaign's frozen
config snapshot + its coord's recorded overrides, its weights loaded, and evaluated once on the split's
TEST partitions (test_id/test_ood, EvaluationPipeline with eval_pt="test"; n-shot buckets from the
split's "trainval/test" view -- test_id classes by their trainval shot counts). Per-trial scores land as
artifacts/<campaign>/test/_datasets/<dataset>/_arms/<arm>/_coords/<coord>/_seeds/<seed>/<group>.json
({'chkpt': the saved checkpoint index, 'scores': the group's scores}); trials whose score files are all
present are skipped, so a relaunch resumes (and a fully-scored campaign just re-renders the tables).
The run ends with the test workbooks, one per eval group at artifacts/<campaign>/test/map/test_<group>.xlsx
(report.update_test_stats), styled per the live config/stats.yaml.
"""

import shutil
import sys

import torch
import torch.distributed as dist

from campaign_runner import _build_trial_cfg_dict
from utils.config import get_config_stats, get_config_train
from utils.data import stage_img_cache
from utils.ddp import setup_ddp, cleanup_ddp, rank0
from utils.eval import EvaluationPipeline
from utils.hardware import apply_backend_flags
from utils.report import _EVAL_GROUPS, update_test_stats
from utils.train import ArtifactManager, format_scores
from utils.utils import get_text_template, load_json, save_json, paths

import pdb


def _dpath_coord(dpath_phase, dataset, arm, coord):
    return dpath_phase / "_datasets" / dataset / "_arms" / arm / "_coords" / coord

def _plan(campaign):
    """(cfg_snapshot, metadata, combos): the trainval phase's frozen config snapshot, its
    phase_metadata.json, and its planned (dataset, arm, coord) combos in campaign order. Raises
    when the campaign has no trainval phase on disk -- there are then no models to test."""
    dpath_trainval = paths["artifacts"] / campaign / "trainval"
    if not dpath_trainval.exists():
        raise FileNotFoundError(
            f"{dpath_trainval} does not exist -- test evaluates the trainval phase's saved models; run the "
            f"campaign through its trainval phase first (camp config trainval: true)."
        )
    cfg_snapshot = load_json(dpath_trainval / "cfg_baseline.json")
    metadata = load_json(dpath_trainval / "phase_metadata.json")
    combos = [(dataset, arm, coord)
              for dataset, arms in metadata["matrix"].items()
              for arm, coords in arms.items()
              for coord in coords]
    return cfg_snapshot, metadata, combos

def _pending(dpath_test, combos, seeds):
    """{(dataset, arm, coord): [seeds]} of the trials still to score: those whose test score files
    (_seeds/<seed>/<group>.json, all eval groups) aren't all on disk. Computed once up front, before
    anything is written, so every rank derives the identical eval sequence (evaluate() is collective)."""
    pending = {}
    for dataset, arm, coord in combos:
        for seed in seeds:
            dpath_scores = _dpath_coord(dpath_test, dataset, arm, coord) / "_seeds" / str(seed)
            if not all((dpath_scores / f"{group_key}.json").exists() for group_key in _EVAL_GROUPS):
                pending.setdefault((dataset, arm, coord), []).append(seed)
    return pending

@rank0
def _seed_test_tree(dpath_test, dpath_trainval, metadata, combos):
    """Make the test tree self-contained for table rendering: phase_metadata.json (the trainval
    phase's planned matrix, which update_test_stats keys its rows off, plus its seeds) plus every planned
    coord's config.json + overrides.json copied over from the trainval tree (the overrides bands read
    them). Refreshed each run, so a matrix grown by a campaign relaunch carries over."""
    dpath_test.mkdir(parents=True, exist_ok=True)
    save_json({key: metadata[key] for key in ("seeds", "matrix")},
              dpath_test / "phase_metadata.json")
    for dataset, arm, coord in combos:
        dpath_dst = _dpath_coord(dpath_test, dataset, arm, coord)
        dpath_dst.mkdir(parents=True, exist_ok=True)
        for fname in ("config.json", "overrides.json"):
            shutil.copyfile(_dpath_coord(dpath_trainval, dataset, arm, coord) / fname, dpath_dst / fname)

@rank0
def _save_test_scores(dpath_scores, eval_metrics, chkpt):
    """The trial's test score files, one per eval group: {'chkpt': the checkpoint index the trainval
    model was saved at, 'scores': the group's scores} -- the shape report._collect_test_scores reads.
    All groups are written together, so any one file's presence marks the trial scored."""
    dpath_scores.mkdir(parents=True, exist_ok=True)
    formatted = format_scores(eval_metrics["scores"])
    for group_key, scores_grp in formatted.items():
        save_json({"chkpt": chkpt, "scores": scores_grp}, dpath_scores / f"{group_key}.json")

def main():
    from models import VLMWrapper  # local: models pulls open_clip/transformers, too heavy for module import
    from utils.loss import configure_phylo_targs  # local: utils.loss pulls torch/biopython (same reason)

    if len(sys.argv) != 2:
        raise SystemExit("usage: torchrun --standalone --nproc-per-node=auto -m test <campaign>")
    campaign = sys.argv[1]
    dpath_trainval = paths["artifacts"] / campaign / "trainval"
    dpath_test = paths["artifacts"] / campaign / "test"

    cfg_snapshot, metadata, combos = _plan(campaign)
    seeds = metadata["seeds"]
    pending = _pending(dpath_test, combos, seeds)

    # fail fast, before any GPU work: every still-unscored trial needs its trainval model on disk -- a
    # missing one means the trainval phase is incomplete (relaunch the campaign to finish it)
    missing = [f"{dataset}/{arm}/{coord}/{seed}"
               for (dataset, arm, coord), combo_seeds in pending.items()
               for seed in combo_seeds
               if not (_dpath_coord(dpath_trainval, dataset, arm, coord) / "_seeds" / str(seed) / "model.pt").exists()]
    if missing:
        raise FileNotFoundError(
            f"no trainval model.pt for {len(missing)} trial(s) of campaign '{campaign}': {missing} -- "
            f"the trainval phase is incomplete; relaunch the campaign to finish it before testing."
        )

    # every pending combo's effective config, built now so a bad one errors at kickoff (mirrors
    # _run_phase's fail-fast validation); the combo's first pending seed stands in for cfg.seed,
    # which nothing on the eval path reads
    cfgs = {}
    for (dataset, arm, coord), combo_seeds in pending.items():
        dpath_coord_tv = _dpath_coord(dpath_trainval, dataset, arm, coord)
        overrides = load_json(dpath_coord_tv / "overrides.json")
        chkpt_stop = load_json(dpath_coord_tv / "config.json")["chkpt_stop"]
        cfg_dict = _build_trial_cfg_dict(
            cfg_snapshot, campaign, "trainval", arm, coord, {**overrides["arm"], **overrides["coord"]},
            combo_seeds[0], dataset, seeds.index(combo_seeds[0]),
            injections={"train_pt": "trainval", "chkpt_stop": chkpt_stop},
        )
        cfgs[(dataset, arm, coord)] = get_config_train(cfg_dict=cfg_dict)

    _, device = setup_ddp()
    _seed_test_tree(dpath_test, dpath_trainval, metadata, combos)

    n_evals = sum(len(combo_seeds) for combo_seeds in pending.values())
    n_scored = len(combos) * len(seeds) - n_evals
    if dist.get_rank() == 0:
        print(f"Campaign: '{campaign}' test ({n_evals} evals" + (f"; {n_scored} already scored)" if n_scored else ")"),
              flush=True)

    # node-local image-cache staging (single node under --standalone); flock-guarded, so every rank
    # calls it -- the first stages while the rest block, then take the fast path
    for dataset in sorted({dataset for (dataset, _, _), cfg in cfgs.items() if cfg.use_img_cache}):
        stage_img_cache(dataset)

    idx_eval = 0
    for combo in combos:
        if combo not in pending:
            continue
        dataset, arm, coord = combo
        cfg = cfgs[combo]
        cfg.device = device
        apply_backend_flags(cfg.hw)
        # the wrapper resolves a pos_prevalence bias init at build (overwritten by the checkpoint below),
        # which under a phylo target needs the run's phylo-target params set, as in train.py
        configure_phylo_targs(cfg.split, cfg.train_pt, cfg.batch_size,
                              cfg.htarg["kernel"], cfg.htarg["exp"]["beta"], cfg.htarg["shuffle"], cfg.seed)
        modelw = VLMWrapper.build(cfg, verbose=(dist.get_rank() == 0))
        text_template_eval = get_text_template(cfg.text_template["eval"], dataset=dataset)
        eval_pipe = EvaluationPipeline(cfg, text_template_eval, modelw.img_pp_inf, eval_pt="test")
        for seed in pending[combo]:
            idx_eval += 1
            dpath_trial_tv = _dpath_coord(dpath_trainval, dataset, arm, coord) / "_seeds" / str(seed)
            state = torch.load(dpath_trial_tv / "model.pt", map_location="cpu", weights_only=True)
            modelw._unwrapped_model.load_state_dict(state)
            eval_metrics, time_eval, _ = eval_pipe.evaluate(modelw, loss_flag=False)
            _save_test_scores(_dpath_coord(dpath_test, dataset, arm, coord) / "_seeds" / str(seed),
                              eval_metrics, cfg.chkpt_stop)
            if dist.get_rank() == 0:
                score = float(eval_metrics["scores"]["native"]["comp"]["map"]["all"])
                print(f"[{idx_eval}/{n_evals}] {dataset}/{arm}/{coord}/{seed} (chkpt {cfg.chkpt_stop}): "
                      f"native comp mAP {score:.4f} ({time_eval:.0f}s)", flush=True)

    dist.barrier()  # every trial's score files on disk before rank 0 renders the tables
    ArtifactManager.dpath_phase = dpath_test
    cfg_stats = get_config_stats()
    update_test_stats(cfg_stats.spread_type, cfg_stats.bold_high, cfg_stats.ordered, cfg_stats.heatmap,
                      cfg_stats.supp_scores, cfg_stats.overrides)
    cleanup_ddp()


if __name__ == "__main__":
    main()
