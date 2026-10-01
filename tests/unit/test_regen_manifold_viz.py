import json

from tools import regen_manifold_viz as rmv


def _stub_render(tmp_path, monkeypatch):
    """render_trial with the rendering stubbed and the best_coord/ rebuild recorded: the trial dir, at its real depth
    under a phase dir (render_trial reads its phase, dataset and arm off the path), and the rebuilds' (phase, dataset, arm)."""
    for fn in ("compute_umap_projections", "render_eval", "render_evolution"):
        monkeypatch.setattr(rmv, fn, lambda *a, **k: None)
    monkeypatch.setattr(rmv, "_viz_context", lambda dpath_trial: None)
    monkeypatch.setattr(rmv, "coord_complete", lambda dataset, arm, coord: False)
    rebuilt = []
    monkeypatch.setattr(rmv, "update_best_coord",
                        lambda dataset, arm: rebuilt.append((rmv.ArtifactManager.dpath_phase, dataset, arm)))
    dpath_trial = tmp_path / "screen" / "_dataset" / "cub" / "_arm" / "hp" / "_coord" / "c0" / "_trial" / "1"
    dpath_trial.mkdir(parents=True)
    return dpath_trial, rebuilt


def test_render_trial_copies_viz_to_selected_and_best(tmp_path, monkeypatch) -> None:
    # the selection is written at trial end, before the detached render draws any stills, so render_trial
    # re-copies the eval's viz/ afterwards: sel/viz/ from the eval trial_metadata.json's chkpt records
    # (chkpt 2), best/viz/ from its own (chkpt 1) -- the rendering itself is stubbed, the stills and cache
    # already on disk; the stills come over, the cache stays in eval/all/ -- then rebuilds the arm's best_coord/
    # mirror, which copies them
    dpath_trial, rebuilt = _stub_render(tmp_path, monkeypatch)
    cfg = {"pooled": {"enabled": False}, "store_cache": True}

    dpath_evals = dpath_trial / "eval" / "all"
    for idx in ("1", "2"):
        for sub in ("vanilla/8panel/joint.png", "cache/projections.npz"):
            fpath = dpath_evals / idx / "viz" / sub
            fpath.parent.mkdir(parents=True)
            fpath.write_text(f"eval{idx}")
    (dpath_trial / "trial_metadata.json").write_text(json.dumps({"chkpt": {"sel": 2, "best": 1}}))

    rmv.render_trial(dpath_trial, cfg_manifold_viz=cfg)

    assert (dpath_trial / "eval" / "sel" / "viz" / "vanilla" / "8panel" / "joint.png").read_text() == "eval2"
    assert (dpath_trial / "eval" / "best" / "viz" / "vanilla" / "8panel" / "joint.png").read_text() == "eval1"
    assert not (dpath_trial / "eval" / "sel" / "viz" / "cache").exists()
    assert not (dpath_trial / "eval" / "best" / "viz" / "cache").exists()
    assert (dpath_evals / "1" / "viz" / "cache" / "projections.npz").exists()  # store_cache: the source cache stays
    assert rebuilt == [(tmp_path / "screen", "cub", "hp")]


def test_render_trial_marks_selected_only_once_the_coord_is_complete(tmp_path, monkeypatch) -> None:
    # the titles' (selected) mark waits for the coord's pick to be final (every planned trial complete) so it never
    # goes stale: an incomplete coord renders its evals with sel unset, a complete one with the recorded index; the
    # trial's own best is marked either way
    dpath_trial, _ = _stub_render(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(rmv, "render_eval", lambda dpath_evals, name, cfg, ctx, chkpt, **k: calls.append((name, chkpt)))
    fpath = dpath_trial / "eval" / "all" / "1" / "viz" / "cache" / "projections.npz"
    fpath.parent.mkdir(parents=True)
    fpath.write_text("")
    (dpath_trial / "trial_metadata.json").write_text(json.dumps({"chkpt": {"sel": 1, "best": 1}}))
    cfg = {"pooled": {"enabled": False}, "store_cache": True}

    rmv.render_trial(dpath_trial, cfg_manifold_viz=cfg)
    monkeypatch.setattr(rmv, "coord_complete", lambda dataset, arm, coord: (dataset, arm, coord) == ("cub", "hp", "c0"))
    rmv.render_trial(dpath_trial, cfg_manifold_viz=cfg)

    assert calls == [("1", {"sel": None, "best": 1}), ("1", {"sel": 1, "best": 1})]


def test_render_trial_leaves_unselected_trials_alone(tmp_path, monkeypatch) -> None:
    # a trial not yet selected (trial_metadata.json's chkpt still unset -- e.g. rendered by hand mid-trial) gets no
    # viz copies
    dpath_trial, _ = _stub_render(tmp_path, monkeypatch)
    fpath = dpath_trial / "eval" / "all" / "1" / "viz" / "vanilla" / "x.png"
    fpath.parent.mkdir(parents=True)
    fpath.write_text("x")
    (dpath_trial / "trial_metadata.json").write_text(json.dumps({"chkpt": {"sel": None, "best": None}}))

    rmv.render_trial(dpath_trial, cfg_manifold_viz={"pooled": {"enabled": False}, "store_cache": True})

    assert not (dpath_trial / "eval" / "sel").exists() and not (dpath_trial / "eval" / "best").exists()


def test_render_trial_full_render_drops_the_caches_when_not_stored(tmp_path, monkeypatch) -> None:
    # store_cache false: the full render is the caches' last consumer, so its final step deletes every eval's
    # viz/cache/ -- the stills and the sel/best copies survive. A partial render (no_evo here) leaves the
    # caches for the half it skipped
    dpath_trial, _ = _stub_render(tmp_path, monkeypatch)
    cfg = {"pooled": {"enabled": False}, "store_cache": False}

    dpath_evals = dpath_trial / "eval" / "all"
    for idx in ("1", "2"):
        for sub in ("vanilla/8panel/joint.png", "cache/projections.npz"):
            fpath = dpath_evals / idx / "viz" / sub
            fpath.parent.mkdir(parents=True)
            fpath.write_text(f"eval{idx}")
    (dpath_trial / "trial_metadata.json").write_text(json.dumps({"chkpt": {"sel": 2, "best": 1}}))

    rmv.render_trial(dpath_trial, skip_evo=True, cfg_manifold_viz=cfg)
    assert (dpath_evals / "1" / "viz" / "cache" / "projections.npz").exists()

    rmv.render_trial(dpath_trial, cfg_manifold_viz=cfg)
    assert not (dpath_evals / "1" / "viz" / "cache").exists()
    assert not (dpath_evals / "2" / "viz" / "cache").exists()
    assert (dpath_evals / "2" / "viz" / "vanilla" / "8panel" / "joint.png").exists()
    assert (dpath_trial / "eval" / "sel" / "viz" / "vanilla" / "8panel" / "joint.png").read_text() == "eval2"


def test_render_test_trial_renders_the_lone_eval_and_rebuilds_the_mirror(tmp_path, monkeypatch) -> None:
    # a test trial's one eval (eval/sel/) gets its UMAPs fit and its stills drawn in place -- the context off the
    # path + the phase's config snapshot (split; the coord's config.json copy prunes it) + the coord's config.json
    # copy (chkpt_stop names the eval in the titles), the partition tier 'test' -- then the arm's test best_coord/
    # mirror, which copies the stills, is rebuilt
    calls = []
    monkeypatch.setattr(rmv, "compute_umap_eval", lambda *a: calls.append(("umap", *a)))
    monkeypatch.setattr(rmv, "render_test_eval", lambda *a: calls.append(("render", *a)))
    monkeypatch.setattr(rmv, "update_test_best_coord",
                        lambda dataset, arm: calls.append(("mirror", rmv.ArtifactManager.dpath_phase, dataset, arm)))
    dpath_trial = tmp_path / "camp" / "_phase" / "test" / "_dataset" / "cub" / "_arm" / "hp" / "_coord" / "c0" / "_trial" / "1"
    dpath_trial.mkdir(parents=True)
    (dpath_trial.parents[7] / "cfg_baseline.json").write_text(json.dumps({"train": {"split": {"split": "s1"}}}))
    (dpath_trial.parents[1] / "config.json").write_text(json.dumps({"split": {"train_pt": "trainval"}, "chkpt_stop": 3}))
    cfg = {"umap": {"n_neighbors": 15}, "pooled": {"enabled": True}, "store_cache": True}

    rmv.render_test_trial(dpath_trial, cfg)

    dpath_eval = dpath_trial / "eval" / "sel"
    viz_context = rmv.VizContext(arm="hp", coord="c0", dataset="cub", split="s1", eval_pt="test")
    assert calls == [("umap", dpath_eval, cfg["umap"]),
                     ("render", dpath_eval, cfg, viz_context, "Chkpt 3"),
                     ("mirror", tmp_path / "camp" / "_phase" / "test", "cub", "hp")]


def test_render_test_trial_drops_the_cache_when_not_stored(tmp_path, monkeypatch) -> None:
    # delete-after-use for a test trial's lone eval: stills drawn and mirror rebuilt, then eval/sel/viz/cache/ goes
    for fn in ("compute_umap_eval", "render_test_eval"):
        monkeypatch.setattr(rmv, fn, lambda *a: None)
    monkeypatch.setattr(rmv, "update_test_best_coord", lambda dataset, arm: None)
    dpath_trial = tmp_path / "camp" / "_phase" / "test" / "_dataset" / "cub" / "_arm" / "hp" / "_coord" / "c0" / "_trial" / "1"
    fpath_cache = dpath_trial / "eval" / "sel" / "viz" / "cache" / "projections.npz"
    fpath_cache.parent.mkdir(parents=True)
    fpath_cache.write_text("x")
    (dpath_trial.parents[7] / "cfg_baseline.json").write_text(json.dumps({"train": {"split": {"split": "s1"}}}))
    (dpath_trial.parents[1] / "config.json").write_text(json.dumps({"chkpt_stop": 3}))

    rmv.render_test_trial(dpath_trial, {"umap": {}, "store_cache": False})

    assert not fpath_cache.parent.exists()


def test_render_campaign_sweeps_the_test_tree_too(tmp_path, monkeypatch) -> None:
    # the campaign sweep covers test/ after screen/ and refine/: its trials render as test trials, those not scored
    # yet (no eval/sel/viz/cache/) skipped -- and all of them under evo_only, having no evolving GIFs
    monkeypatch.setattr(rmv, "paths", {"artifacts": tmp_path})
    calls = []
    monkeypatch.setattr(rmv, "render_trial", lambda dpath_trial, *a: calls.append(("train", dpath_trial)))
    monkeypatch.setattr(rmv, "render_test_trial", lambda dpath_trial, cfg: calls.append(("test", dpath_trial)))
    metadata = json.dumps({"matrix": {"cub": {"hp": ["c0"]}}, "seeds": [42, 43]})
    dpath_screen = tmp_path / "camp" / "_phase" / "screen"
    dpath_test = tmp_path / "camp" / "_phase" / "test"
    for dpath_phase in (dpath_screen, dpath_test):
        dpath_phase.mkdir(parents=True)
        (dpath_phase / "phase_metadata.json").write_text(metadata)
    trials = {phase: dpath_phase / "_dataset" / "cub" / "_arm" / "hp" / "_coord" / "c0" / "_trial"
              for phase, dpath_phase in (("screen", dpath_screen), ("test", dpath_test))}
    for fpath in (trials["screen"] / "1" / "trial_metadata.json",  # screen trial 2 never ran; test trial 1 not scored yet
                  trials["test"] / "2" / "eval" / "sel" / "scores.json",
                  trials["test"] / "2" / "eval" / "sel" / "viz" / "cache" / "projections.npz"):
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text("x")

    rmv.render_campaign("camp")
    assert calls == [("train", trials["screen"] / "1"), ("test", trials["test"] / "2")]

    calls.clear()
    rmv.render_campaign("camp", evo_only=True)
    assert calls == [("train", trials["screen"] / "1")]
