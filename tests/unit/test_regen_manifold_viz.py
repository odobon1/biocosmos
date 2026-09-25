import json

from tools import regen_manifold_viz as rmv


def test_render_trial_copies_viz_to_selected_and_best(tmp_path, monkeypatch) -> None:
    # the selection is written at trial end, before the detached render draws any stills, so render_trial
    # re-copies the eval's viz/ afterwards: _selected/viz/ from the eval its metrics record (chkpt 2), _best/viz/
    # from its own (chkpt 1) -- the rendering itself is stubbed, the stills and cache already on disk
    for fn in ("compute_umap_projections", "render_eval", "render_evolution"):
        monkeypatch.setattr(rmv, fn, lambda *a, **k: None)
    monkeypatch.setattr(rmv, "_viz_context", lambda dpath_trial: None)
    cfg = {"pooled": {"enabled": False}}

    dpath_evals = tmp_path / "evals"
    for idx in ("1", "2"):
        for sub in ("vanilla/8panel/joint.png", "cache/projections.npz"):
            fpath = dpath_evals / "evals" / idx / "viz" / sub
            fpath.parent.mkdir(parents=True)
            fpath.write_text(f"eval{idx}")
    for name, idx in (("_selected", 2), ("_best", 1)):
        fpath = dpath_evals / name / "metrics" / "metrics.json"
        fpath.parent.mkdir(parents=True)
        fpath.write_text(json.dumps({"chkpt": f"{idx}/3 (0.0M/0.0M samples)"}))

    rmv.render_trial(tmp_path, cfg_manifold_viz=cfg)

    for sub in ("vanilla/8panel/joint.png", "cache/projections.npz"):
        assert (dpath_evals / "_selected" / "viz" / sub).read_text() == "eval2"
        assert (dpath_evals / "_best" / "viz" / sub).read_text() == "eval1"


def test_render_trial_leaves_unselected_trials_alone(tmp_path, monkeypatch) -> None:
    # a trial not yet selected (no _selected/ metrics -- e.g. rendered by hand mid-trial) gets no viz copies
    for fn in ("compute_umap_projections", "render_eval", "render_evolution"):
        monkeypatch.setattr(rmv, fn, lambda *a, **k: None)
    monkeypatch.setattr(rmv, "_viz_context", lambda dpath_trial: None)
    fpath = tmp_path / "evals" / "evals" / "1" / "viz" / "vanilla" / "x.png"
    fpath.parent.mkdir(parents=True)
    fpath.write_text("x")

    rmv.render_trial(tmp_path, cfg_manifold_viz={"pooled": {"enabled": False}})

    assert not (tmp_path / "evals" / "_selected").exists() and not (tmp_path / "evals" / "_best").exists()
