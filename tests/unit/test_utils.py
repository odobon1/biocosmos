import numpy as np
import pytest

from utils.utils import RunningMean, PrintLog, get_text_template, save_npz, shuffle_list


def test_running_mean_tracks_average() -> None:
    mean = RunningMean()

    for value in [2.0, 4.0, 6.0, 8.0]:
        mean.update(value)

    assert mean.n == 4
    assert mean.value() == 5.0


def test_shuffle_list_is_seeded_and_non_mutating() -> None:
    values = [1, 2, 3, 4, 5]

    shuffled_a = shuffle_list(values, seed=17)
    shuffled_b = shuffle_list(values, seed=17)

    assert shuffled_a == shuffled_b
    assert values == [1, 2, 3, 4, 5]
    assert shuffled_a != values


def test_get_text_template_returns_known_templates() -> None:
    train_template = get_text_template("train")
    lepid_template = get_text_template("train", dataset="lepid")
    bioclip_template = get_text_template("sci", dataset="cub")

    assert isinstance(train_template, list)
    assert train_template[0] == ["", "a photo of "]
    assert lepid_template[-1] == ["", " butterfly"]
    assert bioclip_template == [["a photo of $SCI$"]]


def test_printlog_eval_handles_missing_loss_key() -> None:
    class _EvalPipe:
        partitions = ["id"]

    eval_metrics = {
        "scores": {
            "native": {
                "id": {"map": {"i2t": 0.1, "i2i": 0.2, "t2i": 0.3}, "acc": {"i2t": 0.4}},
                "comp": {"map": {"all": 0.2, "i2i": 0.2, "id": 0.2}},
            },
            "joint_macro": {
                "id": {"map": {"i2t": 0.13, "i2i": 0.23, "t2i": 0.33}, "acc": {"i2t": 0.43}},
                "comp": {"map": {"all": 0.23, "i2i": 0.23, "id": 0.23}},
            },
        },
        "loss_raw": {"id": None},
        "sim": {"min": None, "max": None, "median": None, "mean": None},
        "targ": {"min": None, "max": None, "median": None, "mean": None},
    }

    PrintLog.eval(eval_metrics, _EvalPipe(), {"native": "Standard", "joint_macro": "GZSL"})


def test_printlog_eval_blocks_follow_the_eval_groups_in_play(capsys) -> None:
    # one composite block per group in play, under its reported name; a group switched off has no
    # scores subtree at all, so a stale block would KeyError rather than print blank
    class _EvalPipe:
        partitions = ["id"]

    eval_metrics = {
        "scores": {
            "native": {"comp": {"map": {"all": 0.2, "i2i": 0.2, "id": 0.2}}},
            "joint_macro": {"comp": {"map": {"all": 0.23, "i2i": 0.23, "id": 0.23}}},
        },
        "loss_raw": {"id": None},
        "sim": {"min": None, "max": None, "median": None, "mean": None},
        "targ": {"min": None, "max": None, "median": None, "mean": None},
    }

    PrintLog.eval(eval_metrics, _EvalPipe(), {"native": "Standard", "joint_macro": "GZSL"})

    printout = capsys.readouterr().out
    assert "Standard Composite mAP" in printout
    assert "GZSL Composite mAP" in printout
    assert "Native" not in printout and "Joint" not in printout


def test_save_npz_round_trips_and_leaves_no_tmp(tmp_path) -> None:
    fpath = tmp_path / "projections.npz"

    save_npz(fpath, a=np.arange(3), b=np.ones((2, 2)))

    with np.load(fpath) as npz:
        assert npz["a"].tolist() == [0, 1, 2]
        assert npz["b"].shape == (2, 2)
    assert list(tmp_path.iterdir()) == [fpath]  # the tmp file was renamed away, not left behind


def test_save_npz_killed_mid_write_leaves_the_existing_file_intact(tmp_path, monkeypatch) -> None:
    # the atomicity save_npz exists for: a render worker terminated mid-rewrite (campaign runner replan,
    # watchdog cancellation) must leave the previously-good cache in place, never a torn npz
    fpath = tmp_path / "projections.npz"
    save_npz(fpath, a=np.arange(3))

    def torn_savez(f, **arrays):
        f.write(b"PK\x03\x04 torn")  # a few bytes land, then the kill
        raise KeyboardInterrupt

    monkeypatch.setattr("utils.utils.np.savez", torn_savez)
    with pytest.raises(KeyboardInterrupt):
        save_npz(fpath, a=np.arange(5))

    with np.load(fpath) as npz:
        assert npz["a"].tolist() == [0, 1, 2]  # the pre-rewrite cache, readable
