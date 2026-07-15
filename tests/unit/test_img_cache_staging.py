import json
import pickle

import pytest

import utils.data as ud


def _make_source_pack(dpath_src, payload=b"fake-image-bytes"):
    dpath_src.mkdir(parents=True)
    (dpath_src / "pack.bin").write_bytes(payload)
    with open(dpath_src / "index.pkl", "wb") as f:
        pickle.dump({"cls/a.png": (0, len(payload))}, f)
    (dpath_src / "meta.json").write_text(json.dumps({"total_bytes": len(payload), "created_utc": "t0"}))


@pytest.fixture
def staged_env(tmp_path, monkeypatch):
    # source pack under a fake paths["img_cache"], staging root under a fake SLURM_TMPDIR
    monkeypatch.setitem(ud.paths, "img_cache", tmp_path / "img_cache")
    monkeypatch.setenv("SLURM_TMPDIR", str(tmp_path / "node_local"))
    _make_source_pack(tmp_path / "img_cache" / "toy")
    return tmp_path


def test_stage_copies_pack_and_is_idempotent(staged_env) -> None:
    ud.stage_img_cache("toy")

    dpath_staged = ud.dpath_img_cache_staged("toy")
    assert (dpath_staged / "pack.bin").read_bytes() == b"fake-image-bytes"
    assert (dpath_staged / "meta.json").exists()

    # second call takes the already-staged fast path: staged files are not rewritten
    mtime = (dpath_staged / "pack.bin").stat().st_mtime_ns
    ud.stage_img_cache("toy")
    assert (dpath_staged / "pack.bin").stat().st_mtime_ns == mtime


def test_stage_restages_when_source_pack_rebuilt(staged_env) -> None:
    ud.stage_img_cache("toy")

    # a rebuilt source pack changes meta.json -> the staged copy must be refreshed
    dpath_src = ud.paths["img_cache"] / "toy"
    (dpath_src / "pack.bin").write_bytes(b"rebuilt!")
    (dpath_src / "meta.json").write_text(json.dumps({"total_bytes": 8, "created_utc": "t1"}))
    ud.stage_img_cache("toy")

    assert (ud.dpath_img_cache_staged("toy") / "pack.bin").read_bytes() == b"rebuilt!"


def test_stage_restages_when_staged_pack_truncated(staged_env) -> None:
    ud.stage_img_cache("toy")

    # a partially-copied pack (size mismatch vs meta) must read as invalid and be re-copied
    (ud.dpath_img_cache_staged("toy") / "pack.bin").write_bytes(b"trunc")
    ud.stage_img_cache("toy")

    assert (ud.dpath_img_cache_staged("toy") / "pack.bin").read_bytes() == b"fake-image-bytes"


def test_stage_restages_when_staged_pack_evicted(staged_env) -> None:
    ud.stage_img_cache("toy")

    # a /tmp reaper can evict pack.bin while the frequently-read meta.json marker survives -- must
    # classify as invalid and re-stage, not crash
    (ud.dpath_img_cache_staged("toy") / "pack.bin").unlink()
    ud.stage_img_cache("toy")

    assert (ud.dpath_img_cache_staged("toy") / "pack.bin").read_bytes() == b"fake-image-bytes"


def test_stage_missing_source_pack_raises(staged_env) -> None:
    with pytest.raises(FileNotFoundError, match="build_img_cache"):
        ud.stage_img_cache("never_built")
