import errno
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from utils import stop


ROOT = Path(__file__).resolve().parents[2]


def _enospc():
    return OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))


def test_stop_is_off_unless_opted_in(tmp_path, monkeypatch) -> None:
    # default behavior: without BC_STOP_FILE there is no stop mechanism -- nothing requests one, not even a signal flag
    monkeypatch.delenv(stop.ENV_STOP_FILE, raising=False)
    monkeypatch.setattr(stop, "_signaled", True)
    assert not stop.stop_enabled()
    assert not stop.stop_requested()


def test_stop_is_requested_for_as_long_as_the_stop_file_exists(tmp_path, monkeypatch) -> None:
    fpath = tmp_path / "stop"
    monkeypatch.setenv(stop.ENV_STOP_FILE, str(fpath))
    assert stop.stop_enabled()
    assert not stop.stop_requested()

    fpath.touch()
    assert stop.stop_requested()
    fpath.touch()  # a repeated request changes nothing
    assert stop.stop_requested()


def test_relay_passes_a_signalled_request_on_as_the_stop_file(tmp_path, monkeypatch) -> None:
    fpath = tmp_path / "stop"
    monkeypatch.setenv(stop.ENV_STOP_FILE, str(fpath))

    stop.relay_stop()  # nothing signalled: nothing passed on
    assert not fpath.exists()

    monkeypatch.setattr(stop, "_signaled", True)
    assert stop.stop_requested()  # the runner itself sees the request before any relay
    stop.relay_stop()
    stop.relay_stop()  # polled every second while a trial runs: repeats are harmless
    assert fpath.exists()


def test_signal_handler_only_records_the_request(tmp_path) -> None:
    # a real SIGUSR1, twice, to a test-owned process running the runner's handler: the handler does nothing but set
    # the flag -- the process carries on, sees one pending request when it next looks, and relays it then
    fpath_stop, fpath_ready = tmp_path / "stop", tmp_path / "ready"
    script = (
        "import sys, time\n"
        "from pathlib import Path\n"
        "from utils import stop\n"
        "stop.install_stop_handler()\n"
        "Path(sys.argv[1]).touch()\n"
        "deadline = time.time() + 60\n"
        "while not stop.stop_requested() and time.time() < deadline:\n"
        "    time.sleep(0.05)\n"
        "time.sleep(0.5)\n"  # room for the second signal to land
        "print('requested' if stop.stop_requested() else 'timeout', Path(stop.os.environ[stop.ENV_STOP_FILE]).exists())\n"
        "stop.relay_stop()\n"
    )
    env = {**os.environ, stop.ENV_STOP_FILE: str(fpath_stop), "PYTHONPATH": str(ROOT)}
    proc = subprocess.Popen([sys.executable, "-c", script, str(fpath_ready)], env=env, stdout=subprocess.PIPE, text=True)
    try:
        deadline = time.time() + 60
        while not fpath_ready.exists():  # the handler is installed: only now is the signal safe to send
            assert proc.poll() is None and time.time() < deadline
            time.sleep(0.05)
        proc.send_signal(signal.SIGUSR1)
        proc.send_signal(signal.SIGUSR1)
        out, _ = proc.communicate(timeout=60)
    finally:
        proc.kill()

    assert proc.returncode == 0  # the signals neither killed it nor raised in it
    assert out.split() == ["requested", "False"]  # recorded by the handler, nothing written by it
    assert fpath_stop.exists()  # the relay, run outside the handler, created the file


def test_storage_full_is_decided_by_errno() -> None:
    assert stop.storage_full(_enospc())
    assert stop.storage_full(OSError(errno.EDQUOT, os.strerror(errno.EDQUOT)))
    assert not stop.storage_full(OSError(errno.EIO, os.strerror(errno.EIO)))
    assert not stop.storage_full(OSError("No space left on device"))  # the words without the errno confirm nothing
    assert not stop.storage_full(RuntimeError("boom"))
    assert not stop.storage_full(None)

    # torch.save reports a failed write as a RuntimeError raised from the OSError: the chain is searched
    try:
        try:
            raise _enospc()
        except OSError as e:
            raise RuntimeError("PytorchStreamWriter failed writing file") from e
    except RuntimeError as wrapped:
        assert stop.storage_full(wrapped)


def test_exit_protocol_maps_a_checkpointed_stop_to_its_marker_and_exit_code(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        with stop.exit_protocol():
            raise stop.TrialStopped

    assert exc.value.code == stop.EXIT_STOPPED
    assert stop.MARKER_STOPPED in capsys.readouterr().err


def test_exit_protocol_maps_a_full_disk_to_its_marker_and_exit_code(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        with stop.exit_protocol():
            raise _enospc()

    assert exc.value.code == stop.EXIT_STORAGE
    err = capsys.readouterr().err
    assert stop.MARKER_STORAGE in err and "No space left on device" in err  # the traceback is kept beside the marker


def test_exit_protocol_leaves_everything_else_alone(capsys) -> None:
    # a crash stays a crash (the runner retries it), a clean return a completion, an explicit exit that exit
    with pytest.raises(RuntimeError, match="boom"):
        with stop.exit_protocol():
            raise RuntimeError("boom")
    with pytest.raises(OSError):
        with stop.exit_protocol():
            raise OSError(errno.EIO, os.strerror(errno.EIO))
    with pytest.raises(SystemExit) as exc:
        with stop.exit_protocol():
            sys.exit(3)
    assert exc.value.code == 3
    with stop.exit_protocol():
        pass
    err = capsys.readouterr().err
    assert stop.MARKER_STOPPED not in err and stop.MARKER_STORAGE not in err
