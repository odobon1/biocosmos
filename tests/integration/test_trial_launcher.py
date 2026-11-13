"""
The trial exit protocol (utils.stop) through the ACTUAL launcher: campaign_runner._run_trial_subprocess -> a real
`torchrun --standalone` -> a trial process -- with a stand-in trial module in place of the training (a test-owned
campaign_trial_runner.py, found first because the launch runs from the test's own dir), ending each of the four
ways the runner must tell apart. torchrun reports every failed worker as its own exit 1, so this is the check that
the endings really do come out the far side of it distinguishable -- and that a stop signalled to the runner
process reaches the trial.

The launch runs in a test-owned driver process, not in pytest's: the launcher reaps every descendant of the process
it runs in on the way out.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from utils import stop


pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]

# the stand-in trial: campaign_trial_runner's own shape -- main under exit_protocol -- with a "training" that ends
# as its mode says. In stop mode it runs until a stop is requested (announcing itself first, so the test knows it
# is up), then stops as the train loop does once it has checkpointed: by raising TrialStopped.
_TRIAL = '''
import errno, json, os, sys, time
from argparse import ArgumentParser
from pathlib import Path

from utils import stop


@stop.exit_protocol()
def main():
    parser = ArgumentParser()
    parser.add_argument("--cfg-json", required=True)
    mode = json.loads(parser.parse_args().cfg_json)["mode"]
    if mode == "crash":
        raise RuntimeError("boom")
    if mode == "storage":
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), "train_state.pt")
    if mode == "stop":
        Path(os.environ["TRIAL_READY_FILE"]).touch()
        deadline = time.time() + 120
        while not stop.stop_requested():
            if time.time() > deadline:
                raise RuntimeError("no stop request arrived")
            time.sleep(0.1)
        raise stop.TrialStopped


if __name__ == "__main__":
    main()
'''

# the driver: the runner's side. One launch per mode through the real _run_trial_subprocess, each outcome printed
# as a json line; the runner's stop handler is installed, as campaign_runner.main installs it.
_DRIVER = '''
import json, subprocess, sys

import campaign_runner as cr
from utils import stop

stop.install_stop_handler()
for mode in sys.argv[1:]:
    cfg_dict = {"campaign": "cmp", "phase": "screen", "arm": "sp", "coord": "base", "dataset": "cub", "seed": 42, "mode": mode}
    try:
        cr._run_trial_subprocess(cfg_dict)
        outcome = {"mode": mode, "returncode": 0, "halt": None}
    except subprocess.CalledProcessError as e:
        outcome = {"mode": mode, "returncode": e.returncode, "halt": cr._halt_code(e)}
    print("OUTCOME " + json.dumps(outcome), flush=True)
'''


def test_launcher_tells_completion_crash_full_disk_and_checkpointed_stop_apart(tmp_path) -> None:
    (tmp_path / "campaign_trial_runner.py").write_text(_TRIAL)
    (tmp_path / "driver.py").write_text(_DRIVER)
    (tmp_path / "config").symlink_to(ROOT / "config")  # the runner's modules read the config dir at import, off the cwd
    fpath_stop, fpath_ready = tmp_path / "stop", tmp_path / "ready"
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",  # torchrun sits beside the interpreter
        stop.ENV_STOP_FILE: str(fpath_stop),
        "TRIAL_READY_FILE": str(fpath_ready),
    }
    proc = subprocess.Popen([sys.executable, "driver.py", "complete", "crash", "storage", "stop"], cwd=tmp_path, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.time() + 600
        while not fpath_ready.exists():  # the stop-mode trial is up, the driver waiting on it
            assert proc.poll() is None, proc.communicate()[1][-3000:]
            assert time.time() < deadline
            time.sleep(0.2)
        assert not fpath_stop.exists()  # nothing has requested a stop yet
        proc.send_signal(signal.SIGUSR1)  # the request, as a supervisor makes it: a signal to the runner process...
        proc.send_signal(signal.SIGUSR1)  # ...and again: repeated requests are one stop
        out, err = proc.communicate(timeout=300)
    finally:
        proc.kill()

    assert proc.returncode == 0, err[-3000:]  # the signals stopped the trial, not the runner
    outcomes = [json.loads(line.removeprefix("OUTCOME ")) for line in out.splitlines() if line.startswith("OUTCOME ")]
    assert outcomes == [
        {"mode": "complete", "returncode": 0, "halt": None},
        {"mode": "crash", "returncode": 1, "halt": None},  # an ordinary crash: retried by the runner
        {"mode": "storage", "returncode": 1, "halt": stop.EXIT_STORAGE},
        {"mode": "stop", "returncode": 1, "halt": stop.EXIT_STOPPED},  # exactly one stop, though signalled twice
    ]
    assert fpath_stop.exists()  # the runner passed the signalled request on as the stop file the trial polls
    # torchrun flattened every failed worker to exit 1 above; the worker's own exit code is in its report all the same
    assert f"exitcode  : {stop.EXIT_STOPPED}" in err and f"exitcode  : {stop.EXIT_STORAGE}" in err
