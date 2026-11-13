"""
The two ways a run ends short of its end that are neither a completion nor a crash to retry, and how each gets from
a trial process to whoever launched it:

  - a REQUESTED STOP (opt-in, off unless BC_STOP_FILE is set): the trial checkpoints at its next batch boundary and
    exits; nothing is marked complete or failed, and a relaunch resumes it from that checkpoint
  - a confirmed STORAGE-FULL failure (ENOSPC / EDQUOT, by errno): retrying cannot succeed until space is freed, so
    the run halts instead of burning its retries

A trial runs under torchrun, which reports ANY failed worker as its own exit code 1, so a worker's exit code alone
never reaches the campaign runner. Each ending is therefore also a marker line on stderr, which the runner already
captures for crash classification (campaign_runner._classify_crash); the runner turns the marker back into the same
exit code for its own caller. exit_protocol does the trial-side (and runner-side) half.

Requesting a stop. BC_STOP_FILE names a file; the stop is requested for as long as that file exists. A trial process
polls it at every batch boundary (stop_requested; train.py). The campaign runner also takes SIGUSR1 as a request
(install_stop_handler: the handler only sets a flag) and passes it on to the running trial by creating the file
(relay_stop). Requests are idempotent -- any number of signals, or touches of the file, stop the run once. The file
should be node-local (a trial and its runner always share a node): on a shared filesystem a newly created path can
stay invisible to a process already polling it. Single-GPU only (enforced: train.run_training, campaign_runner.main).

Stdlib-only, so campaign_trial_runner can import it before anything heavy.
"""

from contextlib import contextmanager
from pathlib import Path
import errno
import os
import signal
import sys
import traceback


ENV_STOP_FILE = "BC_STOP_FILE"

EXIT_STORAGE = 74  # sysexits' EX_IOERR
EXIT_STOPPED = 75  # sysexits' EX_TEMPFAIL: not done, nothing wrong -- run it again

MARKER_STORAGE = "BIOCOSMOS_STORAGE_FULL"
MARKER_STOPPED = "BIOCOSMOS_STOPPED"

_signaled = False  # (campaign runner) a stop signal has arrived


class TrialStopped(Exception):
    """Raised out of the train loop once a requested stop has been checkpointed (train.TrainPipeline.train)."""


def stop_enabled() -> bool:
    return ENV_STOP_FILE in os.environ

def stop_requested() -> bool:
    """Whether a stop has been requested: the stop file exists (or, in the campaign runner, its signal has arrived).
    Always False with the mechanism off (BC_STOP_FILE unset)."""
    return stop_enabled() and (_signaled or os.path.exists(os.environ[ENV_STOP_FILE]))

def _on_stop_signal(signum, frame) -> None:
    global _signaled
    _signaled = True

def install_stop_handler() -> None:
    """(campaign runner) Take SIGUSR1 as a stop request. The handler only records it; the runner acts on it where
    it is safe to -- between trials (stop_requested), and while a trial runs by passing it on (relay_stop)."""
    signal.signal(signal.SIGUSR1, _on_stop_signal)

def relay_stop() -> None:
    """(campaign runner, polled while it waits on a trial) Pass a signalled stop request on to the trial process by
    creating the stop file it polls. A no-op until a signal has arrived, and harmless to repeat."""
    if _signaled:
        Path(os.environ[ENV_STOP_FILE]).touch()

def storage_full(exc: BaseException | None) -> bool:
    """Whether `exc` is a confirmed out-of-space failure: an OSError carrying ENOSPC (device full) or EDQUOT (over
    quota), itself or anywhere in the chain it was raised from (torch.save reports a failed write as a RuntimeError
    raised from the OSError). Decided from the errno alone -- free-space readings say nothing about quotas."""
    while exc is not None:
        if isinstance(exc, OSError) and exc.errno in (errno.ENOSPC, errno.EDQUOT):
            return True
        exc = exc.__cause__ or exc.__context__
    return False

@contextmanager
def exit_protocol():
    """Wrap a process's main so the two endings above leave it as their marker line on stderr + their exit code:
    TrialStopped as MARKER_STOPPED / EXIT_STOPPED, a storage-full failure (storage_full) as its traceback +
    MARKER_STORAGE / EXIT_STORAGE. Anything else propagates untouched -- a crash stays a crash. The marker goes to
    stderr alone, never to disk: it has to get out when the disk is what failed."""
    try:
        yield
    except TrialStopped:
        print(MARKER_STOPPED, file=sys.stderr, flush=True)
        sys.exit(EXIT_STOPPED)
    except Exception as e:
        if not storage_full(e):
            raise
        traceback.print_exc()
        print(MARKER_STORAGE, file=sys.stderr, flush=True)
        sys.exit(EXIT_STORAGE)
