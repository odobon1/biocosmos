import time
from contextlib import contextmanager


class Timer:
    def __init__(self):
        self._start_time = None
        self._elapsed_time = 0.0
        self._active = False

    def start(self):
        if self._active:
            raise RuntimeError("Timer is already active.")
        self._start_time = time.time()
        self._active = True

    def stop(self):
        if not self._active:
            raise RuntimeError("Timer is not active.")
        self._elapsed_time += time.time() - self._start_time
        self._start_time = None
        self._active = False

    def get_elapsed_time(self):
        if self._active:
            elapsed_time = self._elapsed_time + (time.time() - self._start_time)
        else:
            elapsed_time = self._elapsed_time
        return elapsed_time

    def set_elapsed_time(self, elapsed_time):
        if self._active:
            raise RuntimeError("Timer is active. Stop it before setting elapsed time.")
        self._elapsed_time = elapsed_time

    def reset(self):
        self._start_time = None
        self._elapsed_time = 0.0
        self._active = False


class RunningMean:
    """
    Track running mean via Welford's algorithm
    """
    def __init__(self):
        self.n = 0
        self.mean = 0.0

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n

    def value(self):
        return self.mean


class TimeTracker:
    """
    Per-trial timing registry of running MEANS over repeated, homogeneous events (`train` per epoch;
    `eval`/`viz_compute` per eval). A bucket's total time is mean*n; the unattributed remainder of the
    trial wall-clock (checkpoint I/O, sync, etc.) is reported separately as "other" (trial - attributed()).
    """

    MEANS = ("train", "eval", "viz_compute")

    def __init__(self):
        self._means = {name: RunningMean() for name in self.MEANS}

    def add(self, name, value):
        self._means[name].update(value)

    @contextmanager
    def measure(self, name):
        """Time the with-block into bucket `name` (running mean via add()). Records only on a clean exit --
        if the block raises, the elapsed time is NOT folded in (so a failed eval doesn't pollute the mean)
        and the exception propagates."""
        timer = Timer()
        timer.start()
        yield
        timer.stop()
        self.add(name, timer.get_elapsed_time())

    def mean(self, name):
        rm = self._means[name]
        return rm.value() if rm.n > 0 else None

    def n(self, name):
        return self._means[name].n

    def total(self, name):
        rm = self._means[name]
        return rm.mean * rm.n

    def attributed(self):
        return sum(self.total(name) for name in self.MEANS)

    def state_dict(self):
        return {"means": {name: (rm.n, rm.mean) for name, rm in self._means.items()}}

    def load_state_dict(self, state):
        for name, (n, mean) in state["means"].items():
            self._means[name].n = n
            self._means[name].mean = mean
