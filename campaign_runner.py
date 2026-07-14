"""
python -m campaign_runner --dev
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m campaign_runner --dev

Campaigns are defined in config/camps/<campaign>.yaml, e.g. --dev_basic loads config/camps/dev_basic.yaml.
"""

from pathlib import Path
from copy import deepcopy
import ctypes
import itertools
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import traceback
import time
import psutil
import torch
import yaml

from utils.config import CFG_PARAM_ALIASES, CFG_PARAM_VALUE_ALIASES, apply_overrides, apply_train_debug_overrides, load_train_config_dict, load_manifold_viz_config_dict, load_model_specific_config_dict, load_hardware_config_dict
from utils.hardware import get_slurm_alloc
from utils.utils import paths, save_pickle, save_json, load_json, PrintLog

# Trial subprocesses (torchrun) inherit this env. expandable_segments lets the CUDA caching allocator
# hand the training step's reserved-but-unallocated pool to the large O(N^2) t-SNE buffers at eval time,
# preventing combined-set OOM (e.g. lepid, N~94k). setdefault so an explicit shell override still wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


SEED0 = 42  # first trial seed; trial seeds are SEED0 .. SEED0 + n_trials - 1


def _relevant_stderr(stderr: str) -> str:
    """Drop pre-crash noise (eval progress bars, warnings) by keeping the
    captured stderr from the first Python traceback onward. tqdm and warnings
    are emitted while the trial runs, so the first 'Traceback (most recent call
    last):' marks where the relevant error content begins. Falls back to the
    full text when no traceback is present (e.g. a bare SIGKILL)."""
    marker = "Traceback (most recent call last):"
    idx = stderr.find(marker)
    return stderr if idx == -1 else stderr[idx:]

def _log_trial_error(dpath_trial: Path, idx_trial: int, n_trials: int, seed: int, dataset: str, setting: str, exc: Exception) -> None:
    """Log trial error to stdout and to error.log in the trial-seed's directory."""
    dpath_trial.mkdir(parents=True, exist_ok=True)
    fpath_error = dpath_trial / "error.log"
    # Format error message with context
    error_msg = (
        f"\n[{idx_trial}/{n_trials}] TRIAL FAILED\n"
        f"  seed={seed}, dataset={dataset}, setting={setting}"
    )
    stderr_body = None
    if isinstance(exc, subprocess.CalledProcessError):
        stderr = getattr(exc, "stderr", None)
        if stderr:
            stderr_body = _relevant_stderr(stderr)
    # Print to stdout
    print(error_msg, flush=True)
    # Write to error log file
    with open(fpath_error, "w") as f:
        f.write(error_msg + "\n")
        if stderr_body is not None:
            f.write("--- stderr ---\n")
            f.write(stderr_body + "\n")
        else:
            f.write(traceback.format_exc())

def _dpath_campaign(campaign: str) -> Path:
    return paths["artifacts"] / campaign

def _load_or_create_campaign_config(campaign: str) -> dict:
    """Load the campaign's frozen config snapshot, creating it on first launch.

    On first launch four config sources are bundled into a single `artifacts/<campaign>/cfg_baseline.json`
    under the keys `train`, `hardware`, `manifold_viz`, `model_specific`. The `train` snapshot is derived
    from `config/train.yaml` (with `debug_mode` overrides folded in); the other three are `config/hardware.yaml`,
    `config/manifold_viz.yaml`, and `config/model_specific.yaml` verbatim. Every trial starts from the
    `train` snapshot and has the sibling snapshots injected per trial (as `hw`, `manifold_viz`,
    `model_specific`). Model-family `opt` defaults are left unresolved in `train` (kept `null`) and filled
    per trial from the `model_specific` snapshot, so a per-setting `arch.model_type` override still picks
    up the matching family's defaults. Every later relaunch (resume or matrix extension) reloads that
    snapshot rather than re-reading the YAML, so edits to any config file after a campaign's first launch
    never alter that campaign -- all of its trials, original or added later, train against the same
    frozen config."""
    fpath = _dpath_campaign(campaign) / "cfg_baseline.json"
    if fpath.exists():
        return load_json(fpath)

    cfg_train = apply_train_debug_overrides(load_train_config_dict())
    cfg_snapshot = {
        "train": cfg_train,
        "hardware": load_hardware_config_dict(),
        "manifold_viz": load_manifold_viz_config_dict(),
        "model_specific": load_model_specific_config_dict(),
    }
    fpath.parent.mkdir(parents=True, exist_ok=True)
    save_json(cfg_snapshot, fpath)
    return cfg_snapshot

def _fmt_name_value(v) -> str:
    """Format an override value for use in a setting name. Floats that Python renders in scientific
    notation are normalized to read like the YAML that declared them: '7e-06' -> '7.0e-6' (mantissa
    keeps a decimal point, exponent drops zero-padding)."""
    s = str(v)
    if isinstance(v, float) and "e" in s:
        mant, exp = s.split("e")
        if "." not in mant:
            mant += ".0"
        s = f"{mant}e{int(exp)}"
    return s

def _alias_pair(k: str, v) -> str:
    """Render one override as a 'key-value' name component, with the key mapped through
    CFG_PARAM_ALIASES and the value through CFG_PARAM_VALUE_ALIASES (per original key) when an
    alias exists, e.g. ('batch_size', 2048) -> 'bs-2k'."""
    v_aliased = CFG_PARAM_VALUE_ALIASES.get(k, {}).get(v, v)
    return f"{CFG_PARAM_ALIASES.get(k, k)}-{_fmt_name_value(v_aliased)}"

def _derive_item_name(item: dict) -> str:
    """Name an unnamed `baseline_overrides` item by its overrides: _alias_pair components joined
    by '_', e.g. {'loss.targ': 'sw', 'batch_size': 2048} -> 'L1T-sw_bs-2k'."""
    return "_".join(_alias_pair(k, v) for k, v in item.items())

def _item_name(item: dict) -> str | None:
    """The setting-name component an item contributes: its explicit 'name', or a name derived from
    its overrides via _derive_item_name when 'name' is absent. An explicit 'name: null' returns
    None -- the item contributes no component and is skipped when member names are joined."""
    return item["name"] if "name" in item else _derive_item_name(item)

def _expand_combo_lists(item: dict) -> list[dict]:
    """Expand an item's combo lists (list-valued overrides) into scalar items, one per combination
    of list values; several combo lists in one item cross with each other, the last-listed key
    varying fastest. The chosen 'key-value' pairs always show in the setting name: appended to an
    explicit 'name' (e.g. {'batch_size': [1024, 2048], 'name': 'hp'} -> 'hp_bs-1k', 'hp_bs-2k'),
    or picked up by _derive_item_name like any other override when the item is unnamed."""
    list_keys = [k for k, v in item.items() if k != "name" and isinstance(v, list)]
    if not list_keys:
        return [item]
    expanded = []
    for values in itertools.product(*(item[k] for k in list_keys)):
        chosen = dict(zip(list_keys, values))
        scalar_item = {k: chosen.get(k, v) for k, v in item.items()}
        if item.get("name") is not None:
            scalar_item["name"] = "_".join([item["name"], *(_alias_pair(k, chosen[k]) for k in list_keys)])
        expanded.append(scalar_item)
    return expanded

def _expand_settings(combo_groups: list[list[dict]]) -> list[tuple[str, dict]]:
    """Expand the campaign's combo groups into the full list of (name, overrides) settings.

    `baseline_overrides` is a list of combo groups; each combo group is a list of partial settings
    (a dict of dotted-key overrides plus an optional 'name'; an item without one is named from its
    overrides via _derive_item_name, e.g. {'loss.targ': 'sw'} -> 'L1T-sw'). An override
    value given as a list is a combo list: the item is first expanded into one partial setting per
    combination of its list values, named per _expand_combo_lists. The campaign's settings are the
    Cartesian product across combo groups: one partial setting is drawn from each combo group and
    merged into one setting, its name the members' names joined by '_' in combo-group order (e.g.
    'hp' x '2k' -> 'hp_2k'). A single combo group expands to its members unchanged. Combo groups
    are independent dimensions, so no override key may appear in more than one combo group -- a
    shared key would have two values fighting to define it when members merge.

    An item may set 'name' explicitly to null: it then contributes no name component and is skipped
    in the join (e.g. 'hp' x null -> 'hp'). At most one item per combo group may be null (two would
    give two settings the same name), and at least one combo group must have all its items named
    (else the all-null combination would yield an empty setting name)."""
    combo_groups = [
        [scalar_item for item in group for scalar_item in _expand_combo_lists(item)]
        for group in combo_groups
    ]
    group_keys = [
        {k for item in group for k in item if k != "name"}
        for group in combo_groups
    ]
    for (i, keys_i), (j, keys_j) in itertools.combinations(enumerate(group_keys), 2):
        shared = keys_i & keys_j
        if shared:
            raise ValueError(
                f"baseline_overrides key(s) {sorted(shared)} collide between combo groups {i} and {j}; "
                f"each override key must belong to exactly one combo group."
            )

    null_counts = [
        sum("name" in item and item["name"] is None for item in group)
        for group in combo_groups
    ]
    for i, n_null in enumerate(null_counts):
        if n_null > 1:
            raise ValueError(
                f"combo group {i} has {n_null} items with `name: null`; at most one item per combo "
                f"group may set `name: null`."
            )
    if null_counts and all(n_null > 0 for n_null in null_counts):
        raise ValueError(
            "every combo group has an item with `name: null`; at least one combo group must have "
            "all its items named, else the all-null combination yields an empty setting name."
        )

    settings = []
    seen_names: set[str] = set()
    for combo in itertools.product(*combo_groups):
        name = "_".join(
            part for item in combo if (part := _item_name(item)) is not None
        )
        if name in seen_names:
            raise ValueError(f"Duplicate baseline_overrides name: {name}")
        seen_names.add(name)
        payload = {k: deepcopy(v) for item in combo for k, v in item.items() if k != "name"}
        settings.append((name, payload))
    return settings

def _write_setting_overrides(campaign: str, setting: str, normalized_overrides: dict) -> None:
    fpath = _dpath_campaign(campaign) / "settings" / setting / "overrides.json"
    fpath.parent.mkdir(parents=True, exist_ok=True)
    with open(fpath, "w") as f:
        json.dump(normalized_overrides, f, indent=2, sort_keys=True)

def _iter_seeds(n_trials: int) -> list[int]:
    return list(range(SEED0, SEED0 + n_trials))

def _check_no_removals(campaign: str, prev_meta: dict, setting_names: list[str], datasets: list[str], seeds: list[int]) -> None:
    """A campaign's matrix may grow across runs (add settings/datasets/seeds) but never shrink.
    Compare the planned matrix against the one persisted from a prior run and raise if any
    previously-run setting, dataset, or seed is missing -- dropping one would orphan its
    already-computed trials and silently remove them from the campaign."""
    removed = []
    for kind, prev_vals, curr_vals in (
        ("settings", prev_meta["settings"], setting_names),
        ("datasets", prev_meta["datasets"], datasets),
        ("seeds", prev_meta["seeds"], seeds),
    ):
        missing = [v for v in prev_vals if v not in set(curr_vals)]
        if missing:
            removed.append(f"  {kind} removed: {missing}")
    if removed:
        raise RuntimeError(
            f"Campaign '{campaign}' config drops items recorded by a prior run "
            f"(settings/datasets/seeds may be added across runs but never removed):\n"
            + "\n".join(removed)
            + "\nRestore the removed items, or start a new campaign."
        )

def _enable_child_subreaper() -> None:
    """Become the reaper for orphaned descendants. torch elastic starts each
    rank in its own session, so when a rank is SIGKILLed (e.g. OOM) its
    DataLoader workers orphan to init and escape any process-group kill from
    here. As a subreaper we inherit them instead, so _reap_subtree can find
    and kill them."""
    PR_SET_CHILD_SUBREAPER = 36
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)
    except OSError:
        pass

def _reap_subtree(grace: float = 10.0, spare_root: int | None = None) -> None:
    """SIGKILL and reap every descendant process (torchrun, ranks, DataLoader
    workers). Relies on _enable_child_subreaper so orphaned workers reparent
    here and show up as descendants. A background render worker (spare_root and
    its subtree) is left alone so it can keep rendering through the next trial.

    SIGINT and SIGTERM are blocked for the duration of the teardown so a second
    Ctrl-C (or a SIGTERM) arriving mid-reap can't abort the kill loop and leak
    live GPU procs into the next trial; the pending signal is held and delivered
    once the subtree is fully reaped, then propagates as usual to the handler."""
    # Arm the block as the very first action so the whole teardown is covered. Standard signals do not queue:
    # any number of SIGINT/SIGTERM that arrive while blocked coalesce to a single pending one, so triple-,
    # quadruple-, N-Ctrl-C are all handled identically -- one KeyboardInterrupt is delivered after the reap.
    try:
        prev_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    except (ValueError, OSError):
        prev_mask = None  # unsupported / off main thread: reap unguarded rather than fail
    try:
        parent = psutil.Process()
        spare: set[int] = set()
        if spare_root is not None:
            try:
                rp = psutil.Process(spare_root)
                spare = {rp.pid, *(c.pid for c in rp.children(recursive=True))}
            except psutil.NoSuchProcess:
                pass
        deadline = time.time() + grace
        while True:
            procs = [p for p in parent.children(recursive=True) if p.pid not in spare]
            if not procs:
                return
            for p in procs:
                try:
                    p.kill()
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(procs, timeout=2)
            if time.time() >= deadline:
                return
    finally:
        if prev_mask is not None:
            signal.pthread_sigmask(signal.SIG_SETMASK, prev_mask)

def _run_trial_subprocess(cfg_dict: dict, spare_render_pid: int | None = None) -> None:
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc-per-node=auto",
        "-m",
        "campaign_trial_runner",
        "--cfg-json",
        json.dumps(cfg_dict),
    ]

    # Enable the NCCL flight recorder for this trial: a per-rank ring buffer of the most recent collectives
    # that is dumped on a watchdog timeout, so a hang leaves a trace naming which collective each rank was
    # stuck on (and its state: scheduled/started/completed) — far more than the one-line "last enqueued/
    # completed" the crash log otherwise gives. Dumps go under the campaign dir (survives trial-dir wipes on
    # resume); the per-trial prefix keeps trials from clobbering each other. Analyze with `torchfrtrace`.
    dpath_traces = _dpath_campaign(cfg_dict["campaign"]) / "nccl_traces"
    dpath_traces.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("TORCH_NCCL_TRACE_BUFFER_SIZE", "2000")  # collectives retained per rank
    env.setdefault("TORCH_NCCL_DUMP_ON_TIMEOUT", "1")
    env.setdefault(
        "TORCH_NCCL_DEBUG_INFO_TEMP_FILE",
        str(dpath_traces / f"{cfg_dict['setting']}_{cfg_dict['dataset']}_{cfg_dict['seed']}_rank"),
    )

    # start_new_session isolates torchrun from the terminal's Ctrl-C so the
    # campaign drives teardown itself (via _reap_subtree) rather than racing
    # torchrun's own signal handling.
    proc = subprocess.Popen(
        cmd,
        stdout=None,
        stderr=subprocess.PIPE,
        start_new_session=True,
        env=env,
    )

    # Drain stderr in a thread: when a rank is SIGKILLed (e.g. OOM), its
    # DataLoader workers are orphaned but keep the stderr pipe's write end open,
    # so a read-until-EOF loop in this process would hang forever even after
    # torchrun itself exits.
    stderr_chunks: list[bytes] = []
    def _drain() -> None:
        assert proc.stderr is not None
        while chunk := proc.stderr.read1(4096):
            sys.stderr.buffer.write(chunk)
            sys.stderr.buffer.flush()
            stderr_chunks.append(chunk)

    reader = threading.Thread(target=_drain, daemon=True)
    reader.start()

    try:
        return_code = proc.wait()
    finally:
        # Tear down the entire descendant subtree on any exit from the wait.
        # Covers normal crash recovery (orphaned DataLoader workers left by a
        # SIGKILLed rank, which would otherwise keep leaking into the cgroup
        # memory budget of later trials) and Ctrl-C / SIGTERM of the campaign
        # (the detached trial would otherwise keep running). A process-group
        # kill is insufficient: elastic puts each rank in its own session.
        _reap_subtree(spare_root=spare_render_pid)

    # Bounded: don't re-hang if a worker is wedged in uninterruptible sleep and
    # still holding the pipe; the daemon thread is torn down at interpreter exit.
    reader.join(timeout=30)

    if return_code != 0:
        stderr_data = b"".join(stderr_chunks)
        stderr_body = "\n".join(stderr_data.decode(errors="replace").splitlines())
        raise subprocess.CalledProcessError(return_code, cmd, stderr=stderr_body)

def _check_trial_completion(dpath_trial: Path) -> bool:
    fpath_metadata_trial = dpath_trial / "trial_metadata.json"
    if not fpath_metadata_trial.exists():
        complete = False
    else:
        metadata_trial = load_json(fpath_metadata_trial)
        complete = metadata_trial["complete"]
    return complete

def _mark_trial_complete(dpath_trial: Path) -> None:
    # marked only after a clean subprocess exit + cleanup, so a mid-finalization crash never leaves a
    # trial falsely flagged complete (which would make a re-run skip it instead of resuming)
    fpath_metadata_trial = dpath_trial / "trial_metadata.json"
    metadata_trial = load_json(fpath_metadata_trial)
    metadata_trial["complete"] = True
    save_json(metadata_trial, fpath_metadata_trial)

def _spawn_render(trial_rel: str, render_evo: bool) -> subprocess.Popen:
    """Spawn the post-trial manifold-viz render as a detached, CPU-only process so it overlaps the next
    trial's training. It renders purely from the trial's cached projections.npz (no GPU/DDP), using the
    campaign's frozen config snapshot. CUDA_VISIBLE_DEVICES is cleared so it never contends for the GPUs,
    and RENDER_MAX_WORKERS caps its CPU fan-out to a quarter of the cores so it doesn't oversubscribe the
    next trial's dataloaders -- the render has the whole next trial to finish, so it can afford to go slow."""
    cmd = [sys.executable, "-m", "tools.manifold_viz", trial_rel, "snapshot"]
    if not render_evo:
        cmd.append("no_evo")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    env.setdefault("RENDER_MAX_WORKERS", str(max(1, len(os.sched_getaffinity(0)) // 4)))
    return subprocess.Popen(cmd, env=env, start_new_session=True)

def _raise_interrupt(signum, frame) -> None:
    raise KeyboardInterrupt

def run_campaign(campaign: str, n_trials: int, datasets: list[str], baseline_overrides: list[list[dict]]) -> None:
    # Validate the planned matrix before any side effects: every setting's name must be unique.
    settings = _expand_settings(baseline_overrides)
    seeds = _iter_seeds(n_trials)

    _enable_child_subreaper()
    # Route SIGTERM (e.g. `kill`, SLURM scancel) through the same path as Ctrl-C
    # so the trial's subtree is torn down before the campaign exits.
    signal.signal(signal.SIGTERM, _raise_interrupt)

    time_data = {
        "last_updated": time.time(),
        "elapsed": 0.0,
    }
    dpath_campaign = _dpath_campaign(campaign)
    dpath_campaign.mkdir(parents=True, exist_ok=True)
    save_pickle(time_data, dpath_campaign / "time.pkl")

    n_gpus = torch.cuda.device_count()
    slurm_alloc = get_slurm_alloc()
    setting_names = [name for name, _ in settings]
    fpath_meta = dpath_campaign / "campaign_metadata.json"
    if fpath_meta.exists():
        metadata_camp = load_json(fpath_meta)
        if metadata_camp["n_gpus"] != n_gpus:
            raise RuntimeError(
                f"GPU count mismatch: campaign '{campaign}' was run with "
                f"{metadata_camp['n_gpus']} GPUs but current environment has {n_gpus}."
            )
        # campaign matrix is additive across runs: items may be added but never removed
        _check_no_removals(campaign, metadata_camp, setting_names, datasets, seeds)
    else:
        metadata_camp = {
            "duration": "0-00:00:00",
            "n_gpus": n_gpus,
            "n_cpus": slurm_alloc["n_cpus"],
            "ram": slurm_alloc["ram"],
        }
    # record the (possibly grown) planned matrix so the next run can detect removals
    metadata_camp["settings"] = setting_names
    metadata_camp["datasets"] = list(datasets)
    metadata_camp["seeds"] = seeds
    save_json(metadata_camp, fpath_meta)

    cfg_snapshot = _load_or_create_campaign_config(campaign)
    cfg_baseline = cfg_snapshot["train"]
    cfg_manifold_viz = cfg_snapshot["manifold_viz"]
    cfg_model_specific = cfg_snapshot["model_specific"]
    cfg_hardware = cfg_snapshot["hardware"]
    max_retries = cfg_hardware["max_retries"]  # consecutive no-progress trial retries before giving up

    for setting, setting_payload in settings:
        _write_setting_overrides(campaign, setting, setting_payload)

    n_trials_total = len(seeds) * len(datasets) * len(settings)
    print(f"Campaign: '{campaign}' ({n_trials_total} trials)")

    trials = [
        (setting, dataset, seed)
        for seed in seeds
        for dataset in datasets
        for setting, _ in settings
    ]
    PrintLog.manifest(dpath_campaign, trials, in_progress=None)

    render_proc: subprocess.Popen | None = None
    render_evo = cfg_baseline["dev"]["traintime_evals"]  # evolution GIFs need mid-evals to evolve across

    idx_trial = 0
    for idx_seed, seed in enumerate(seeds):
        for dataset in datasets:
            for setting, setting_payload in settings:
                idx_trial += 1

                dpath_trial = _dpath_campaign(campaign) / "settings" / setting / dataset / str(seed)
                if _check_trial_completion(dpath_trial):
                    print(f"[{idx_trial}/{n_trials_total}] SKIP (completed): {setting}/{dataset}/{seed}")
                    continue

                cfg_dict = deepcopy(cfg_baseline)
                cfg_dict["campaign"] = campaign
                cfg_dict["setting"] = setting
                cfg_dict["seed"] = seed
                cfg_dict["dataset"] = dataset
                cfg_dict["idx_seed"] = idx_seed
                cfg_dict["manifold_viz"] = cfg_manifold_viz
                cfg_dict["model_specific"] = cfg_model_specific
                cfg_dict["hw"] = cfg_hardware
                cfg_dict["_setting_overrides"] = setting_payload
                cfg_dict = apply_overrides(cfg_dict, setting_payload)

                if dpath_trial.exists():
                    print(f"[{idx_trial}/{n_trials_total}] RESUME: {setting}/{dataset}/{seed}")
                else:
                    print(f"[{idx_trial}/{n_trials_total}] {setting}/{dataset}/{seed}")

                PrintLog.manifest(dpath_campaign, trials, in_progress=(setting, dataset, seed))
                spare_pid = render_proc.pid if render_proc is not None and render_proc.poll() is None else None

                # Retry-with-resume loop: a crash mid-training costs only the work since the last checkpoint,
                # not the whole trial. `stalled` counts consecutive attempts that didn't advance the
                # checkpoint; any attempt that does reset it, so distinct flakes recover indefinitely.
                fpath_ckpt = dpath_trial / "chkpts/in_progress/train_state.pt"
                stalled = 0
                succeeded = False
                while True:
                    ckpt_mtime = fpath_ckpt.stat().st_mtime if fpath_ckpt.exists() else -1.0
                    try:
                        _run_trial_subprocess(cfg_dict, spare_render_pid=spare_pid)
                        shutil.rmtree(dpath_trial / "chkpts/in_progress")
                        _mark_trial_complete(dpath_trial)
                        PrintLog.manifest(dpath_campaign, trials, in_progress=None)
                        succeeded = True
                        break
                    except KeyboardInterrupt:
                        print(
                            f"\n[{idx_trial}/{n_trials_total}] INTERRUPTED — terminated trial process group; exiting campaign.",
                            flush=True,
                        )
                        if render_proc is not None and render_proc.poll() is None:
                            render_proc.terminate()
                        PrintLog.manifest(dpath_campaign, trials, in_progress=None)
                        return
                    except Exception as e:
                        made_progress = fpath_ckpt.exists() and fpath_ckpt.stat().st_mtime > ckpt_mtime
                        stalled = 0 if made_progress else stalled + 1
                        if stalled > max_retries:
                            _log_trial_error(
                                dpath_trial=dpath_trial,
                                idx_trial=idx_trial,
                                n_trials=n_trials_total,
                                seed=seed,
                                dataset=dataset,
                                setting=setting,
                                exc=e,
                            )
                            PrintLog.manifest(dpath_campaign, trials, in_progress=None)
                            break
                        reason = "resumed past last checkpoint" if made_progress else f"no progress {stalled}/{max_retries}"
                        print(
                            f"\n[{idx_trial}/{n_trials_total}] TRIAL FAILED ({reason}) — retrying with resume: {setting}/{dataset}/{seed}",
                            flush=True,
                        )
                        PrintLog.manifest(dpath_campaign, trials, in_progress=(setting, dataset, seed))

                if not succeeded:
                    continue

                # Render this trial's manifold viz off-process (CPU-only), overlapping the next trial's
                # training. At most one render in flight: wait on the prior one first (near-instant in
                # practice, since a trial far outlasts a render).
                if render_proc is not None and render_proc.poll() is None:
                    render_proc.wait()
                render_proc = _spawn_render(f"{campaign}/settings/{setting}/{dataset}/{seed}", render_evo)

    # let the last trial's render finish before the campaign exits
    if render_proc is not None:
        try:
            render_proc.wait()
        except KeyboardInterrupt:
            render_proc.terminate()


def _parse_campaign_name(argv: list[str]) -> str:
    if len(argv) != 1:
        avail = ", ".join(sorted(p.stem for p in (paths["config"] / "camps").glob("*.yaml")))
        raise SystemExit(f"Usage: python -m campaign_runner --<campaign>\nAvailable campaigns: {avail}")
    return argv[0].lstrip("-")

def _load_campaign_config(name: str) -> dict:
    fpath = paths["config"] / "camps" / f"{name}.yaml"
    if not fpath.exists():
        avail = ", ".join(sorted(p.stem for p in (paths["config"] / "camps").glob("*.yaml")))
        raise SystemExit(f"Campaign config not found: {fpath}\nAvailable campaigns: {avail}")
    with open(fpath) as f:
        return yaml.safe_load(f)

def main() -> None:
    name = _parse_campaign_name(sys.argv[1:])
    cfg = _load_campaign_config(name)
    suffix = cfg["suffix"]
    campaign = f"{name}_{suffix}" if suffix is not None else name
    run_campaign(
        campaign=campaign,
        n_trials=cfg["n_trials"],
        datasets=cfg["datasets"],
        baseline_overrides=cfg["baseline_overrides"],
    )


if __name__ == "__main__":
    main()