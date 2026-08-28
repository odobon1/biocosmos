"""
python -m campaign_runner
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m campaign_runner

Campaign execution is driven by the queue in config/camp_queue.yaml: its `campaigns` list names the
runs, in order -- camp.<name> runs the campaign defined by config/camps/<name>.yaml. The queue
file is re-read after every campaign, so entries may be added (at any position) while one runs;
the runner exits once every listed entry has been run.
"""

from pathlib import Path
from copy import deepcopy
from decimal import Decimal
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

from utils.config import (
    CFG_PARAM_ALIASES, 
    CFG_PARAM_VALUE_ALIASES, 
    CFG_UNIVERSAL_VALUE_ALIASES,
    CampaignConfig,
    apply_overrides, 
    apply_train_debug_overrides, 
    get_config_stats, 
    get_config_train, 
    load_train_config_dict, 
    load_manif_viz_config_dict, 
    load_model_specific_config_dict, 
    load_dataset_specific_config_dict, 
    load_hardware_config_dict
)
from utils.data import stage_img_cache
from utils.hardware import get_slurm_alloc
from utils.report import update_arm_stats, update_dataset_stats, update_campaign_stats, pick_best_coords
from utils.train import ArtifactManager
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

def _log_trial_error(dpath_trial: Path, idx_trial: int, n_trials: int, seed: int, dataset: str, arm: str, coord: str,
                     exc: Exception, failure: str) -> None:
    """Log trial error to stdout and to error.log in the trial-seed's directory. `failure` is the
    aggregate cause over the fatal retry loop's crashes -- RAM / VRAM / Other when every crash was
    that one kind, Mixed when they span kinds -- written as a 'failure=' marker that
    PrintLog.manifest parses for the Failed section."""
    dpath_trial.mkdir(parents=True, exist_ok=True)
    fpath_error = dpath_trial / "error.log"
    # Format error message with context
    error_msg = (
        f"\n[{idx_trial}/{n_trials}] TRIAL FAILED\n"
        f"  seed={seed}, dataset={dataset}, arm={arm}, coord={coord}, failure={failure}"
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

def _log_crash(dpath_trial: Path, exc: Exception) -> None:
    """Write a per-crash error log under <trial>/errors/, one file per crash (recovered *or* fatal),
    creating the dir on demand so it exists only once a crash has actually happened. Files are named
    error-<n_samps_seen>-<k>.log: <n_samps_seen> is the trial's last-checkpointed sample count read from
    trial_metadata.json (the crashed subprocess is gone, so progress comes off disk; 0 before the first
    checkpoint), and <k> is the next free index for that count (0 for the first crash there, so repeated
    crashes at the same checkpoint don't clobber). The fatal crash is additionally summarized in the
    trial-root error.log by _log_trial_error, which the manifest keys the 'Failed' bucket off of."""
    fpath_meta = dpath_trial / "trial_metadata.json"
    n_samps = load_json(fpath_meta)["progress"]["n_samps_seen"] if fpath_meta.exists() else 0
    dpath_errors = dpath_trial / "errors"
    dpath_errors.mkdir(parents=True, exist_ok=True)
    idxs = [int(p.stem.rsplit("-", 1)[1]) for p in dpath_errors.glob(f"error-{n_samps}-*.log")]
    k = max(idxs) + 1 if idxs else 0
    stderr = exc.stderr if isinstance(exc, subprocess.CalledProcessError) else None
    body = _relevant_stderr(stderr) if stderr else traceback.format_exc()
    (dpath_errors / f"error-{n_samps}-{k}.log").write_text(body + "\n")

def _classify_crash(exc: Exception) -> str:
    """Bucket a trial crash by cause, from the subprocess's captured stderr (falls back to the
    exception text): 'vram' = CUDA OOM (a rank raised torch.OutOfMemoryError); 'ram' = cgroup OOM
    (the kernel SIGKILLed a rank or DataLoader worker over the job's RAM limit); 'other' = the rest.
    VRAM is checked first: a CUDA OOM's teardown can drag SIGKILL noise into stderr, but the
    reverse doesn't happen (a cgroup kill leaves no CUDA OOM traceback)."""
    stderr = exc.stderr if isinstance(exc, subprocess.CalledProcessError) else None
    text = stderr if stderr else str(exc)
    if "CUDA out of memory" in text or "torch.OutOfMemoryError" in text:
        return "vram"
    if "SIGKILL" in text or "killed by signal: Killed" in text:
        return "ram"
    return "other"

def _render_phase_tables(campaign: str, phase: str) -> None:
    """Re-render every cross-coord level's tables/plots/workbooks (arm_stats, dataset_stats, campaign_stats) of
    the phase from whatever is on disk, over its recorded matrix. Trials render each level only when a seed
    completes across that level's cycle (train.py), so a phase that ends mid-cycle -- one interrupted, or with
    a (dataset, arm, coord) that never succeeds -- would otherwise leave them a cycle behind. Checkpoint
    selection is NOT redone: every completed trial already reselected its own (dataset, arm, coord) at its own
    trial end."""
    if phase == "trainval":  # runs no evals -> nothing to select or aggregate
        return
    cfg_stats = get_config_stats()
    ArtifactManager.dpath_phase = _dpath_phase(campaign, phase)
    matrix = load_json(ArtifactManager.dpath_phase / "campaign_metadata.json")["matrix"]
    style = (cfg_stats.spread_type, cfg_stats.bold_high, cfg_stats.ordered, cfg_stats.heatmap, cfg_stats.supp_scores)
    for dataset, arms in matrix.items():
        for arm in arms:
            update_arm_stats(dataset, arm, *style)
        update_dataset_stats(dataset, *style)
    update_campaign_stats(*style, cfg_stats.overrides)

def _bump_crash_counts(dpath_trial: Path, dpath_phase: Path, kind: str) -> None:
    """Increment n_crashes[kind] ('ram' | 'vram' | 'other', see _classify_crash) at the trial,
    coord, and campaign levels. The three counters are bumped independently rather than re-summed
    from the trials, so the coord and campaign totals stay accurate even when a no-progress restart
    wipes the trial dir (which resets that trial's own counts). Each file seeds the zeroed dict at
    creation (campaign at kickoff, coord/trial by the subprocess), so a bump is a plain
    read-increment-save; the coord/trial files are guarded because a crash can precede the
    subprocess writing them, whereas campaign_metadata.json always exists by the time any trial runs."""
    dpath_coord = dpath_trial.parent
    for fpath in (
        dpath_trial / "trial_metadata.json",
        dpath_coord / "coord_metadata.json",
        dpath_phase / "campaign_metadata.json",
    ):
        if fpath.exists():
            metadata = load_json(fpath)
            metadata["n_crashes"][kind] += 1
            save_json(metadata, fpath)

def _dpath_campaign(campaign: str) -> Path:
    """The campaign's root dir, artifacts/<campaign>/ -- holds the phase dirs (screening/, qual/); the name-dedupe
    check keys off it."""
    return paths["artifacts"] / campaign

def _dpath_phase(campaign: str, phase: str) -> Path:
    """A phase's dir, artifacts/<campaign>/<phase>/ ('screening' | 'qual' | 'trainval'): the root of every artifact the runner and
    that phase's trials write (datasets/, campaign_stats/, campaign_metadata.json, cfg_baseline.json, manifest.log,
    time.pkl, nccl_traces/)."""
    return _dpath_campaign(campaign) / phase

def _get_commit_hash() -> str:
    """HEAD commit hash of the repo this runner lives in, for campaign provenance."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

def _load_or_create_campaign_config(campaign: str) -> dict:
    """Load the campaign's frozen config snapshot, creating it on first launch.

    On first launch five config sources are bundled into a single `artifacts/<campaign>/screening/cfg_baseline.json` (the qual phase carries a copy, _copy_qual_picks)
    under the keys `train`, `hardware`, `manif_viz`, `model_specific`, `dataset_specific`. The `train`
    snapshot is derived from `config/train.yaml` (with `debug_mode` overrides folded in); the other four are
    `config/hardware.yaml`, `config/manif_viz.yaml`, `config/model_specific.yaml`, and
    `config/dataset_specific.yaml` verbatim. Every trial starts from the `train` snapshot and has the
    sibling snapshots injected per trial (as `hw`, `manif_viz`, `model_specific`, `dataset_specific`).
    Model-family `opt` defaults are left unresolved in `train` (kept `null`) and filled per trial from the
    `model_specific` snapshot, so a per-arm/coord `arch.model_type` override still picks up the matching
    family's defaults; a null `n_epochs` / `n_chkpts` is likewise left unresolved and filled per trial from the
    `dataset_specific` snapshot as per the trial's dataset. Every later relaunch (resume or matrix extension) reloads that
    snapshot rather than re-reading the YAML, so edits to any config file after a campaign's first launch
    never alter that campaign -- all of its trials, original or added later, train against the same
    frozen config."""
    fpath = _dpath_phase(campaign, "screening") / "cfg_baseline.json"
    if fpath.exists():
        return load_json(fpath)

    cfg_train = apply_train_debug_overrides(load_train_config_dict())
    cfg_snapshot = {
        "train": cfg_train,
        "hardware": load_hardware_config_dict(),
        "manif_viz": load_manif_viz_config_dict(),
        "model_specific": load_model_specific_config_dict(),
        "dataset_specific": load_dataset_specific_config_dict(),
    }
    fpath.parent.mkdir(parents=True, exist_ok=True)
    save_json(cfg_snapshot, fpath)
    return cfg_snapshot

def _fmt_name_value(v) -> str:
    """Format an override value for use in an arm / coord name. Floats read like the YAML that declares
    them: any nonzero float below 1e-2 in magnitude (or one Python itself renders in scientific notation)
    is written '<shortest mantissa, with a decimal point>e<exponent, no zero-padding>' -- 0.0002 -> '2.0e-4',
    7e-06 -> '7.0e-6', 1.131e-4 -> '1.131e-4'. Python alone switches to scientific notation only below
    1e-4, which would put 'LR-0.0002' next to 'LR-2.0e-5' in an LR sweep; larger floats keep their plain
    rendering ('0.3', '0.05')."""
    if isinstance(v, float) and v != 0.0 and (abs(v) < 1e-2 or "e" in repr(v)):
        sign, digits, exp = Decimal(repr(v)).as_tuple()  # repr: the shortest round-trip digits
        mant = f"{digits[0]}.{''.join(map(str, digits[1:])) or '0'}"
        return f"{'-' if sign else ''}{mant}e{exp + len(digits) - 1}"
    return str(v)

def _alias_pair(k: str, v) -> str:
    """Render one override as a 'key-value' name component, with the key mapped through
    CFG_PARAM_ALIASES and the value through CFG_PARAM_VALUE_ALIASES (per original key), falling
    back to CFG_UNIVERSAL_VALUE_ALIASES (key-independent) when no per-key alias exists,
    e.g. ('batch_size', 2048) -> 'bs-2k'."""
    if v in CFG_PARAM_VALUE_ALIASES.get(k, {}):
        v_aliased = CFG_PARAM_VALUE_ALIASES[k][v]
    # identity guard: dict lookup uses ==, and True == 1 / False == 0 would alias numeric values
    elif v is None or isinstance(v, bool):
        v_aliased = CFG_UNIVERSAL_VALUE_ALIASES[v]
    else:
        v_aliased = v
    return f"{CFG_PARAM_ALIASES.get(k, k)}-{_fmt_name_value(v_aliased)}"

def _derive_item_name(item: dict) -> str:
    """Name an unnamed `ablation_arms` / `hpo_coords` item by its overrides: _alias_pair components
    joined by '_', e.g. {'loss.targ': 'mp', 'batch_size': 2048} -> 'Targ-MP_BS-2k'."""
    return "_".join(_alias_pair(k, v) for k, v in item.items())

def _item_name(item: dict) -> str | None:
    """The name component an item contributes: its explicit 'name', or a name derived from its
    overrides via _derive_item_name when 'name' is absent. An explicit 'name: null' returns
    None -- the item contributes no component and is skipped when member names are joined."""
    return item["name"] if "name" in item else _derive_item_name(item)

def _expand_combo_lists(item: dict) -> list[dict]:
    """Expand an item's combo lists (list-valued overrides) into scalar items, one per combination
    of list values; several combo lists in one item cross with each other, the last-listed key
    varying fastest. The chosen 'key-value' pairs always show in the name: appended to an
    explicit 'name' (e.g. {'batch_size': [1024, 2048], 'name': 'hp'} -> 'hp_BS-1k', 'hp_BS-2k'),
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

def _expand_combo_groups(combo_groups: list[list[dict]], param: str) -> list[tuple[str, dict]]:
    """Expand one camp-yaml override list (`param`: 'ablation_arms' -> the arms, 'hpo_coords' -> the
    coords) from its combo groups into the full list of (name, overrides) members.

    The list is a list of combo groups; each combo group is a list of partial members (a dict of
    dotted-key overrides plus an optional 'name'; an item without one is named from its overrides
    via _derive_item_name, e.g. {'loss.targ': 'mp'} -> 'Targ-MP'). An override value given as a
    list is a combo list: the item is first expanded into one partial member per combination of its
    list values, named per _expand_combo_lists. The members are the Cartesian product across combo
    groups: one partial member is drawn from each combo group and merged into one member, its name
    the parts' names joined by '_' in combo-group order (e.g. 'hp' x '2k' -> 'hp_2k'). A single
    combo group expands to its items unchanged. Combo groups are independent dimensions, so no
    override key may appear in more than one combo group -- a shared key would have two values
    fighting to define it when parts merge.

    An item may set 'name' explicitly to null: it then contributes no name component and is skipped
    in the join (e.g. 'hp' x null -> 'hp'). At most one item per combo group may be null (two would
    give two members the same name), and at least one combo group must have all its items named
    (else the all-null combination would yield an empty name)."""
    if not combo_groups:
        raise ValueError(f"{param} must list at least one combo group.")
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
                f"{param} key(s) {sorted(shared)} collide between combo groups {i} and {j}; "
                f"each override key must belong to exactly one combo group."
            )

    null_counts = [
        sum("name" in item and item["name"] is None for item in group)
        for group in combo_groups
    ]
    for i, n_null in enumerate(null_counts):
        if n_null > 1:
            raise ValueError(
                f"{param} combo group {i} has {n_null} items with `name: null`; at most one item per combo "
                f"group may set `name: null`."
            )
    if all(n_null > 0 for n_null in null_counts):
        raise ValueError(
            f"every {param} combo group has an item with `name: null`; at least one combo group must have "
            f"all its items named, else the all-null combination yields an empty name."
        )

    members = []
    seen_names: set[str] = set()
    for combo in itertools.product(*combo_groups):
        name = "_".join(
            part for item in combo if (part := _item_name(item)) is not None
        )
        if name in seen_names:
            raise ValueError(f"Duplicate {param} name: {name}")
        seen_names.add(name)
        payload = {k: deepcopy(v) for item in combo for k, v in item.items() if k != "name"}
        members.append((name, payload))
    return members

def _expand_matrix(ablation_arms: list[list[dict]], hpo_coords: list[list[dict]]) -> tuple[list[tuple[str, dict]], list[tuple[str, dict]]]:
    """(arms, coords): the campaign's arms expanded from `ablation_arms` and its coords from `hpo_coords`
    (each per _expand_combo_groups). Every arm is crossed with every coord, the two override sets merging
    into one trial config, so the two lists form ONE override space: a key may belong to exactly one combo
    group across both (an arm key that is also a coord key would have two values fighting to define it)."""
    arms = _expand_combo_groups(ablation_arms, "ablation_arms")
    coords = _expand_combo_groups(hpo_coords, "hpo_coords")
    shared = {k for _, payload in arms for k in payload} & {k for _, payload in coords for k in payload}
    if shared:
        raise ValueError(
            f"override key(s) {sorted(shared)} appear in both ablation_arms and hpo_coords; "
            f"each override key must belong to exactly one combo group."
        )
    return arms, coords

def _write_overrides(dpath_coord: Path, arm_payload: dict, coord_payload: dict) -> None:
    """The coord dir's overrides.json: the arm's and the coord's declared overrides, kept apart under
    'arm' / 'coord' (the stats overrides bands read each side separately)."""
    dpath_coord.mkdir(parents=True, exist_ok=True)
    with open(dpath_coord / "overrides.json", "w") as f:
        json.dump({"arm": arm_payload, "coord": coord_payload}, f, indent=2, sort_keys=True)

def _iter_seeds(n_trials: int) -> list[int]:
    return list(range(SEED0, SEED0 + n_trials))

def _check_no_removals(campaign: str, prev_meta: dict, arm_names: list[str], coord_names: list[str], datasets: list[str], seeds: list[int]) -> None:
    """A campaign's matrix may grow across runs (add arms/coords/datasets/seeds) but never shrink.
    Compare the planned matrix against the one persisted from a prior run and raise if any
    previously-run arm, coord, dataset, or seed is missing -- dropping one would orphan its
    already-computed trials and silently remove them from the campaign."""
    removed = []
    for kind, prev_vals, curr_vals in (
        ("arms", prev_meta["arms"], arm_names),
        ("coords", prev_meta["coords"], coord_names),
        ("datasets", prev_meta["datasets"], datasets),
        ("seeds", prev_meta["seeds"], seeds),
    ):
        missing = [v for v in prev_vals if v not in set(curr_vals)]
        if missing:
            removed.append(f"  {kind} removed: {missing}")
    if removed:
        raise RuntimeError(
            f"Campaign '{campaign}' config drops items recorded by a prior run "
            f"(arms/coords/datasets/seeds may be added across runs but never removed):\n"
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

def _stash_nccl_dumps(dpath_phase: Path) -> None:
    # Tuck any flight-recorder dump files from the phase dir into nccl_traces/,
    # creating the dir only when a dump actually exists.
    fpaths = list(dpath_phase.glob("nccl_trace_*"))
    if not fpaths:
        return
    dpath_traces = dpath_phase / "nccl_traces"
    dpath_traces.mkdir(exist_ok=True)
    for fpath in fpaths:
        fpath.rename(dpath_traces / fpath.name.removeprefix("nccl_trace_"))

def _build_trial_cfg_dict(cfg_snapshot: dict, campaign: str, phase: str, arm: str, coord: str, overrides: dict,
                          seed: int, dataset: str, idx_seed: int,
                          idx_trial: int | None = None, n_trials_total: int | None = None, injections: dict | None = None) -> dict:
    """Effective per-trial config dict: frozen campaign snapshot + trial identity (incl. the phase whose tree the
    trial writes to) + `injections` (phase-level settings laid over the snapshot -- the trainval phase's train_pt /
    chkpt_stop; not overrides, so not recorded as such) + the merged arm + coord overrides."""
    cfg_dict = deepcopy(cfg_snapshot["train"])
    cfg_dict["campaign"] = campaign
    cfg_dict["phase"] = phase
    cfg_dict["arm"] = arm
    cfg_dict["coord"] = coord
    cfg_dict["seed"] = seed
    cfg_dict["dataset"] = dataset
    cfg_dict["idx_seed"] = idx_seed
    cfg_dict["idx_trial"] = idx_trial
    cfg_dict["n_trials_total"] = n_trials_total
    cfg_dict["manif_viz"] = cfg_snapshot["manif_viz"]
    cfg_dict["model_specific"] = cfg_snapshot["model_specific"]
    cfg_dict["dataset_specific"] = cfg_snapshot["dataset_specific"]
    cfg_dict["hw"] = cfg_snapshot["hardware"]
    cfg_dict["_overrides"] = overrides
    if injections:
        cfg_dict.update(injections)
    return apply_overrides(cfg_dict, overrides)

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
    # completed" the crash log otherwise gives. The C++ writer won't create parent dirs (a dump into a
    # missing dir is silently lost), so dumps target a prefix at the phase dir (always exists, survives
    # trial-dir wipes on resume) and are stashed into nccl_traces/ post-trial — the dir exists only if some
    # trial actually dumped. The per-trial prefix keeps trials from clobbering each other. Analyze with
    # `torchfrtrace`.
    dpath_phase = _dpath_phase(cfg_dict["campaign"], cfg_dict["phase"])
    env = os.environ.copy()
    env.setdefault("TORCH_NCCL_TRACE_BUFFER_SIZE", "2000")  # collectives retained per rank
    env.setdefault("TORCH_NCCL_DUMP_ON_TIMEOUT", "1")
    env.setdefault(
        "TORCH_NCCL_DEBUG_INFO_TEMP_FILE",
        str(dpath_phase / f"nccl_trace_{cfg_dict['arm']}_{cfg_dict['coord']}_{cfg_dict['dataset']}_{cfg_dict['seed']}_rank"),
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
        _stash_nccl_dumps(dpath_phase)

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

def _spawn_render(trial_rel: str) -> subprocess.Popen:
    """Spawn the post-trial manifold-viz render as a detached, CPU-only process so it overlaps the next
    trial's training. It renders purely from the trial's cached projections.npz (no GPU/DDP), using the
    campaign's frozen config snapshot. CUDA_VISIBLE_DEVICES is cleared so it never contends for the GPUs,
    and the worker is held to a quarter of the cores so it doesn't oversubscribe the next trial's
    dataloaders -- the render has the whole next trial to finish, so it can afford to go slow. That core
    budget is enforced twice over: RENDER_MAX_WORKERS caps the plot-job process fan-out, and the
    numba/BLAS thread caps hold the UMAP stage (NN-descent + layout run numba-parallel, the pooled PCA
    runs on BLAS), which runs single-process BEFORE that fan-out and would otherwise burst to every core."""
    cmd = [sys.executable, "-m", "tools.regen_manif_viz", trial_rel, "snapshot"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    cap = str(max(1, len(os.sched_getaffinity(0)) // 4))
    for var in ("RENDER_MAX_WORKERS", "NUMBA_NUM_THREADS", "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(var, cap)
    return subprocess.Popen(cmd, env=env, start_new_session=True)

def _trial_has_manif_cache(dpath_trial: Path) -> bool:
    """Whether this trial actually produced any manifold-viz cache worth post-trial rendering.

    Most trials may sit outside the manif_viz seed window; those write no projections/embeddings, so
    spawning the detached render worker would just start Python to discover there is nothing to do.
    """
    dpath_evals = dpath_trial / "evals"
    if not dpath_evals.exists():
        return False
    for d in dpath_evals.iterdir():
        if any((d / name).exists() for name in ("projections.npz", "projections_pooled.npz", "embs.npz")):
            return True
    return False

def _raise_interrupt(signum, frame) -> None:
    raise KeyboardInterrupt

def _del_base_eval_cache() -> None:
    dpath = paths["root"] / "base_eval_cache"
    if dpath.exists():
        shutil.rmtree(dpath)
        print("deleted base_eval_cache/ (dev.del_base_eval_cache)", flush=True)

def _phase_metadata(campaign: str, dpath_phase: Path, arm_names: list[str], coord_names: list[str], datasets: list[str],
                    seeds: list[int], matrix: dict) -> tuple[dict, Path]:
    """Load (or, on the phase's first launch, create) the phase's campaign_metadata.json and record its planned
    matrix; returns (metadata, its path). `matrix` is {dataset: {arm: [coords]}} -- the phase's planned (dataset,
    arm, coord) combos in campaign order: every coord under every arm for the screening phase, each arm's picked
    coord for the qual and trainval phases -- the shape the stats code keys its sweep gates and table rows off. The
    GPU count must match the phase's first launch, and the planned matrix may only grow across launches
    (_check_no_removals)."""
    n_gpus = torch.cuda.device_count()
    slurm_alloc = get_slurm_alloc()
    fpath_meta = dpath_phase / "campaign_metadata.json"
    if fpath_meta.exists():
        metadata = load_json(fpath_meta)
        if metadata["n_gpus"] != n_gpus:
            raise RuntimeError(
                f"GPU count mismatch: campaign '{campaign}' was run with "
                f"{metadata['n_gpus']} GPUs but current environment has {n_gpus}."
            )
        # campaign matrix is additive across runs: items may be added but never removed
        _check_no_removals(campaign, metadata, arm_names, coord_names, datasets, seeds)
    else:
        metadata = {
            "duration": "0-00:00:00",
            "commit": _get_commit_hash(),  # repo HEAD at first launch; not updated by relaunches
            "n_gpus": n_gpus,
            "n_cpus": slurm_alloc["n_cpus"],
            "ram": slurm_alloc["ram"],
            "memory": {"ram": None, "vram": None},  # peak used/total, max-merged across trials
            "n_crashes": {"ram": 0, "vram": 0, "other": 0},  # running totals of crashes across all trials, bucketed by cause (see _classify_crash / _bump_crash_counts)
        }
    # record the (possibly grown) planned matrix so the next run can detect removals
    metadata["arms"] = arm_names
    metadata["coords"] = coord_names
    metadata["datasets"] = list(datasets)
    metadata["seeds"] = seeds
    metadata["matrix"] = matrix
    save_json(metadata, fpath_meta)
    return metadata, fpath_meta

def _run_phase(campaign: str, phase: str, cfg_snapshot: dict, arms: list[tuple[str, dict]], coords: list[tuple[str, dict]],
               datasets: list[str], seeds: list[int], matrix: dict, chkpt_stops: dict | None = None) -> bool:
    """Run one phase's trial matrix under artifacts/<campaign>/<phase>/: `matrix` ({dataset: {arm: [coords]}}, the
    phase's planned (dataset, arm, coord) combos in campaign order; see _phase_metadata) x `seeds`, seed-major
    then dataset, arm, coord. `arms` / `coords` are the (name, overrides) members the matrix draws on. Every
    artifact of the phase -- its campaign_metadata.json, time.pkl, manifest.log, the coord dirs, the stats
    trees -- lives under its dir. Completed trials are skipped, so a relaunch resumes/extends the phase.
    `chkpt_stops` ({(dataset, arm, coord): checkpoint index}, the trainval phase) trains every combo on the
    trainval partition and stops it at its index (TrainConfig.chkpt_stop) instead of running to sample_volume.
    Returns 'complete' once every planned trial of the phase is complete, 'incomplete' when the run finished but
    some trial failed for good (the campaign stops at this phase -- see run_campaign), 'interrupted' on Ctrl-C /
    SIGTERM."""
    dpath_phase = _dpath_phase(campaign, phase)
    dpath_phase.mkdir(parents=True, exist_ok=True)
    save_pickle({"last_updated": time.time(), "elapsed": 0.0}, dpath_phase / "time.pkl")

    arm_payloads, coord_payloads = dict(arms), dict(coords)
    metadata, fpath_meta = _phase_metadata(campaign, dpath_phase, list(arm_payloads), list(coord_payloads), datasets, seeds, matrix)
    combos = [(dataset, arm, coord) for dataset in datasets for arm in arm_payloads for coord in matrix[dataset][arm]]

    def injections(dataset, arm, coord):
        return {"train_pt": "trainval", "chkpt_stop": chkpt_stops[(dataset, arm, coord)]} if chkpt_stops is not None else {}

    cfg_hardware = cfg_snapshot["hardware"]
    max_retries = cfg_hardware["max_retries"]  # consecutive no-progress trial retries before giving up

    # Fail-fast config validation: construct every (dataset, arm, coord)'s effective TrainConfig now, so a
    # misconfigured combination (e.g. a batch_size that doesn't band-shard over world_size x loss_chunk_size)
    # kills the campaign at kickoff instead of erroring when its trial finally launches. Seed only needs
    # to be representative -- config validation is seed-independent beyond requiring a non-null seed.
    for dataset, arm, coord in combos:
        cfg_dict = _build_trial_cfg_dict(cfg_snapshot, campaign, phase, arm, coord, {**arm_payloads[arm], **coord_payloads[coord]},
                                         seeds[0], dataset, 0, injections=injections(dataset, arm, coord))
        try:
            get_config_train(cfg_dict=cfg_dict)
        except Exception as e:
            raise ValueError(f"invalid config for arm '{arm}' / coord '{coord}' on dataset '{dataset}': {e}") from e

    # Node-local image-cache staging, up front: fail fast (before any trial) if a pack is missing, and record
    # per-dataset staging seconds. null = dataset unused this campaign, or img caching off in every arm and
    # coord. Effective per trial = frozen hw baseline overlaid with its arm's / coord's hw.use_img_cache
    # override, so an override-level enable is still checked/staged at startup rather than erroring
    # mid-campaign.
    metadata["runtime_img_cache"] = {ds: None for ds in sorted(paths["imgs"])}
    use_img_cache = any(
        payload.get("hw.use_img_cache", cfg_hardware["use_img_cache"]) for _, payload in [*arms, *coords]
    )
    if use_img_cache:
        missing = [ds for ds in datasets if not (paths["img_cache"] / ds / "meta.json").exists()]
        if missing:
            raise FileNotFoundError(
                f"use_img_cache is enabled but no image pack exists for {missing} under {paths['img_cache']} "
                f"-- build first: python -m tools.build_img_cache"
            )
        for dataset in datasets:
            metadata["runtime_img_cache"][dataset] = round(stage_img_cache(dataset), 2)
    save_json(metadata, fpath_meta)

    n_trials_total = len(seeds) * len(combos)
    print(f"Campaign: '{campaign}' {phase} ({n_trials_total} trials)")

    trials = [(dataset, arm, coord, seed) for seed in seeds for dataset, arm, coord in combos]
    PrintLog.manifest(dpath_phase, trials, in_progress=None)

    del_base_eval_cache_trial = cfg_snapshot["train"]["dev"]["del_base_eval_cache"]["trial"]
    render_proc: subprocess.Popen | None = None

    idx_trial = 0
    for idx_seed, seed in enumerate(seeds):
        for dataset, arm, coord in combos:
            idx_trial += 1

            dpath_coord = dpath_phase / "datasets" / dataset / "arms" / arm / "coords" / coord
            dpath_trial = dpath_coord / str(seed)
            trial_id = f"{dataset}/{arm}/{coord}/{seed}"
            if _check_trial_completion(dpath_trial):
                print(f"[{idx_trial}/{n_trials_total}] SKIP (completed): {trial_id}")
                continue

            # the coord dir (and with it the arm dir) is created here, at trial launch, not at campaign
            # kickoff -- a planned arm/coord whose trials never start leaves no
            # artifacts/<campaign>/<phase>/datasets/<dataset>/arms/ entry
            _write_overrides(dpath_coord, arm_payloads[arm], coord_payloads[coord])

            cfg_dict = _build_trial_cfg_dict(cfg_snapshot, campaign, phase, arm, coord, {**arm_payloads[arm], **coord_payloads[coord]},
                                             seed, dataset, idx_seed, idx_trial, n_trials_total, injections=injections(dataset, arm, coord))

            if dpath_trial.exists():
                print(f"[{idx_trial}/{n_trials_total}] RESUME: {trial_id}")
            else:
                print(f"[{idx_trial}/{n_trials_total}] {trial_id}")

            if del_base_eval_cache_trial:
                _del_base_eval_cache()

            PrintLog.manifest(dpath_phase, trials, in_progress=(dataset, arm, coord, seed))
            spare_pid = render_proc.pid if render_proc is not None and render_proc.poll() is None else None

            # Retry-with-resume loop: a crash mid-training costs only the work since the last checkpoint,
            # not the whole trial. `stalled` counts consecutive attempts that didn't advance the
            # checkpoint; any attempt that does reset it, so distinct flakes recover indefinitely.
            fpath_ckpt = dpath_trial / "chkpts/in_progress/train_state.pt"
            stalled = 0
            crash_kinds = []  # every crash kind across this trial's retry loop, for the fatal failure= label
            succeeded = False
            while True:
                ckpt_mtime = fpath_ckpt.stat().st_mtime if fpath_ckpt.exists() else -1.0
                try:
                    _run_trial_subprocess(cfg_dict, spare_render_pid=spare_pid)
                    shutil.rmtree(dpath_trial / "chkpts")  # only holds in_progress/ -- no weights are saved
                    _mark_trial_complete(dpath_trial)
                    PrintLog.manifest(dpath_phase, trials, in_progress=None)
                    succeeded = True
                    break
                except KeyboardInterrupt:
                    print(
                        f"\n[{idx_trial}/{n_trials_total}] INTERRUPTED — terminated trial process group; exiting campaign.",
                        flush=True,
                    )
                    if render_proc is not None and render_proc.poll() is None:
                        render_proc.terminate()
                    _render_phase_tables(campaign, phase)
                    PrintLog.manifest(dpath_phase, trials, in_progress=None)
                    return "interrupted"
                except Exception as e:
                    _log_crash(dpath_trial, e)
                    kind = _classify_crash(e)
                    _bump_crash_counts(dpath_trial, dpath_phase, kind)
                    crash_kinds.append(kind)
                    made_progress = fpath_ckpt.exists() and fpath_ckpt.stat().st_mtime > ckpt_mtime
                    stalled = 0 if made_progress else stalled + 1
                    if stalled > max_retries:
                        # a no-progress restart wipes the trial dir (metadata + errors/), so the
                        # in-loop crash_kinds is the only record covering ALL of this loop's crashes
                        kinds = set(crash_kinds)
                        failure = {"ram": "RAM", "vram": "VRAM", "other": "Other"}[next(iter(kinds))] if len(kinds) == 1 else "Mixed"
                        _log_trial_error(
                            dpath_trial=dpath_trial,
                            idx_trial=idx_trial,
                            n_trials=n_trials_total,
                            seed=seed,
                            dataset=dataset,
                            arm=arm,
                            coord=coord,
                            exc=e,
                            failure=failure,
                        )
                        PrintLog.manifest(dpath_phase, trials, in_progress=None)
                        break
                    reason = "resumed past last checkpoint" if made_progress else f"no progress {stalled}/{max_retries}"
                    print(
                        f"\n[{idx_trial}/{n_trials_total}] TRIAL FAILED ({reason}) — retrying with resume: {trial_id}",
                        flush=True,
                    )
                    PrintLog.manifest(dpath_phase, trials, in_progress=(dataset, arm, coord, seed))

            if not succeeded:
                continue

            # Render this trial's manifold viz off-process (CPU-only), overlapping the next trial's
            # training, but only when this trial actually produced manifold caches. Trials outside the
            # manif_viz seed window have nothing to render, so skip the extra Python process entirely.
            # At most one render in flight: wait on the prior one only when a new render is about to
            # start.
            if _trial_has_manif_cache(dpath_trial):
                if render_proc is not None and render_proc.poll() is None:
                    render_proc.wait()
                render_proc = _spawn_render(f"{campaign}/{phase}/datasets/{dataset}/arms/{arm}/coords/{coord}/{seed}")

    _render_phase_tables(campaign, phase)

    # let the last trial's render finish before the phase exits
    if render_proc is not None:
        try:
            render_proc.wait()
        except KeyboardInterrupt:
            render_proc.terminate()
            return "interrupted"

    n_failed = sum(
        not _check_trial_completion(dpath_phase / "datasets" / dataset / "arms" / arm / "coords" / coord / str(seed))
        for dataset, arm, coord, seed in trials
    )
    if n_failed:
        print(f"Campaign: '{campaign}' {phase} incomplete -- {n_failed} trial(s) failed (see its manifest.log); "
              f"the next phase is not started. Relaunch to resume them.", flush=True)
        return "incomplete"
    return "complete"

def _qual_picks(campaign: str, datasets: list[str], arm_names: list[str]) -> dict[tuple[str, str], str]:
    """{(dataset, arm): coord}: the coord each arm goes into the qual phase with, per dataset. Picks are frozen
    in the qual phase's campaign_metadata.json matrix at its first launch -- a relaunch reuses them even if the
    screening results have since moved (added seeds/coords) -- and a (dataset, arm) without a recorded pick
    (the first qual launch, or an arm/dataset added later) is picked fresh from the screening tree: the coord
    with the highest across-trial mean Native mAP composite All score at its selected checkpoint
    (report.pick_best_coords; every arm has one, the screening phase having completed in full)."""
    fpath_meta = _dpath_phase(campaign, "qual") / "campaign_metadata.json"
    recorded = load_json(fpath_meta)["matrix"] if fpath_meta.exists() else {}
    ArtifactManager.dpath_phase = _dpath_phase(campaign, "screening")
    fresh = pick_best_coords()  # {(arm, dataset): coord}
    return {
        (dataset, arm): recorded[dataset][arm][0] if dataset in recorded and arm in recorded[dataset] else fresh[(arm, dataset)]
        for dataset in datasets
        for arm in arm_names
    }

def _write_phase_snapshot(dpath_phase: Path, cfg_snapshot: dict) -> None:
    """Give a later phase its own copy of the campaign's frozen config snapshot (the screening phase's
    cfg_baseline.json), so every phase tree is self-contained for the regen tools. Write-once."""
    dpath_phase.mkdir(parents=True, exist_ok=True)
    fpath = dpath_phase / "cfg_baseline.json"
    if not fpath.exists():
        save_json(cfg_snapshot, fpath)

def _copy_qual_picks(campaign: str, picks: dict[tuple[str, str], str], cfg_snapshot: dict) -> None:
    """Seed the qual tree from the screening tree: the campaign's frozen config snapshot (_write_phase_snapshot),
    then for each pick its coord dir wholesale -- every screening seed's trial, config/overrides/coord_metadata
    and coord_stats -- so the qual tree reads as if the coord had run there from the start; the qual phase then
    tops it up to n_trials_qual seeds. Copy-once: an existing qual coord dir (a relaunch) is left as is."""
    dpath_qual = _dpath_phase(campaign, "qual")
    _write_phase_snapshot(dpath_qual, cfg_snapshot)
    for (dataset, arm), coord in picks.items():
        rel = Path("datasets") / dataset / "arms" / arm / "coords" / coord
        if not (dpath_qual / rel).exists():
            shutil.copytree(_dpath_phase(campaign, "screening") / rel, dpath_qual / rel)

def _trainval_stops(campaign: str, matrix: dict) -> dict[tuple[str, str, str], int]:
    """{(dataset, arm, coord): checkpoint index} over the qual phase's matrix: the checkpoint each pick is trained
    up to in the trainval phase -- its qual-selected one, the argmax of the pick's across-trial mean Native mAP
    composite All curve over its qual trials (report.update_chkpt_selection's best_chkpt.map.native, read from the
    qual coord's coord_metadata.json; final once every qual trial of the pick is in)."""
    return {
        (dataset, arm, coord): load_json(
            _dpath_phase(campaign, "qual") / "datasets" / dataset / "arms" / arm / "coords" / coord / "coord_metadata.json"
        )["best_chkpt"]["map"]["native"]["idx"]
        for dataset, arms in matrix.items()
        for arm, coords in arms.items()
        for coord in coords
    }

def run_campaign(campaign: str, n_trials_screen: int, n_trials_qual: int | None, trainval: bool, datasets: list[str],
                 ablation_arms: list[list[dict]], hpo_coords: list[list[dict]]) -> bool:
    """Run the campaign: the screening phase -- every arm x coord on every dataset for n_trials_screen seeds,
    under artifacts/<campaign>/screening/ -- then, unless n_trials_qual is null, the qual phase under
    artifacts/<campaign>/qual/: each arm's best screening coord per dataset (_qual_picks), its screening trials
    copied over (_copy_qual_picks) and topped up to n_trials_qual seeds, so the qual tree reads as if
    n_trials_qual trials had run for the pick -- then, with `trainval`, the trainval phase under
    artifacts/<campaign>/trainval/: every pick again over the qual seeds, each run on the trainval partition and
    stopped at the pick's qual-selected checkpoint (_trainval_stops) with its weights saved there (train.py); no
    evals. A phase whose trials don't all complete (a trial failed for good) ends the campaign there: the next
    phase is not started until a relaunch has resumed the failed trials. Returns False when the run was
    interrupted (Ctrl-C / SIGTERM) -- the campaign queue stops on it -- True otherwise (the campaign is over,
    complete or not)."""
    # Validate the planned matrix before any side effects: every arm / coord name must be unique, and
    # no override key may be claimed by both an arm and a coord.
    arms, coords = _expand_matrix(ablation_arms, hpo_coords)
    arm_names = [name for name, _ in arms]

    _enable_child_subreaper()
    # Route SIGTERM (e.g. `kill`, SLURM scancel) through the same path as Ctrl-C
    # so the trial's subtree is torn down before the campaign exits.
    signal.signal(signal.SIGTERM, _raise_interrupt)

    # campaign-level fires once, when the campaign is first created -- a relaunch (resume/extension)
    # is not a new beginning, so the cache the campaign's own trials built survives it
    first_launch = not (_dpath_phase(campaign, "screening") / "campaign_metadata.json").exists()
    cfg_snapshot = _load_or_create_campaign_config(campaign)
    if first_launch and cfg_snapshot["train"]["dev"]["del_base_eval_cache"]["campaign"]:
        _del_base_eval_cache()

    matrix = {dataset: {arm: [name for name, _ in coords] for arm in arm_names} for dataset in datasets}
    outcome = _run_phase(campaign, "screening", cfg_snapshot, arms, coords, datasets, _iter_seeds(n_trials_screen), matrix)
    if outcome != "complete" or n_trials_qual is None:
        return outcome != "interrupted"

    picks = _qual_picks(campaign, datasets, arm_names)
    _copy_qual_picks(campaign, picks, cfg_snapshot)
    picked = set(picks.values())
    coords_qual = [(name, payload) for name, payload in coords if name in picked]
    matrix = {dataset: {arm: [picks[(dataset, arm)]] for arm in arm_names} for dataset in datasets}
    seeds_qual = _iter_seeds(n_trials_qual)
    outcome = _run_phase(campaign, "qual", cfg_snapshot, arms, coords_qual, datasets, seeds_qual, matrix)
    if outcome != "complete" or not trainval:
        return outcome != "interrupted"

    # trainval: the qual matrix once more, on the trainval partition -- one run per qual seed, each stopped at the
    # pick's qual-selected checkpoint. The LR schedule keeps its full n_epochs horizon (warmup a fraction of it,
    # cosine over all of it), so checkpoint k sees the LR it saw in train; only the epochs are longer.
    _write_phase_snapshot(_dpath_phase(campaign, "trainval"), cfg_snapshot)
    outcome = _run_phase(campaign, "trainval", cfg_snapshot, arms, coords_qual, datasets, seeds_qual, matrix,
                         chkpt_stops=_trainval_stops(campaign, matrix))
    return outcome != "interrupted"


def _load_campaign_config(name: str) -> CampaignConfig:
    fpath = paths["config"] / "camps" / f"{name}.yaml"
    if not fpath.exists():
        avail = ", ".join(sorted(p.stem for p in (paths["config"] / "camps").glob("*.yaml")))
        raise SystemExit(f"Campaign config not found: {fpath}\nAvailable campaigns: {avail}")
    with open(fpath) as f:
        return CampaignConfig(**yaml.safe_load(f))

def _dedupe_campaign_name(campaign: str) -> str:
    """Return the first campaign name without an existing artifacts dir: `campaign` itself, else
    `<campaign>2`, `<campaign>3`, ..."""
    if not _dpath_campaign(campaign).exists():
        return campaign
    n = 2
    while _dpath_campaign(f"{campaign}{n}").exists():
        n += 1
    return f"{campaign}{n}"

def _launch_camp(name: str) -> bool:
    """Run the campaign defined by config/camps/<name>.yaml: resolve the campaign name (suffix +
    dev.continue_campaign dedupe) and hand the expanded matrix to run_campaign; returns its
    completed flag."""
    cfg = _load_campaign_config(name)
    campaign = f"{name}_{cfg.suffix}" if cfg.suffix is not None else name
    if not load_train_config_dict()["dev"]["continue_campaign"]:
        deduped = _dedupe_campaign_name(campaign)
        if deduped != campaign:
            print(f"campaign '{campaign}' already exists -- starting '{deduped}' (dev.continue_campaign: false)", flush=True)
            campaign = deduped
    return run_campaign(
        campaign=campaign,
        n_trials_screen=cfg.n_trials_screen,
        n_trials_qual=cfg.n_trials_qual,
        trainval=cfg.trainval,
        datasets=cfg.datasets,
        ablation_arms=cfg.ablation_arms,
        hpo_coords=cfg.hpo_coords,
    )

def _load_queue() -> list[str]:
    """The `campaigns` list from config/camp_queue.yaml (a blank list parses to None -> [])."""
    with open(paths["config"] / "camp_queue.yaml") as f:
        return yaml.safe_load(f)["campaigns"] or []

def _validate_queue_entry(spec: str) -> None:
    """Shallow fail-fast check of one queue entry: the camp. prefix and a loadable, valid config yaml (CampaignConfig)."""
    kind, _, name = spec.partition(".")
    if kind != "camp":
        raise SystemExit(f"Invalid camp_queue.yaml entry '{spec}': entries take the form camp.<name>.")
    _load_campaign_config(name)

def _next_queue_entry(executed: list[str]) -> str | None:
    """Re-read camp_queue.yaml and return the first entry not yet run this session, or None when
    the queue is drained. Each executed entry consumes one matching occurrence from the list, so
    entries may be added at any position while a campaign runs (and a duplicated name queues a
    second run). Every pending entry is validated on each call: the one about to run hard-fails,
    later ones only warn -- the file is re-read anyway, so a bad late addition can be fixed in
    place before it is reached."""
    pending = _load_queue()
    for spec in executed:
        if spec in pending:
            pending.remove(spec)  # first occurrence
    for spec in pending[1:]:
        try:
            _validate_queue_entry(spec)
        except (SystemExit, yaml.YAMLError, TypeError, ValueError) as e:
            print(f"camp_queue: pending entry '{spec}' is invalid -- fix before it is reached: {e}", flush=True)
    if not pending:
        return None
    _validate_queue_entry(pending[0])
    return pending[0]

def _run_queue_entry(spec: str) -> bool:
    """Dispatch one validated queue entry; returns run_campaign's completed flag."""
    _, _, name = spec.partition(".")
    return _launch_camp(name)

def main() -> None:
    if sys.argv[1:]:
        raise SystemExit("Usage: python -m campaign_runner (no arguments; campaigns are queued in config/camp_queue.yaml)")
    executed: list[str] = []
    while (spec := _next_queue_entry(executed)) is not None:
        print(f"camp_queue: launching '{spec}'", flush=True)
        try:
            completed = _run_queue_entry(spec)
        except KeyboardInterrupt:
            completed = False
        if not completed:
            print(f"camp_queue: '{spec}' interrupted -- exiting queue", flush=True)
            return
        executed.append(spec)
    print(f"camp_queue: drained -- {len(executed)} campaign(s) run", flush=True)


if __name__ == "__main__":
    main()
