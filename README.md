The BioCosmos project aims to develop a multimodal, AI-powered interface for biologists to search a species database via image or text. At it's core is a vision-language model (VLM) trained to capture fine-grained cross-modal relationships: jointly learned image and text encoders bridge the gap between visual data and natural language.

This element of the BioCosmos project focuses on developing fine-tuning methods to obtain robust VLM performance on long-tailed, hierarchically structured data in support of multimodal biodiversity retrieval tasks. Our goal is to systematically experiment with methods best suited to this regime and to deliver the strongest fine-tuned VLMs possible for downstream use in the BioCosmos app.

# Setup

## System Requirements & Assumptions

Hardware: This setup is optimized for NVIDIA B200 GPUs on HiPerGator.
Runtime: CUDA is mandatory; CPU-only execution is not supported.
Distributed Training: The codebase assumes Distributed Data Parallel (DDP) execution via `torchrun`. Non-distributed runs are not supported, although the DDP code still supports single-GPU runs.

Setup is intended for use with B200s on HiPerGator.

## Codebase

Pull repo and navigate:
```
git clone https://github.com/odobon1/biocosmos.git
cd biocosmos
```

## Environment

Two environments are provided; they differ only in their PyTorch/CUDA build, which determines the GPU architectures each supports. Pick the one matching the GPU you're running on:

| Environment file | Env name | PyTorch / CUDA | Supported GPUs |
|---|---|---|---|
| `environment_b200.yaml` | `biocosmos_b200` | 2.7.1 / cu128 | A100, H100/H200, **B200** |
| `environment.yaml` | `biocosmos` | 2.5.1 / cu121 | **V100**, A100, H100/H200 |

B200 requires `biocosmos_b200` (only it carries Blackwell `sm_100` kernels); V100 requires `biocosmos` (only it carries Volta `sm_70` kernels). A100 and H100/H200 run on either.

Create and activate the one you need, e.g. for B200:
```
conda env create -f environment_b200.yaml
conda activate biocosmos_b200
```

## Preprocessing

All metadata artifacts (including splits) and data indexing structures needed for train and eval are committed to the repo, so preprocessing does not need to be run for normal use. See [preprocessing/README.md](preprocessing/README.md) for details on the pipeline and how to regenerate these artifacts.

# Dataset Characteristics

| Name        | Alias   | Class Level | Available Taxonomic Ranks                | Class Imbalance    | Total Images | Filetype | Resolution                                                                                           | Imagery                                                                      |
|-------------|---------|-------------|------------------------------------------|--------------------|--------------|----------|--------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| Bryozoa     | `bryo`  | genus       | family, genus                            | moderate imbalance | 18,696       | `.jpg`   | Variable, high resolution<ul><li>short-side&nbsp;min:&nbsp;144</li><li>long-side&nbsp;max:&nbsp;5120</li><li>short-side&nbsp;median:&nbsp;1536</li><li>long-side&nbsp;median:&nbsp;2048</li></ul>  | Natural settings; **greyscale**; many images with scientific annotations |
| CUB         | `cub`   | species     | order, family, genus, species            | well-balanced      | 11,788       | `.jpg`   | Variable resolution<ul><li>short-side&nbsp;min:&nbsp;120</li><li>long-side&nbsp;max:&nbsp;500</li><li>short-side&nbsp;median:&nbsp;357</li><li>long-side&nbsp;median:&nbsp;500</li></ul>     | Natural settings                                                        |
| Lepidoptera | `lepid` | species     | family, subfamily, tribe, genus, species | extreme imbalance  | 619,248      | `.png`   | 512×512 (fixed)                                                                                        | Curated, museum-quality specimens; uniformly preprocessed **(cite Lepidoptera publication)** |
| Nymphalidae | `nymph` | species     | subfamily, genus, species                | extreme imbalance  | 230,689      | `.png`   | 336×336 (fixed)                                                                                        | Curated, museum-quality specimens; uniformly preprocessed             |

A note on CUB: standard protocol test partitions used (not stratified; appears to be randomly sampled); our custom stratification method used for generating ID + OOD validation partitions. Well-balanced → impossible to draw OOD classes satisfying typical OOD sample-volume tolerance thresholds.

# Testing

The repo includes a `pytest` suite under `tests/` for fast unit tests and lightweight integration checks. See [tests/README.md](tests/README.md) for common test commands and usage details.

# Train & Eval
Training and evaluation are config-driven: switch models, losses, LR schedules, batch size in YAML (no code edits). The main training config lives at `config/train.yaml` and the standalone eval config at `config/eval.yaml`.

Note: With `hardware.loss_chunk_size: null`, the full similarity matrix is computed for all model types, including SigLIP. Setting an integer `loss_chunk_size` enables a SigLIP-style implementation of the global-batch BCE-family loss (`bce`/`bif_bce`): the BxB rows are sharded across ranks (each rank computes only its B/world_size band, as in the chunked loss decomposition from the SigLIP paper) and swept in C-row tiles, with negatives taken from gathered embeddings rather than the paper's ring permute. See `config/hardware.yaml` for details.

## Train
1. Edit `config/train.yaml`:
    * `model_type`, `loss_type`, `targ_type`, `lr_sched_type`, etc.
    * Mixed precision & activation checkpointing can be toggled in `config/hardware.yaml`: `mixed_prec`, `act_chkpt` (autocast dtype is always bf16)
2. Training runs through campaigns — see [Run a campaign](#run-a-campaign) below. `config/train.yaml` is the base config (layer 1) every campaign trial starts from. For a single one-off run, use a minimal campaign (e.g. `config/camps/dev.yaml`).

## Evaluate a trained model
**Note:** trials no longer save model weights, so campaigns produce no checkpoint directory to point this at — `rdpath_model` needs an externally supplied one laid out as `<seed>/chkpts/<name>/model.pt` (the loader reads the trial's `trial_metadata.json` and the setting's `config.json` from that dir's ancestors).
1. In `config/eval.yaml`, set `rdpath_model` to a checkpointed model directory (e.g. `artifacts/dev/settings/sp/lepid/42/chkpts/final`).
2. Run:
    ```
    torchrun --standalone --nproc-per-node=auto -m eval
    ```
    When `rdpath_model` is set, eval overrides `dataset`, `split`, `model_type` from setting + trial saved metadata.

    Note: n-shot performance is reported for the ID partition only; the bucket set follows `eval_type` — `val` → `train/val` buckets, `test` → `trainval/test` buckets.

## Evaluate a base model
1. In `config/eval.yaml`, set `rdpath_model: null`.
2. Run:
    ```
    torchrun --standalone --nproc-per-node=auto -m eval
    ```

## Run a campaign
1. Define the campaign in `config/camps/<campaign>.yaml`:
    * `n_trials` — number of random seeds per setting/dataset combo
    * `datasets` — the datasets to train on, a list of dataset names (e.g. `bryo`, `cub`, `lepid`, `nymph`); every setting is trained on each one
    * `baseline_overrides` — the per-setting config overrides, given as a list of **combo groups**. Each combo group is a list of partial settings (a set of overrides plus an optional `name`; when `name` is omitted, it is derived from the overrides as `key-value` pairs joined by `_`, with keys and values mapped through the alias tables `CFG_PARAM_ALIASES` / `CFG_PARAM_VALUE_ALIASES` in `utils/config.py` — value aliases are per original key — and anything without an alias passing through verbatim, e.g. `{batch_size: 2_048}` → `bs-2k`, `{loss2.mix: 0.3, loss2.targ: phylo}` → `loss2.mix-0.3_L2T-hp`). An override value given as a **list** is a **combo list**: the item expands into one partial setting per combination of its list values (several combo lists in one item cross with each other, the last-listed varying fastest), and the chosen `key-value` pairs are always reflected in the setting name — appended to the item's `name` when one is given (e.g. `{batch_size: [1_024, 2_048], name: hp}` → `hp_bs-1k`, `hp_bs-2k`), or folded into the derived name as usual when not. The campaign's settings are the **Cartesian product** across combo groups: one partial setting is taken from each combo group and merged into a single setting, named by joining the members' names with `_`. With a single combo group, its members become the settings directly.
    * `baseline` — `true`/`false`; when `true`, the campaign additionally runs a **`baseline`** setting ahead of the `baseline_overrides` settings: the frozen `train.yaml` snapshot as-is, with no overrides applied. The setting name `baseline` is reserved — a `baseline_overrides` item may not use it.
    * `suffix` — appended to the campaign name (`null` for none)

   For example, crossing a loss combo group with a batch-size combo group:
    ```yaml
    baseline_overrides:
      - - {loss2.mix: 0.3, loss2.targ: phylo, name: hp}
        - {loss.targ: mp, name: mp}
      - - {batch_size: 2_048, name: 2k}
        - {batch_size: 1_024, name: 1k}
    ```
   produces four settings — `hp_2k`, `hp_1k`, `sw_2k`, `sw_1k` — each merging one partial setting from every combo group. It is equivalent to spelling out the product as a single combo group:
    ```yaml
    baseline_overrides:
      - - {loss2.mix: 0.3, loss2.targ: phylo, batch_size: 2_048, name: hp_2k}
        - {loss2.mix: 0.3, loss2.targ: phylo, batch_size: 1_024, name: hp_1k}
        - {loss.targ: mp, batch_size: 2_048, name: sw_2k}
        - {loss.targ: mp, batch_size: 1_024, name: sw_1k}
    ```
   A single combo group expands to its members unchanged:
    ```yaml
    baseline_overrides:
      - - {loss.targ: sp, name: sp}
        - {loss.targ: mp, name: mp}
    ```
   produces `sp` and `mp`.

   Each resulting **setting** — a member of this Cartesian product, not a single `baseline_overrides` item (they coincide only for a single combo group) — is trained for `n_trials` seeds on every dataset, so the campaign runs *settings × datasets × `n_trials`* trials in total.

   The campaign is named `<campaign>_<suffix>` (or just `<campaign>` when `suffix` is `null`). By default (`dev.continue_campaign: true` in `config/train.yaml`) a relaunch under an existing name resumes/extends that campaign; with `dev.continue_campaign: false`, a name whose `artifacts/<name>` dir already exists falls back to the first free `<name>2`, `<name>3`, … and starts a fresh campaign there instead.

   Putting it together, a complete single-combo-group campaign (`config/camps/foobar.yaml`):
    ```yaml
    n_trials: 5
    datasets: [cub, lepid]

    baseline: false
    baseline_overrides:
      - - {loss.targ: sp,  name: sp}
        - {loss.targ: mp, name: mp}
        - {loss.targ: phylo,    name: hp}

    suffix: null
    ```
   Queued as `camp.foobar` and launched via `python -m campaign_runner`, this runs **30 trials**: 3 settings (`sp`, `mp`, `hp`) × 2 datasets (`cub`, `lepid`) × 5 seeds (`42`–`46` — trial seeds are `SEED0 .. SEED0 + n_trials - 1`, and `SEED0 = 42`). Each trial's artifacts land under `artifacts/foobar/settings/<setting>/<dataset>/<seed>/`.
2. Queue the campaign in `config/camp_queue.yaml` and launch the runner (it takes no arguments):
    ```yaml
    campaigns:
      - camp.dev_basic
    ```
    ```
    python -m campaign_runner
    ```
   The runner works through `campaigns` in order — `camp.<name>` runs `config/camps/<name>.yaml`, `qual.<name>` runs `config/quals/<name>.yaml` (see qualified campaigns below). After each campaign finishes, the queue file is **re-read**, so more campaigns can be queued up while one runs — appended to the bottom or inserted at any position — and the runner exits once every listed entry has been run. Each entry is one run: listing the same name twice queues a second run (which resumes/extends or dedupes per `dev.continue_campaign`). At launch and at every re-read, all still-pending entries are shallow-validated (known `camp.`/`qual.` prefix, config file loads); the entry about to run fails hard, while later entries only print a warning — fix them in place before they're reached, since the file is re-read anyway. Ctrl-C (or SIGTERM) stops the whole queue, not just the running campaign.
3. Each trial is launched in a fresh subprocess (`campaign_trial_runner`) to isolate DDP/DataLoader worker state between trials.
4. If a trial fails, campaign execution continues and the error is written to that trial's `error.log` (`artifacts/<campaign>/settings/<setting>/<dataset>/<seed>/error.log`).
5. `artifacts/<campaign>/manifest.log` tracks trial progress, bucketing every planned trial (by `setting/dataset/seed`) into Failed / Completed / In Progress / Queued. Completed and Failed entries also show the trial's recorded wall-clock, and Failed entries additionally show epoch progress and the failure type (`<setting>/<dataset>/<seed> --- D-HH:MM:SS --- E/N --- RAM|VRAM|Other|Mixed`, epoch index / `n_epochs`; `n/a` for a trial that failed before ever writing metadata; the failure type aggregates the fatal retry loop's crashes -- Mixed when they span categories). It is regenerated at kickoff and at each trial's start and finish.

**Note:** When resuming a campaign, the environment must allocate the same number of GPUs as the original run. The GPU count is saved to `artifacts/<campaign>/campaign_metadata.json` on first launch; a mismatch on resume raises an error before any trials execute.

**Note:** A campaign's config is **frozen at first launch**, bundled into a single snapshot. `config/train.yaml` (with `debug_mode` overrides folded in) and its three sibling config files are snapshotted together to `artifacts/<campaign>/cfg_baseline.json`, under the keys `train`, `hardware`, `manif_viz`, `model_specific` (`config/hardware.yaml` → `hardware`, `config/model_specific.yaml` → `model_specific`, `config/manif_viz.yaml` → `manif_viz`). Every trial starts from the `train` snapshot, has the sibling snapshots injected (as `hw`, `model_specific`, `manif_viz`), and layers its setting's `baseline_overrides` on top. Model-family `opt` defaults (`opt.wd`/`opt.beta2`) are left `null` in the baseline and resolved per trial from the cached `model_specific` snapshot, so a per-setting `arch.model_type` override still picks up the matching family's defaults. On any relaunch — resuming, or extending the matrix with added settings/datasets/seeds — the snapshot is read back from disk rather than re-read from the YAML, so edits to any of these config files after a campaign's first launch don't affect it: every trial (original or added later) uses the same frozen config. The one part still computed live per trial is the dataloader/GPU scaling (`n_workers`/`n_gpus`/`n_cpus`/`ram`), derived from the SLURM allocation so a resume adapts to the node; the static `hw` knobs (`mixed_prec`, `act_chkpt`, `loss_chunk_size`, `prefetch_factor`, `max_n_workers_gpu`, `persistent_workers`, `use_img_cache`, `eval`) are frozen and overridable per setting via `hw.*` in `baseline_overrides`. `config/stats.yaml` (stats-table/metrics-workbook rendering settings) is deliberately **not** part of the snapshot: it is read live at each stats render (trial completion, and `python -m tools.regen_stats <campaign>`), so edits to it apply to the next re-render of any campaign, frozen or not.

**Note:** Each setting's declared overrides are written to `artifacts/<campaign>/settings/<setting>/overrides.json` when the setting's first trial launches — a setting's directory is not created until a trial of it actually starts, so a planned-but-never-run setting leaves no `settings/` entry. This records the overrides **as declared** in `baseline_overrides` — verbatim, not a diff against the baseline — so a key appears even when its value equals the baseline's (e.g. `loss.targ: sp` is listed even if `config/train.yaml` already sets it).

**Note:** Combo groups are independent dimensions, so the same override key may not appear in more than one combo group — a shared key would have two values fighting to define it when settings merge, and raises an error at kickoff.

**Note:** Every resulting setting `name` must be unique (for crossed combo groups, the joined name, e.g. `hp_2k`); a duplicate raises an error at kickoff.

**Note:** A campaign's matrix is **additive across runs**. After a campaign has run, you may **add** settings (new members within an existing `baseline_overrides` combo group), `datasets`, or seeds (by raising `n_trials`) and relaunch to extend it — already-completed trials are skipped and only the new ones run. (Adding a whole new *combo group* re-joins every setting name, e.g. `hp` → `hp_2k`, so it reads as removing all prior settings — start a new campaign for that.) You may **never remove** a setting, dataset, or seed that a prior run recorded: the planned settings/datasets/seeds are saved to `artifacts/<campaign>/campaign_metadata.json` at each launch, and a relaunch whose config drops any previously-recorded item raises an error before any trials execute (removing one would orphan its already-computed trials). To drop items, start a new campaign instead.

## Run a qualified campaign

A **qualified campaign** tops up a subset of a completed campaign's settings — the ones that "qualified" — to a higher trial count, without re-running what the base campaign already computed. Define it in `config/quals/<name>.yaml`:

* `n_trials_qual` — **total** trials per qualified setting, base trials included (e.g. the base ran 1 seed and `n_trials_qual: 3` → 2 new trials per setting). Must be ≥ the base campaign's trial count.
* `base_campaign` — the campaign to qualify from, by its `artifacts/` name. Every trial in its recorded matrix must be **complete**, else launch errors.
* `qualified_settings` — the base settings to carry forward, by setting name (each must exist in the base campaign).

There is no `datasets` field — the base campaign's datasets are used. Example (`config/quals/dev.yaml`):
```yaml
n_trials_qual: 3

base_campaign: dev43

qualified_settings:
  - phylo2
  - sp
```
Launch by queueing it in `config/camp_queue.yaml` (as `qual.<name>`, e.g. `qual.dev`) and running `python -m campaign_runner` — qual entries go through the same campaign queue as regular ones. A qual may be queued behind the campaign that produces its base (e.g. `camp.dev` then `qual.dev`): its base-campaign checks only run when its turn comes.

This creates campaign **`<base_campaign>_qual`** (e.g. `dev43_qual`), seeded from the base campaign: `cfg_baseline.json` is copied over — qual trials train against the **base campaign's frozen config**, not the current yamls — and each qualified setting's whole `settings/<setting>/` directory (trials, metadata, per-dataset stats) is copied as if the qualified campaign had run those trials itself. Per-setting overrides come from the base campaign's persisted `settings/<setting>/overrides.json`, not from any camps yaml. The campaign then runs like any other over the matrix *`qualified_settings` × base datasets × `n_trials_qual` seeds*: the copied trials are already complete and are skipped, so only the seeds above the base campaign's run (base ran seed `42`, `n_trials_qual: 3` → seeds `43`, `44`).

The usual campaign rules apply unchanged: `dev.continue_campaign` resume/dedupe semantics (a relaunch under `dev.continue_campaign: false` falls back to `<base_campaign>_qual2`, …), the additive-matrix rule (relaunch with more `qualified_settings` — newly-qualified settings are copied in — or a larger `n_trials_qual` to extend; never remove), and the GPU-count match on resume — additionally checked against the **base** campaign at launch, since the copied trials ran under its world size. The `manif_viz` seed window (`n_seeds`/`n_seeds_offset`, frozen from the base campaign's snapshot) gates which trials compute manifold viz by `idx_seed` over the qual campaign's seed sweep as usual — with the base's `n_seeds: 1, n_seeds_offset: 0`, the copied first seed already carries its viz and the added seeds compute none.

## Config Override Layers

Training config is assembled from multiple sources. Layers are listed in increasing priority order — each layer overwrites anything set by earlier layers.

| Priority | Source | Applied by | Description |
|----------|--------|-----------|-------------|
| 1 (lowest) | `config/train.yaml` | `load_train_config_dict()` | Base config; the starting point for all training runs. |
| 2 | `config/hardware.yaml` | `load_hardware_config_dict()` (→ `hw`) | Static hardware knobs (`mixed_prec`, `act_chkpt`, `loss_chunk_size`, `prefetch_factor`, `max_n_workers_gpu`, `persistent_workers`, `use_img_cache`, `eval`) under the `hw` key. `use_img_cache: true` reads images from the prebuilt per-dataset pack (`tools/build_img_cache.py`), staged once per node to node-local scratch (`SLURM_TMPDIR`/`TMPDIR`/`/tmp`) instead of per-sample files on the shared FS; the campaign runner stages up front (error at startup if a pack is missing — also when caching is enabled only via a per-setting `hw.use_img_cache` override) and records per-dataset staging seconds to `campaign_metadata.json` under `runtime_img_cache` (`null` = dataset unused or caching off). Staged copies live at `<SLURM_TMPDIR\|TMPDIR\|/tmp>/img_cache-<user>/<dataset>/`; job-scoped scratch is purged by the scheduler, while the `/tmp` fallback persists (and is revalidated/re-staged automatically if a tmp reaper evicts files). Standalone entrypoints (`eval.py`, tools) stage on first dataloader construction. Cached to `cfg_baseline.json` (under the `hardware` key) at first launch and injected per trial; the live `n_workers`/`n_gpus`/`n_cpus`/`ram` scaling is computed separately from the SLURM allocation. |
| 3 | Campaign runner injections | `run_campaign()` | Injects `campaign`, `setting`, `seed`, `dataset` from the campaign matrix (these per-trial keys exist only here, not in `train.yaml`). |
| 4 | `config/model_specific.yaml` | `apply_model_specific_opt_defaults()` | Fills `opt.wd` and `opt.beta2` **only if `null`**, based on model family (`clip` or `siglip`). Has no effect if those fields are already set in `config/train.yaml`. |
| 5 | `debug_mode` overrides | `apply_train_debug_overrides()` | If `dev.debug_mode: true`, forces `split → "dev"` + overrides specified in `dev.debug`. |
| 6 (highest) | `baseline_overrides` (campaign) | `get_config_train()` via `_setting_overrides` | Per-setting overrides defined in the campaign config (`config/camps/<campaign>.yaml`); dot-paths reach any field, including `hw.*`. A dot-path must address a field that already exists in the config — overrides replace declared fields, never create them, so a typo'd or renamed key raises at campaign startup instead of silently landing in a field nothing reads. |

<br>

# Experimental Procedure

Experiments proceed in two stages. A fixed number of epochs is used across all trials (no early stopping), so results stay comparable across runs.

## Stage 1 — Selection

Model selection: hyperparameter tuning, preliminary ablations, etc.
* Train on the `train` partition.
* Evaluate on the in-distribution (ID) and out-of-distribution (OOD) **validation** partitions.

## Stage 2 — Final Testing

Final performance measurement, performed ideally only once. Each additional look at test performance risks leaking test-set information into subsequent decisions (adaptive overfitting), inflating the reported numbers relative to true generalization; keeping the test partitions untouched until the end preserves them as an unbiased estimate.
* With hyperparameters and design choices fixed from Stage 1, train on the `trainval` partition.
* Evaluate on the ID and OOD **test** partitions.
* No train-time evaluations are performed while training on `trainval`; final performance is collected via standalone evaluation.

<br>

# Supported Architectures

The **Max Batch Size** column indicates max batch size whilst training on a single B200 using mixed precision and activation checkpointing.

## CLIP - ViT Variants

| Model ID <br> (Internal) | Model ID <br> (open_clip)     | Max Batch Size | Pretrain  | Num Params | Embedding <br> Dimension | (V) Layers | (V) Width | (V) Heads | (T) Layers | (T) Width | (T) Heads | **(\*1)** Learning Rate | Resolution (px) |
|--------------------------|-------------------------------|----------------|-----------|------------|--------------------------|------------|-----------|-----------|------------|-----------|-----------|-------------------------|-----------------|
| `clip_vitb32`            | `ViT-B-32`                    | 16,384         | **(\*2)** | 151M       | 512                      | 12         | 768       | 12        | 12         | 512       | 8         | 5e-4                    | 224             |
| `clip_vitb16`            | `ViT-B-16`                    | 8,192          | **(\*3)** | 150M       | 512                      | 12         | 768       | 12        | 12         | 512       | 8         | 5e-4                    | 224             |
| `clip_vitl14`            | `ViT-L-14`                    | 4,096          | **(\*4)** | 428M       | 768                      | 24         | 1,024     | 16        | 12         | 768       | 12        | 4e-4                    | 224             |
| `clip_vitl14_336`        | `ViT-L-14-336`                | 2,048          | `openai`  | 428M       | 768                      | 24         | 1,024     | 16        | 12         | 768       | 12        | 2e-5 (investigate)      | 336             |
| `bioclip`                | `hf-hub:imageomics/bioclip`   | 8,192          | NA        | -          | -                        | -          | -         | -         | -          | -         | -         |                         | -               |
| `bioclip2`               | `hf-hub:imageomics/bioclip-2` | 4,096          | NA        | -          | -                        | -          | -         | -         | -          | -         | -         |                         | -               |

**(V)** denotes vision transformer, **(T)** denotes text transformer. Model specs not included for `bioclip` and `bioclip2` because they are redundant. `bioclip` is a CLIP ViT-B/16 fine-tuned on the TOL-10M dataset, `bioclip2` is a CLIP ViT-L/14 fine-tuned on the TOL-200M dataset.

**(\*1)** Peak LR from seminal CLIP arXiv Table 20.

**(\*2)** `ViT-B-32` Pretrain-Dataset Model Weights Available:
`openai`, `laion400m_e31`, `laion400m_e32`, `laion2b_e16`, `laion2b_s34b_b79k`, `datacomp_xl_s13b_b90k`, `datacomp_m_s128m_b4k`, `commonpool_m_clip_s128m_b4k`, `commonpool_m_laion_s128m_b4k`, `commonpool_m_image_s128m_b4k`, `commonpool_m_text_s128m_b4k`, `commonpool_m_basic_s128m_b4k`, `commonpool_m_s128m_b4k`, `datacomp_s_s13m_b4k`, `commonpool_s_clip_s13m_b4k`, `commonpool_s_laion_s13m_b4k`, `commonpool_s_image_s13m_b4k`, `commonpool_s_text_s13m_b4k`, `commonpool_s_basic_s13m_b4k`, `commonpool_s_s13m_b4k`, `metaclip_400m`, `metaclip_fullcc`

**(\*3)** `ViT-B-16` Pretrain-Dataset Model Weights Available: `openai`, `laion400m_e31`, `laion400m_e32`, `laion2b_s34b_b88k`, `datacomp_xl_s13b_b90k`, `datacomp_l_s1b_b8k`, `commonpool_l_clip_s1b_b8k`, `commonpool_l_laion_s1b_b8k`, `commonpool_l_image_s1b_b8k`, `commonpool_l_text_s1b_b8k`, `commonpool_l_basic_s1b_b8k`, `commonpool_l_s1b_b8k`, `dfn2b`, `metaclip_400m`, `metaclip_fullcc`

**(\*4)** `ViT-L-14` Pretrain-Dataset Model Weights Available: `openai`, `laion400m_e31`, `laion400m_e32`, `laion2b_s32b_b82k`, `datacomp_xl_s13b_b90k`, `commonpool_xl_clip_s13b_b90k`, `commonpool_xl_laion_s13b_b90k`, `commonpool_xl_s13b_b90k`, `metaclip_400m`, `metaclip_fullcc`, `dfn2b`, `dfn2b_s39b`

As we can see the CLIP ViT-series have received quite a lot of attention (no pun attended).

Note: despite the name, both the open-source OpenCLIP and the OpenAI CLIP model weights are available through `open_clip`. Pretrain dataset = "openai" --> **OpenAI CLIP** (a.k.a. "original CLIP"), trained using a private recipe on a private pretraining dataset. Pretrain dataset = <anything other than "openai"> --> **OpenCLIP** ~ community replications of CLIP trained on open sourced datasets as specified in the Pretrain column.

## SigLIP Variants

| Model ID <br> (Internal) | Model ID <br> (open_clip) | Max Batch Size | Pretrain | Num Params | Embedding <br> Dimension | Resolution (px) |
|--------------------------|---------------------------|----------------|----------|------------|--------------------------|-----------------|
| `siglip_vitb16`          | `ViT-B-16-SigLIP`         | 8,192          | `webli`  | 203M       | 768                      | 224             |
| `siglip_vitb16_256`      | `ViT-B-16-SigLIP-256`     | X              | `webli`  | 203M       | 768                      | 256             |
| `siglip_vitb16_384`      | `ViT-B-16-SigLIP-384`     | 2,048          | `webli`  | 203M       | 768                      | 384             |
| `siglip_vitl16_256`      | `ViT-L-16-SigLIP-256`     | X              | `webli`  | 653M       | 1,024                    | 256             |
| `siglip_vitl16_384`      | `ViT-L-16-SigLIP-384`     | 2,048          | `webli`  | 653M       | 1,024                    | 384             |
| `siglip_vitso400m14`     | `ViT-SO400M-14-SigLIP`    | 2,048          | `webli`  | 877M       | 1,152                    | 224             |
| `siglip2_vitb16`         | `ViT-B-16-SigLIP2`        | 8,192          | `webli`  | 375M       | 768                      | 224             |
| `siglip2_vitb16_384`     | `ViT-B-16-SigLIP2-384`    | 2,048          | `webli`  | 375M       | 768                      | 384             |
| `siglip2_vitl16_384`     | `ViT-L-16-SigLIP2-384`    | 2,048          | `webli`  | 822M       | 1,024                    | 384             |
| `siglip2_vitso400m14`    | `ViT-SO400M-14-SigLIP2`   | 2,048          | `webli`  | 1,136M     | 1,152                    | 224             |
| `siglip2_vitgopt16_384`  | `ViT-gopt-16-SigLIP2-384` | 512            | `webli`  | 1,870M     | 1,536                    | 384             |

# Notes

## Batching
During training, partial batches are dropped by default. For more granular batching methods e.g. dorsal/ventral, partial batches are dropped
from each category, which may result in fewer batches per epoch. Dorsal/ventral batching can only be toggled for train.
For eval, loss is computed for full batches only, although performance computation includes partial batches.

For train-time eval loss, the gathered eval embeddings are deterministically shuffled (a fixed-seed permutation, identical on every rank) before being sliced into chunks of size `eval_batch_size × world_size`. The shuffle mixes the rank-ordered, class-clustered gather output so each chunk presents a varied set of in-batch negatives, and the fixed seed keeps the chunking reproducible across runs — making the eval batch loss an apples-to-apples comparison with the global train batch loss. The trailing partial chunk is dropped.

**Chain-shuffled passes for small train sets** — `chain_floor` in `config/train.yaml` (`null` disables): when the train set is smaller than `chain_floor`, batches are cut from one continuous stream of independently shuffled full permutations of the train set, and each dataloader pass consumes the stream's next batch-aligned window of `ceil(chain_floor / train_set_size)` permutations' worth of samples instead of a single permutation. Every aligned block of `train_set_size` consecutive stream samples covers the dataset exactly once, so coverage stays balanced while pass length — and the maximum usable `batch_size` — is decoupled from dataset size. This avoids the per-pass dataloader-restart stall that dominates wall-clock when a tiny dataset runs thousands of one-pass epochs. Because the stream runs across passes, the permutation tail that doesn't fill a pass's last batch isn't dropped — it leads the next pass's window (when `batch_size > train_set_size`, whole permutations' worth can wrap). Epoch accounting credits each pass with `ceil(consumed_samples / train_set_size)` permutations' worth of samples.

## Base Model Performance Cache
The base-model evaluation at the start of each trial (the model's performance before any training steps) is cached under `base_eval_cache/` — a flat directory holding one pickle per combo, named by the serialized combo key (`<model_type>__<dataset>__<split>__<non_causal>__<text_template.eval>__<vis_proj_head>__<seed>.pkl`). The key (`ArtifactManager.base_eval_key`) covers the config settings that determine the base model's eval output (numerics-level knobs — `hw` `mixed_prec`, t-SNE perplexity — are deliberately not keyed). The base eval evaluates the model **as configured for the trial** — a SigLIP `arch.siglip.vis_proj_head` head and CLIP `arch.clip.non_causal` both apply — so configs differing in any key component get separate entries. Family-inert components are normalized to `None` so equivalent configs share one entry: `non_causal` is CLIP-only, `vis_proj_head` is SigLIP-only, and `seed` only enters through the random init of a `linear`/`mlp` `vis_proj_head` head (`seed=None` otherwise, so all seeds of a headless setting share one base). On the first trial for a given combo the base eval runs and its entry is written to the combo's file: `metrics` (the formatted `scores` trees for every eval group — what the per-group metrics files carry, minus the `loss_raw`, `sim`, `targ`, and `chkpt` fields), `projections` (the raw t-SNE/PCA arrays; viz trials only, else `None`), and `embs` (the raw eval embeddings; viz trials only, else `None`). UMAP is fit post-trial and so is never carried in the cache. The write is a temp-file + atomic replace of just that combo's file, so readers never see a torn file and concurrent campaigns only ever touch the same file when computing the same combo (in which case they overwrite each other with equivalent entries). Subsequent trials with the same combo reuse the entry instead of re-running the base eval, materializing it to `evals/base/`. A manifold-viz trial's cache hit additionally requires the entry's `projections`, and a pooled-viz trial's its `embs` — an entry lacking a piece the trial needs (e.g. metrics-only, written by a non-viz trial) reads as a miss: the base eval recomputes and overwrites the entry with the richer version, while trials that don't need the missing pieces can still reuse it. To force base evals to recompute, delete `base_eval_cache/` (or just the affected combo files). The `dev.del_base_eval_cache` flags (train.yaml) automate this: `campaign: true` deletes `base_eval_cache/` once, when a campaign is first created (a relaunch of an existing campaign is not a new beginning, so the cache its own trials built survives), and `trial: true` deletes it before every trial launch so each trial re-runs its base eval — useful for apples-to-apples wall-clock comparisons across trials.

The cached scores are reproducible across single-GPU and multi-GPU runs: performance metrics are computed on the full set of embeddings gathered from all ranks, so the complete evaluation set, and thus the resulting scores, are identical regardless of `world_size`. A cache written on one GPU count is therefore safe to reuse on another.

## Train-Time Eval Snapshots
Evaluations run at every checkpoint threshold during training (`n_chkpts` evenly spaced thresholds, the last landing at the conclusion of training), each persisting under `artifacts/<campaign>/settings/<setting>/<dataset>/<seed>/evals/eval<k>/` (`k` = 1..`n_chkpts`, `eval<n_chkpts>` being the final eval). Each eval writes one metrics file per eval group (`{native,native_macro,joint,joint_macro}.json`), and — when the trial's seed falls in `manif_viz.yaml`'s seed window (the manifold-viz master switch, see below) — caches its **raw** t-SNE/PCA projections to `projections.npz` (UMAP appended post-trial) and its raw embeddings to `embs.npz`, and renders its plots to `viz/`; `evals/base/` (materialized from the base-eval cache) opens the sequence.

**Checkpoint selection.** Selection is **per setting**, not per trial: for each (setting, dataset) × selection criterion × eval group, one checkpoint index is chosen as the **argmax of the across-trial mean curve** — `argmax(mean(...))`, not `mean(argmax(...))` — and *every* trial of that setting/dataset is then scored at that same index. Each criterion curves the comp score it selects on (`map` → `scores.comp.map.all`, `acc` → `scores.comp.acc.i2t`) at every checkpoint of every completed trial; checkpoint 0 is the base eval, which is plotted but never a candidate, so the winner is the argmax over `1..n_chkpts` with the earliest taking ties. Selection is purely over the on-disk eval metrics: **no model weights are saved** (trials write no `chkpts/final/` or `chkpts/best/`; the only thing under `chkpts/` is `in_progress/`, the transient resume state, deleted when the trial completes).

The pick **moves as trials land** — a new trial changes the mean curve — so at each trial completion the whole selection is redone for that setting/dataset over *all* its completed trials, and each one's `evals/_best/{map,acc}/{native,native_macro,joint,joint_macro}.json` is rewritten as a copy of that trial's own `evals/eval<idx_best>/<group>.json` (the copied `chkpt` field records the source checkpoint). A trial's scores in the reports can therefore change when a *sibling* trial finishes. **All reporting** aggregates these `_best` files, not the final eval's. Per criterion: the per-dataset `stats/{map,acc}/<group>/metrics.json` aggregates (+ their `metrics_listview.json` siblings) and the `stats/metrics/{map,acc}/<group>.xlsx` workbooks — each workbook keeps both its mAP and accuracy sheets, all sourced from its own criterion's checkpoints (e.g. the `map/` workbooks' accuracy sheet holds the acc scores at the best-mAP checkpoint), with the banner naming the selection (e.g. `bc_dev - dev3 (Joint-Macro; mAP-selection)`), plus a third `Hardware Performance` sheet laid out like the score sheets (per-dataset tables, cross-dataset Mean table with the per-setting crash totals, per-seed blocks) over the trials' wall-clock/peak-memory readings from `trial_metadata.json`; the `stats/<dataset>/map/<group>/metrics.png` tables source the mAP-selected checkpoints and `acc/<group>/metrics.png` the acc-selected ones.

**Convergence plots.** Alongside each table, `stats/<dataset>/{map,acc}/<group>/convergence.png` overlays *every* setting's across-trial mean curve (read back from the per-setting `chkpt_means.pkl`) on one log-scaled checkpoint axis: all settings in grey, and the winner — the setting with the highest mean at its **own** selected checkpoint, ties to the first setting in campaign order — redrawn in black on top, its selection starred and its score drawn across the plot as a red dashed line. Checkpoint 0 (the base eval) has no place on a log axis and is dropped, so the curves start at checkpoint 1; a setting with no completed trial in that dataset has no `chkpt_means.pkl` and is simply absent.

A trial counts as complete — for both the mean curve and the aggregates — once its **final eval** is on disk. Each eval file's `chkpt` field carries `"<k>/<n_chkpts>"`, so the highest-numbered `evals/eval*` dir answers this without `n_chkpts` being threaded in from config. (The old signal, a written `evals/_best/`, can no longer serve: `_best/` is now derived from the completed set rather than written by each trial for itself.)

**Render cadence.** The per-dataset `stats/{map,acc}/<group>/` artifacts and the campaign-level `stats/<dataset>/{map,acc}/<group>/{metrics,convergence}.png` tables and plots for that trial's dataset refresh at every trial completion. The **cross-dataset** `stats/metrics/{map,acc}/<group>.xlsx` workbooks refresh only when a **seed completes across the whole matrix** — i.e. that seed has a completed trial in every (setting, dataset), one full pass of the campaign (trials run seed-major, so this is the natural boundary). Mid-sweep they would otherwise mix settings reselected against different trial counts. Since a campaign can end *between* sweeps — interrupted, or with a (setting, dataset) that never succeeds — the runner also renders the tables, plots, and workbooks once on the way out, so the workbooks never sit a sweep behind. `python -m tools.regen_stats <campaign>` re-renders them unconditionally. Similarly, `python -m tools.regen_learning_curves <campaign>` re-renders every trial's `learning_curves/*.png` from its persisted `data_trial.pkl` with the current `utils/report.py` plotting code — no train/eval rerun — so plot styling/layout edits take effect for already-run campaigns.

**Checkpoint-mean curves.** The curves the selection is made from are kept as artifacts: `stats/{map,acc}/<group>/chkpt_means.pkl` holds `{"n_trials", "chkpts", "means", "spreads", "idx_best"}` (numpy arrays; `spreads` uses `config/stats.yaml`'s `spread_type` and is all-zero for a single trial, `idx_best` is the selected index), and `chkpt_means.png` plots the mean curve with a mean ± spread band and the selection marked. The selection itself is also recorded in `setting_metadata.json` under `best_chkpt[<dataset>][<criterion>][<group>]` as `{"idx", "n_trials", "mean"}`, rewritten at each trial end.

Within each eval dir, `projections.npz` is the durable cache (the sharded t-SNE that produced it runs once, collectively across ranks, reusing the eval embeddings; UMAP is appended to the same file post-trial, see below) and the `viz/` plots are rendered from it on rank 0 — a separation that keeps the collective compute off the render path and lets each eval's t-SNE orientation be derived from the on-disk caches rather than carried as live state across evals.

## Manifold Visualizations
Each `viz/` directory holds PCA, t-SNE, UMAP, and **spherical UMAP** projections of the eval image embeddings, grouped by panel layout above the method dir: the two-panel grids under `viz/2panel/{pca,tsne,umap,umap_sphere}/`, the 2×4 composite under `viz/7panel/{pca,tsne,umap,umap_sphere}/`, and a cross-method grid under `viz/8panel/` (which tiles every method into one figure, so it has no per-method subdir). For each method there are six outputs, each a flush grid of panels (a static `.png`): the ID-only (`id`), OOD-only (`ood`), and joint ID+OOD (`joint`) projections, plus the joint projection with one partition masked out — OOD hidden (`joint_id`) and ID hidden (`joint_ood`) — all written under `2panel/`, and a 2×4 joint composite written to `7panel/{pca,tsne,umap,umap_sphere}/joint`. Each non-composite output pairs the same projection colored two ways side by side: by leaf class (left) and by penultimate-level group (right). The composite's columns are OOD / ID / ID+OOD / n-shot and its rows are leaf / penult (the n-shot panel occupies the leaf row only, no penult cell); the n-shot panel colors ID points by their n-shot bucket (matching the learning curves) and draws OOD points black. The masked variants (`joint_id` / `joint_ood`) share the exact `joint` geometry — the other partition's points are made transparent, not removed, so coordinates and axis limits are identical — making it easy to see where each partition falls within the shared embedding. The **8panel** output adds, for each of the five 2panel subjects (`id`, `ood`, `joint`, `joint_id`, `joint_ood`), a 2×4 grid — one **column** per projection method (PCA, t-SNE, UMAP, UMAP-sphere, left to right, named by column header) against one **row** per coloring (leaf on top, penult underneath) — written to `viz/8panel/<subject>`, which is the view for comparing how the methods lay out the same embedding. (Its name is its cell count — methods × colorings — like its `2panel` / `7panel` siblings, so adding a projection method renames the group.) Generation of each group is independently toggled by `manif_viz.yaml`'s `plot_2panel` / `plot_7panel` / `plot_8panel`, applying to both the per-eval plots and the evolving GIFs. These three flags gate only rendering; above them, `manif_viz.yaml`'s `n_seeds` / `n_seeds_offset` are the master switch for the whole manifold-viz subsystem. Together they define a **window** over each setting/dataset group's seed sweep (whose length is the camp config's `n_trials`): a trial gets manifold viz iff `n_seeds_offset <= idx_seed < n_seeds_offset + n_seeds`. So `n_seeds: 1, n_seeds_offset: 0` vizzes each group's first seed, `n_seeds: 1, n_seeds_offset: 1` skips the first and vizzes the second, and `n_seeds: 0` disables the subsystem outright: projections are neither computed nor cached and nothing is rendered (the eval embeddings aren't even gathered for viz), a fast path for runs that don't need the visualizations. A window the seed sweep doesn't reach — say `n_seeds_offset: 1` on a single-seed campaign — is **not** an error and draws no warning; those trials simply get no viz, and because the gate is evaluated per trial from its own `idx_seed`, extending the campaign's seed count later brings the newly-run trials into the window automatically. When a trial's seed is in the window but all three `plot_*` flags are off, projections are still computed and cached to `projections.npz` (e.g. for later offline rendering) while no plots are drawn.

`manif_viz.yaml`'s `tsne.perplexity` / `tsne.n_iter` and `umap.n_neighbors` / `umap.min_dist` / `umap.n_iter` / `umap.n_iter_sphere` set the two non-linear methods' hyperparameters. (Changing any of them invalidates the cached coordinates in `projections.npz`; regenerate the affected evals — including the shared `base_eval_cache/` for the t-SNE ones.)

**Spherical UMAP.** The embeddings are L2-normalized (`models.py`) — they already live on the surface of a unit hypersphere, and the loss compares them by cosine similarity — so flattening them into a plane has to tear that manifold somewhere. `umap_sphere` matches both ends of the projection to the data instead: a cosine input metric and `output_metric="haversine"`, which embeds onto a 2-sphere rather than a plane. Its cache entry is `(N, 3)` **unit vectors**, not the `(N, 2)` of the planar methods. It is drawn as one opaque ball — the panel color from `bg_color`, on a white background — seen from a single fixed viewpoint, with the camera-facing hemisphere alone plotted, over spherical gridlines (parallels and meridians every 30°) that give the layout a frame of reference and make the pole legible. The panel is not a 3D axes: the viewpoint is fixed and the far side is never drawn, so the ball is a flat filled circle with the near hemisphere orthographically projected onto it. That is deliberate — a real 3D axes re-sorts scatter points by depth on every draw, which would silently undo the shuffled draw order every other panel gets.

**Where each method is computed.** PCA and t-SNE are computed **in the training loop**, collectively across ranks on the GPU (the sharded `_tsne_torch`), and cached to `projections.npz`. Neither UMAP variant has a sharded GPU implementation here, so both are fit **after** the trial, on CPU, by the same detached render worker that draws the plots (`tools/regen_manif_viz.py` → `compute_umap_projections`) — reading each eval's cached `embs.npz` and appending `umap_{id,ood,joint}` and `umap_sphere_{id,ood,joint}` to that eval's `projections.npz`. That keeps them entirely off the collective path (a hang there costs a multi-GPU trial) and overlaps their cost with the next trial's training. It is idempotent: once the coordinates are cached, a re-render reuses them rather than refitting.

**Re-rendering by hand.** `python -m tools.regen_manif_viz <campaign>` re-renders every trial in a campaign from its cached projections — no train/eval rerun — sweeping the matrix in `campaign_metadata.json` and skipping trials that never ran. It reads the **live** `config/manif_viz.yaml`, so edits to colors, `bg_color`, `eval_duration`, `orient.ema_tau` and the `plot_*` group toggles take effect for an already-run campaign; the cached coordinates are reused, so `tsne.*` / `umap.*` changes need the coords deleted first. Pass a single trial (`<campaign>/settings/<setting>/<dataset>/<seed>`) to scope it to one, `evo_only` / `no_evo` to render only the evolving GIFs or only the per-eval plots, and `snapshot` to use the campaign's frozen config instead of the live one — which is the form the post-trial render worker spawns. Because UMAP needs the raw embeddings, **every** viz trial caches `embs.npz` per eval (roughly `N × embed_dim × 2` bytes), not just pooled ones.

The two UMAP variants differ *only* in their output space, so they share their input side: one cosine nearest-neighbor search per subject is computed once (`_umap_knn`) and handed to both fits via umap-learn's `precomputed_knn`. This is why both read their `n_neighbors` from the same `manif_viz.yaml` `umap` block — the graph is only shareable when the two agree — and why the flat fit uses a cosine metric too (on normalized vectors cosine and euclidean induce identical neighbor *sets*, so that is a change of units, not of structure).

Be aware what that does and doesn't buy. Measured on 6k × 512 embeddings at UMAP's default epoch counts the neighbor search is ~0.1s, the flat layout ~0.5s, and the **spherical layout ~16s** — so sharing the graph saves almost nothing in wall clock, and the honest cost model is that a spherical *epoch* is roughly 30× a flat one, dominated entirely by umap-learn's haversine gradient. That is why the spherical variant runs its own, lower epoch count (`umap.n_iter_sphere`, vs the flat variant's `umap.n_iter`): the chained warm starts land each fit near its converged layout already, so the sphere doesn't need the full default schedule. Sharing is kept for the guarantee rather than the seconds: both variants are provably fit from an identical input graph, so any difference between their plots is output-space only. If the post-trial render stops finishing inside the next trial, the levers that actually matter are `umap.n_iter_sphere` (the dominant term) and `pooled.budget` (the pooled fit's point count).

**Cross-eval orientation.** A projection's frame is not pinned by the data: t-SNE and UMAP can come out of consecutive evals arbitrarily rotated or mirrored, and PCA's component *signs* are chosen from the data (sklearn's `svd_flip`) so they too can flip between evals. Untreated, the evolving GIFs spin and mirror instead of showing the embedding move. Each method is therefore aligned to a running reference of per-class centers of mass, EMA-smoothed by `manif_viz.yaml`'s `orient.ema_tau`: **t-SNE and UMAP** get a full rigid alignment (count-weighted orthogonal Procrustes — rotation + reflection — over the whole class constellation, which is robust where a single anchor triangle would flip on near-collinear anchors); **UMAP-sphere** gets the same treatment solved in SO(3): count-weighted Procrustes finds the one rigid motion of the whole ball that moves the cluster constellation — seen and unseen clusters alike — the least, so the ball and its gridlines stay fixed from the viewer's standpoint while the clusters glide across them between evals, the way the planar evolutions behave. Minimal rotation also keeps the same side of the ball facing the camera, which matters since only one hemisphere is drawn. Nothing is pinned after the first eval: the bootstrap names the starting frame the way t-SNE's does — the largest class at the north pole, the second-largest on the central meridian, the third fixing handedness — and later evals drift off those landmarks only as much as the layout genuinely moves. while **PCA** keeps its principal axes and has only its per-axis sign aligned (a closed-form choice per axis; rotating PCA would destroy what the plot shows). UMAP additionally gets continuity at the source: each eval's fit is **initialized from the previous eval's layout** (the eval set is in identical per-sample order at every threshold, so row *i* is the same sample), with the base eval seeding from its own PCA. That matters because orientation can only remove residual rigid motion — it cannot repair a layout that genuinely reorganized, which UMAP will do if every eval starts from a fresh init.

Class colors are consistent across **every** plot pertaining to a given `(dataset, split)`. The leaf and penult color maps are built over the full set of dataset classes (keyed by class identity, not per-plot position), so any given class or penultimate-level group renders in the same color in every plot — across base / train-time / final evals, across all four projection methods, and across the ID, OOD, and joint panels (where shared penult-level groups line up by color).

Within every panel, points are drawn in a **shuffled** order (fixed seed, so it is reproducible) rather than class-by-class, so no single class ends up plotted entirely on top of the others. This applies to the spherical panels too: those cull the hemisphere facing away from the camera and then project the survivors onto the viewpoint's image plane, so they are ordinary 2D scatters by the time the draw order is applied — a real 3D axes would silently re-sort them by depth and undo it.

A set of **evolving GIFs** is also written under `<trial>/viz/{2panel,7panel}/{pca,tsne,umap,umap_sphere}/` and `<trial>/viz/8panel/` at the end of training — one per output above (the `2panel/` grids `id`, `ood`, `joint`, `joint_id`, `joint_ood`, the `7panel/` composite `joint`, and the five `8panel/` cross-method plots). Each plays the training trajectory across the eval sequence (`base → eval1 → … → eval<n_chkpts>`), holding the axes and gridlines fixed across evals so only the points move. Every method's orientation is stabilized across evals as described above (reference EMA-smoothed by `orient.ema_tau`) so the projections don't spin or mirror-flip from one eval to the next. Each eval's standalone plots reuse that exact orientation (derived from the cached reference — each eval caches its running orientation reference next to its `projections.npz` so the next eval reads it in O(1) instead of re-sweeping every prior eval, falling back to a full re-sweep if that cache is absent or was written under a different `ema_tau`), so they match the corresponding evolving-GIF frame.

### Pooled (shared-frame) manifold viz
The per-eval plots and evolving GIFs above fit each method **separately** per eval and stitch the frames together via cross-eval orientation. The **pooled** viz instead fits **one** shared projection per method over **all** eval thresholds' embeddings pooled, then renders each threshold as a **masked subset** of that single layout — so the geometry is identical across thresholds (no orientation needed) and the plots show the eval set migrating through a fixed frame as training progresses. It is enabled by `manif_viz.yaml`'s `pooled.enabled` and writes the same panel groups (gated by the same `plot_2panel` / `plot_7panel` / `plot_8panel` flags) under `<eval>/viz_pooled/` (per-threshold) and `<trial>/viz_pooled/` (evolving GIFs), alongside the oriented `viz/` outputs. All four methods get a pooled fit: PCA and t-SNE at end-of-trial on the GPU, both UMAP variants post-trial on CPU in the render worker (`compute_umap_pooled`, again off one shared neighbor search), which reuses the exact subsample the GPU pass recorded so every method covers the same points in the same row order. Since one fit spans the whole sequence, the pooled outputs need neither the init chain nor orientation.

The t-SNE's memory scales linearly with the pool size, but its per-iteration repulsion compute is still O(N²), so the pool is class-stratified **subsampled** to about `manif_viz.yaml`'s `pooled.budget` points total across all thresholds — an absolute sample volume, not a multiplier — to bound the pooled fit's runtime. If the full pool `N_full × n_evals` (where `N_full` is one eval's ID+OOD size) is at most `budget`, all samples are used; otherwise the pool is subsampled to roughly `budget` points total (a larger `budget` costs proportionally more compute time; memory is not a constraint). The subsample indices are fixed across thresholds (the eval set is in identical per-sample order at every threshold), so a plotted point is the **same physical sample** in every frame; PCA and both UMAP variants use the same subsample as t-SNE (its row indices are recorded in `projections_pooled.npz` as `idx_id`/`idx_ood`), and class colors are built from the **full** eval set (not the subsample), so they still match the other plots. This is fed by the same per-eval `embs.npz` caches the UMAP fits read; the end-of-trial pooled PCA/t-SNE (a single collective sharded t-SNE, run once after the final eval) reads them all and writes each threshold's masked block to `projections_pooled.npz`. Note the pooled UMAP is a **CPU** fit over the whole `budget`-sized pool, so `budget` governs its cost far more directly than it does the GPU-sharded t-SNE's — it is the knob to turn down if the post-trial render stops finishing inside the next trial.

## Tensor Dimensionality Annotation Conventions:
B: Batch dim. <br>
SB: Sub-batch dim. (multi-GPU) <br>
C: Channels <br>
H: Image height <br>
W: Image width <br>
L: Num. classes <br>
D: Embedding dim. <br>
T: Num. tokens i.e. context length <br>
P: Num. text sequences <br>
Q: Num queries (retrieval) <br>
N: Gallery size (retrieval) <br>
G: Num. GPUs (DDP) i.e. world size <br>
U: Chunk size <br>
