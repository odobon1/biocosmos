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

Create and activate environment:
```
conda env create -f environment_b200.yaml
conda activate biocosmos_b200
```
Note: `environment.yaml` can be used for non-B200 jobs.

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

Note: The full similarity matrix is computed for all model types, including SigLIP (i.e. the chunked implementation described in the seminal SigLIP work is not currently utilized).

## Train
1. Edit `config/train.yaml`:
    * `model_type`, `loss_type`, `targ_type`, `lr_sched_type`, etc.
    * Mixed precision & activation checkpointing can be toggled: `mixed_prec`, `act_chkpt`
2. Run:
    ```
    torchrun --standalone --nproc-per-node=auto -m train
    ```
    This reads `config/train/train.yaml`, seeds, builds the model, applies class weighting (if enabled) and trains.

    Tip: Cosine LR scheduler parameters are in `config/train.yaml` under `opt.lr`.

## Evaluate a trained model
1. In `config/eval.yaml`, set `rdpath_model` to checkpointed model directory (e.g. `artifacts/dev/iw/lepid/42/chkpts/final`).
2. Run:
    ```
    torchrun --standalone --nproc-per-node=auto -m eval
    ```
    When `rdpath_model` is set, eval overrides `dataset`, `split`, `model_type`, `non_causal`, `img_norm` from setting + trial saved metadata.

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
    * `datasets`
    * `baseline_overrides` — per-setting overrides (each item needs a unique `name`)
    * `suffix` — appended to the campaign name (`null` for none)

   The campaign is named `<campaign>_<suffix>` (or just `<campaign>` when `suffix` is `null`).
2. Launch the campaign, selecting the config by name:
    ```
    python -m campaign_runner --<campaign>   # e.g. python -m campaign_runner --dev_basic
    ```
3. Each trial is launched in a fresh subprocess (`campaign_trial_runner`) to isolate DDP/DataLoader worker state between trials.
4. If a trial fails, campaign execution continues and the error is written to that trial's `error.log` (`artifacts/<campaign>/<setting>/<dataset>/<seed>/error.log`).
5. `artifacts/<campaign>/manifest.log` tracks trial progress, bucketing every planned trial (by `setting/dataset/seed`) into Failed / Completed / In Progress / Queued. It is regenerated at kickoff and at each trial's start and finish.

**Note:** When resuming a campaign, the environment must allocate the same number of GPUs as the original run. The GPU count is saved to `artifacts/<campaign>/campaign_metadata.json` on first launch; a mismatch on resume raises an error before any trials execute.

## Config Override Layers

Training config is assembled from multiple sources. Layers are listed in increasing priority order — each layer overwrites anything set by earlier layers.

| Priority | Source | Applied by | Description |
|----------|--------|-----------|-------------|
| 1 (lowest) | `config/train.yaml` | `load_train_config_dict()` | Base config; the starting point for all training runs. |
| 2 | Campaign runner injections | `run_campaign()` | Overwrites `campaign`, `setting`, `seed`, `dataset`, `standalone` from the campaign matrix. Not applicable in standalone training. |
| 3 | `config/model_specific.yaml` | `apply_model_specific_opt_defaults()` | Fills `opt.l2reg` and `opt.beta2` **only if `null`**, based on model family (`clip` or `siglip`). Has no effect if those fields are already set in `config/train.yaml`. |
| 4 | `debug_mode` overrides | `apply_train_debug_overrides()` | If `dev.debug_mode: true`, forces `split → "dev"`, `sample_volume → 20_000`, `chkpt_every → 10_000`, `batch_size → 1_024`. |
| 5 (highest) | `baseline_overrides` (campaign) | `build_train_config()` via `_setting_overrides` | Per-setting overrides defined in the campaign config (`config/camps/<campaign>.yaml`). |

In standalone training (`python -m train`), only layers 1, 3, and 4 apply.

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

## Base Model Performance Cache
The base-model evaluation at the start of each trial (the untrained model's performance) is cached at `base_eval_cache/<model_type>/<img_norm>/<dataset>/<split>/`. On the first trial for a given `(model_type, img_norm, dataset, split)`, the base eval runs and its scores are cached to `metrics.json` and its raw t-SNE/PCA projections to `projections.npz`; subsequent trials reuse the cached results instead of re-running the base eval. The cached `metrics.json` mirrors a checkpoint `metrics.json` minus the `loss_raw` and `n_samps_seen` fields. To force base evals to recompute, delete the `base_eval_cache/` directory.

The cached scores are reproducible across single-GPU and multi-GPU runs: performance metrics are computed on the full set of embeddings gathered from all ranks, so the complete evaluation set, and thus the resulting scores, are identical regardless of `world_size`. A cache written on one GPU count is therefore safe to reuse on another.

## Train-Time Eval Snapshots
`dev.traintime_evals` controls whether evaluations run at the checkpoint thresholds during training: when `true`, each threshold eval runs and persists under `artifacts/<campaign>/<setting>/<dataset>/<seed>/evals/<thresh>/`; when `false`, only the base and final evals run (so the learning curves carry just those two eval points). Each eval that runs writes a `metrics.json`, and — when `dev.viz_manifold` is on (the manifold-viz master switch, see below) — caches its **raw** t-SNE/PCA projections to `projections.npz` and renders its plots to `viz/`; `evals/_base/` (copied from the base-model cache) and `evals/final/` are always written, with `evals/<thresh>/` snapshots in between when `traintime_evals` is on.

Within each eval dir, `projections.npz` is the durable cache (the sharded t-SNE that produced it runs once, collectively across ranks, reusing the eval embeddings) and the `viz/` plots are rendered from it on rank 0 — a separation that keeps the collective compute off the render path and lets each eval's t-SNE orientation be derived from the on-disk caches rather than carried as live state across evals.

## Manifold Visualizations
Each `viz/` directory holds t-SNE and PCA projections of the eval image embeddings, grouped by panel layout above the method dir: the two-panel grids under `viz/2panel/{tsne,pca}/`, the 2×4 composite under `viz/7panel/{tsne,pca}/`, and a cross-method 2×2 under `viz/4panel/` (which tiles both methods into one figure, so it has no per-method subdir). For each method there are six outputs, each a flush grid of panels (a `.png` when `n_stoch_layers: 1`, otherwise a strobe `.gif` — see below): the ID-only (`id`), OOD-only (`ood`), and joint ID+OOD (`fullset`) projections, plus the joint projection with one partition masked out — OOD hidden (`fullset_id`) and ID hidden (`fullset_ood`) — all written under `2panel/`, and a 2×4 fullset composite written to `7panel/{tsne,pca}/fullset`. Each non-composite output pairs the same projection colored two ways side by side: by leaf class (left) and by penultimate-level group (right). The composite's columns are OOD / ID / ID+OOD / n-shot and its rows are leaf / penult (the n-shot panel occupies the leaf row only, no penult cell); the n-shot panel colors ID points by their n-shot bucket (matching the learning curves) and draws OOD points black. The masked variants (`fullset_id` / `fullset_ood`) share the exact `fullset` geometry — the other partition's points are made transparent, not removed, so coordinates and axis limits are identical — making it easy to see where each partition falls within the shared embedding. The **4panel** output adds, for each of the five 2panel subjects (`id`, `ood`, `fullset`, `fullset_id`, `fullset_ood`), a 2×2 that stacks both projection methods (PCA top, t-SNE bottom) against both colorings (leaf left, penult right) for that subject, written to `viz/4panel/<subject>` — handy for comparing how the two methods lay out the same embedding. Generation of each group is independently toggled by `dev.manifold_viz.plot_2panel` / `plot_4panel` / `plot_7panel` (train.yaml), applying to both the per-eval plots and the evolution GIFs. These three flags gate only rendering; above them, `dev.viz_manifold` (train.yaml) is the master switch for the whole manifold-viz subsystem — when `false`, projections are neither computed nor cached and nothing is rendered (it forces the `plot_*` flags off, and the eval embeddings aren't even gathered for viz), a fast path for runs that don't need the visualizations. When `viz_manifold` is on but all three `plot_*` flags are off, projections are still computed and cached to `projections.npz` (e.g. for later offline rendering) while no plots are drawn.

`manifold_viz.yaml`'s `tsne.perplexity` sets the t-SNE perplexity. (Changing it invalidates cached `projections.npz`; regenerate the affected evals — including the shared `base_eval_cache`.)

Class colors are consistent across **every** plot pertaining to a given `(dataset, split)`. The leaf and penult color maps are built over the full set of dataset classes (keyed by class identity, not per-plot position), so any given class or penultimate-level group renders in the same color in every plot — across base / train-time / final evals, across t-SNE and PCA, and across the ID, OOD, and fullset panels (where shared penult-level groups line up by color).

`manifold_viz.yaml`'s `n_stoch_layers` controls *stochastic layering*: within a plot, points are drawn in a (deterministic) shuffled order rather than class-by-class, so no single class is plotted entirely on top of the others. The value is the number of distinct shuffle orderings rendered per eval — `n_stoch_layers: 1` writes a single static PNG (one ordering), while `n_stoch_layers > 1` writes a GIF that cycles through that many orderings.

When `dev.traintime_evals` is on, a set of **evolution GIFs** is also written under `<trial>/viz/{2panel,7panel}/{tsne,pca}/` and `<trial>/viz/4panel/` at the end of training — one per output above (the `2panel/` grids `id`, `ood`, `fullset`, `fullset_id`, `fullset_ood`, the `7panel/` composite `fullset`, and the five `4panel/` cross-method plots). Each plays the training trajectory across the eval sequence (`_base → <thresh> → … → final`), holding the axes and gridlines fixed across evals so only the points move. The t-SNE orientation is stabilized across evals (each eval's per-class CoM constellation is aligned to a running reference via orthogonal Procrustes — rotation + reflection — with the reference EMA-smoothed by `tsne.ema_tau`) so the projection doesn't spin or mirror-flip from one eval to the next; using the whole constellation rather than a single anchor triangle keeps the handedness decision robust when any few anchor classes happen to be near-collinear. Each eval's standalone plots reuse that exact orientation (derived from the cached reference — each eval caches its running orientation reference next to its `projections.npz` so the next eval reads it in O(1) instead of re-sweeping every prior eval, falling back to a full re-sweep if that cache is absent or was written under a different `ema_tau`), so they match the corresponding evolution-GIF frame.

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
G: Num. GPUs (DDP) <br>
U: Chunk size <br>
