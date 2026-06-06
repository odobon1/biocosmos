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

## Data

## Preprocessing

Run setup script:
```
./setup.sh
```

`setup.sh` generates metadata (including split) and various data indexing structures needed for train and eval. See [preprocessing/README.md](preprocessing/README.md) for details.

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
1. In `config/eval.yaml`, set `rfpath_model` to checkpointed model weights (e.g. `artifacts/dev/iw/lepid/42/chkpts/best_comp/model.pt`).
2. Run:
    ```
    torchrun --standalone --nproc-per-node=auto -m eval
    ```
    When `rfpath_model` is set, eval overrides `dataset`, `split`, `model_type`, `non_causal`, `img_norm` from setting + trial saved metadata.

## Evaluate a base model
1. In `config/eval.yaml`, set `rfpath_model: null`.
2. Run:
    ```
    torchrun --standalone --nproc-per-node=auto -m eval
    ```

## Run a campaign
1. Configure campaign matrix in `campaign_runner.py`:
    * `CAMPAIGN`
    * `SEED0`, `NUM_SEEDS`
    * `DATASETS`
    * `BASELINE_OVERRIDES`
2. Launch campaign:
    ```
    torchrun --standalone --nproc-per-node=auto -m campaign_runner
    ```
3. Each trial is launched in a fresh subprocess (`campaign_trial_runner`) to isolate DDP/DataLoader worker state between trials.
4. If a trial fails, campaign execution continues and details are appended to `artifacts/<CAMPAIGN>/errors.log`.

**Note:** When resuming a campaign, the environment must allocate the same number of GPUs as the original run. The GPU count is saved to `artifacts/<CAMPAIGN>/metadata_campaign.json` on first launch; a mismatch on resume raises an error before any trials execute.

## Config Override Layers

Training config is assembled from multiple sources. Layers are listed in increasing priority order — each layer overwrites anything set by earlier layers.

| Priority | Source | Applied by | Description |
|----------|--------|-----------|-------------|
| 1 (lowest) | `config/train.yaml` | `load_train_config_dict()` | Base config; the starting point for all training runs. |
| 2 | Campaign runner injections | `run_campaign()` | Overwrites `campaign`, `setting`, `seed`, `dataset`, `standalone` from the campaign matrix. Not applicable in standalone training. |
| 3 | `config/model_specific.yaml` | `apply_model_specific_opt_defaults()` | Fills `opt.l2reg` and `opt.beta2` **only if `null`**, based on model family (`clip` or `siglip`). Has no effect if those fields are already set in `config/train.yaml`. |
| 4 | `debug_mode` overrides | `apply_train_debug_overrides()` | If `dev.debug_mode: true`, forces `split → "dev"`, `sample_volume → 20_000`, `eval_every → 10_000`, `batch_size → 1_024`. |
| 5 (highest) | `BASELINE_OVERRIDES` (campaign) | `build_train_config()` via `_setting_overrides` | Per-setting overrides defined at the top of `campaign_runner.py`. |

In standalone training (`python -m train`), only layers 1, 3, and 4 apply.

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

## Tensor Dimensionality Annotation Conventions:
B: Batch dim. <br>
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
