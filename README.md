The BioCosmos project aims to develop a multimodal, AI-powered interface for biologists to search a species database via image or text. At it's core is a vision-language model (VLM) trained to capture fine-grained cross-modal relationships: jointly learned image and text encoders bridge the gap between visual data and natural language.

This element of the BioCosmos project focuses on developing fine-tuning methods to obtain robust VLM performance on long-tailed, hierarchically structured data in support of multimodal biodiversity retrieval tasks. Our goal is to systematically experiment with methods best suited to this regime (e.g. class-imbalance handling, hierarchical targets/metrics, n-shot-aware sampling, hard-negative shaping) and to deliver the strongest fine-tuned VLMs possible for downstream use in the BioCosmos app.

# Setup

Setup is intended for use with B200s on HiPerGator.

Pull repo and navigate:
```
git clone https://github.com/odobon1/biocosmos.git
cd biocosmos
```

Place environment_b200.yaml in your home directory `/home/<user>`.
Navigate to home directory and create env:
```
conda env create -f environment_b200.yaml
```
Note: `environment.yaml` can be used for non-B200 jobs.

Navigate back to repo directory and activate env:
```
conda activate biocosmos_b200
```

Run setup script (this takes about an hour to run):
```
./setup.sh
```

`setup.sh` generates metadata, including split S29-42 (by default), generates various data indexing structures needed for train and eval. See [metadata/README.md](metadata/README.md) for details.

# Train & Eval
Training and evaluation are config-driven: switch models, losses, LR schedules, batch size in YAML (no code edits). The training config lives at `config/train/train.yaml` and the standalone eval config at `config/eval.yaml`. The loader merges `train.yaml` with `lr_sched.yaml` and `loss.yaml` under the hood.

## Train
1. Edit `config/train/train.yaml`:
    * `model_type`, `loss_type`, `targ_type`, `lr_sched_type`, etc.
    * Mixed precision & activation checkpointing can be toggled: `mixed_prec`, `act_chkpt`
2. Run:
    ```
    python train.py
    ```
    This reads `config/train/train.yaml`, seeds, builds the model, applies class weighting (if enabled) and trains.

    Tip: LR scheduler options and parameters are in `config/train/lr_sched.yaml`. Pick via `lr_sched_type` in `train.yaml`.

## Evaluate a trained model
1. In `config/eval.yaml`, set `rdpath_trial` to the trial directory (e.g. `artifacts/dev/dev/42`).
2. Run:
    ```
    python eval.py
    ```
    When `rdpath_trial` is set, eval overrides `model_type`, `loss_type`, and `split_name` from the trial's saved metadata.

## Evaluate a base model
1. In `config/eval.yaml`, set `rdpath_trial: null`, then set `model_type`, `loss_type`, `split_name`, etc. as desired.
2. Run:
    ```
    python eval.py
    ```

<br>

# Supported Architectures

The **Max Batch Size** column indicates max batch size whilst training on a single B200 using mixed precision and activation checkpointing.

## CLIP - ViT Variants

| Model ID <br> (Internal) | Model ID <br> (open_clip)     | Max Batch Size | Pretrain  | Num Params | Embedding <br> Dimension | (V) Layers | (V) Width | (V) Heads | (T) Layers | (T) Width | (T) Heads | **(\*1)** Learning Rate |
|--------------------------|-------------------------------|----------------|-----------|------------|--------------------------|------------|-----------|-----------|------------|-----------|-----------|-------------------------|
| `clip_vitb32`            | `ViT-B-32`                    | 16,384         | **(\*2)** | 151M       | 512                      | 12         | 768       | 12        | 12         | 512       | 8         | 5e-4                    |
| `clip_vitb16`            | `ViT-B-16`                    | 8,192          | **(\*3)** | 150M       | 512                      | 12         | 768       | 12        | 12         | 512       | 8         | 5e-4                    |
| `clip_vitl14`            | `ViT-L-14`                    | 4,096          | **(\*4)** | 428M       | 768                      | 24         | 1,024     | 16        | 12         | 768       | 12        | 4e-4                    |
| `clip_vitl14_336`        | `ViT-L-14-336`                | 2,048          | `openai`  | 428M       | 768                      | 24         | 1,024     | 16        | 12         | 768       | 12        | 2e-5 (investigate)      |
| `bioclip`                | `hf-hub:imageomics/bioclip`   | 8,192          | NA        | -          | -                        | -          | -         | -         | -          | -         | -         |                         |
| `bioclip2`               | `hf-hub:imageomics/bioclip-2` | 4,096          | NA        | -          | -                        | -          | -         | -         | -          | -         | -         |                         |

**(V)** denotes vision transformer, **(T)** denotes text transformer. Model specs not included for `bioclip` and `bioclip2` because they are redundant. `bioclip` is a CLIP ViT-B/16 fine-tuned on the TOL-10M dataset, `bioclip2` is a CLIP ViT-L/14 fine-tuned on the TOL-200M dataset.

**(\*1)** Peak LR from seminal CLIP arXiv Table 20.

**(\*2)** `ViT-B-32` Pretrain-Dataset Model Weights Available:
`openai`, `laion400m_e31`, `laion400m_e32`, `laion2b_e16`, `laion2b_s34b_b79k`, `datacomp_xl_s13b_b90k`, `datacomp_m_s128m_b4k`, `commonpool_m_clip_s128m_b4k`, `commonpool_m_laion_s128m_b4k`, `commonpool_m_image_s128m_b4k`, `commonpool_m_text_s128m_b4k`, `commonpool_m_basic_s128m_b4k`, `commonpool_m_s128m_b4k`, `datacomp_s_s13m_b4k`, `commonpool_s_clip_s13m_b4k`, `commonpool_s_laion_s13m_b4k`, `commonpool_s_image_s13m_b4k`, `commonpool_s_text_s13m_b4k`, `commonpool_s_basic_s13m_b4k`, `commonpool_s_s13m_b4k`, `metaclip_400m`, `metaclip_fullcc`

**(\*3)** `ViT-B-16` Pretrain-Dataset Model Weights Available: `openai`, `laion400m_e31`, `laion400m_e32`, `laion2b_s34b_b88k`, `datacomp_xl_s13b_b90k`, `datacomp_l_s1b_b8k`, `commonpool_l_clip_s1b_b8k`, `commonpool_l_laion_s1b_b8k`, `commonpool_l_image_s1b_b8k`, `commonpool_l_text_s1b_b8k`, `commonpool_l_basic_s1b_b8k`, `commonpool_l_s1b_b8k`, `dfn2b`, `metaclip_400m`, `metaclip_fullcc`

**(\*4)** `ViT-L-14` Pretrain-Dataset Model Weights Available: `openai`, `laion400m_e31`, `laion400m_e32`, `laion2b_s32b_b82k`, `datacomp_xl_s13b_b90k`, `commonpool_xl_clip_s13b_b90k`, `commonpool_xl_laion_s13b_b90k`, `commonpool_xl_s13b_b90k`, `metaclip_400m`, `metaclip_fullcc`, `dfn2b`, `dfn2b_s39b`

As we can see the CLIP ViT-series have received quite a lot of attention (no pun attended).

Note: despite the name, both the open-source OpenCLIP and the OpenAI CLIP model weights are available through `open_clip`. Pretrain dataset = "openai" --> **OpenAI CLIP** (a.k.a. "original CLIP"), trained using a private recipe on a private pretraining dataset. Pretrain dataset = <anything other than "openai"> --> **OpenCLIP** ~ community replications of CLIP trained on open sourced datasets as specified in the Pretrain column.

## CLIP - ResNet Variants

| Model ID <br> (Internal) | Model ID <br> (open_clip) | Max Batch Size | Pretrain                     | Num Params | Embedding <br> Dimension |
|--------------------------|---------------------------|----------------|------------------------------|------------|--------------------------|
| `clip_rn50`              | `RN50`                    | 2,048          | `openai`, `yfcc15m`, `cc12m` | 102M       | 1,024                    |
| `clip_rn101`             | `RN101`                   | 2,048          | `openai`, `yfcc15m`          | 120M       | 1,024                    |
| `clip_rn50x4`            | `RN50x4`                  | 1,024          | `openai`                     | 178M       | 1,024                    |
| `clip_rn50x16`           | `RN50x16`                 | 256            | `openai`                     | 291M       | 1,024                    |
| `clip_rn50x64`           | `RN50x64`                 | 128            | `openai`                     | 623M       | 1,024                    |

## SigLIP Variants

| Model ID <br> (Internal) | Model ID <br> (open_clip) | Max Batch Size | Pretrain | Num Params | Embedding <br> Dimension |
|--------------------------|---------------------------|----------------|----------|------------|--------------------------|
| `siglip_vitb16`          | `ViT-B-16-SigLIP`         | 8,192          | `webli`  | 203M       | 768                      |
| `siglip_vitb16_384`      | `ViT-B-16-SigLIP-384`     | 2,048          | `webli`  | 203M       | 768                      |
| `siglip_vitl16_384`      | `ViT-L-16-SigLIP-384`     | 2,048          | `webli`  | 653M       | 1,024                    |
| `siglip_vitso400m14`     | `ViT-SO400M-14-SigLIP`    | 2,048          | `webli`  | 877M       | 1,152                    |
| `siglip2_vitb16`         | `ViT-B-16-SigLIP2`        | 8,192          | `webli`  | 375M       | 768                      |
| `siglip2_vitb16_384`     | `ViT-B-16-SigLIP2-384`    | 2,048          | `webli`  | 375M       | 768                      |
| `siglip2_vitl16_384`     | `ViT-L-16-SigLIP2-384`    | 2,048          | `webli`  | 822M       | 1,024                    |
| `siglip2_vitso400m14`    | `ViT-SO400M-14-SigLIP2`   | 2,048          | `webli`  | 1,136M     | 1,152                    |
| `siglip2_vitgopt16_384`  | `ViT-gopt-16-SigLIP2-384` | 512            | `webli`  | 1,870M     | 1,536                    |

## ViTamin Variants

| Model ID <br> (Internal) | Model ID <br> (open_clip) | Max Batch Size | Pretrain     | Num Params | Embedding <br> Dimension |
|--------------------------|---------------------------|----------------|--------------|------------|--------------------------|
| `vitamin_s`              | `ViTamin-S`               | 4,096          | `datacomp1b` | 62M        | 512                      |
| `vitamin_s_ltt`          | `ViTamin-S-LTT`           | 4,096          | `datacomp1b` | 62M        | 512                      |
| `vitamin_b`              | `ViTamin-B`               | 2,048          | `datacomp1b` | 128M       | 768                      |
| `vitamin_b_ltt`          | `ViTamin-B-LTT`           | 2,048          | `datacomp1b` | 128M       | 768                      |
| `vitamin_l`              | `ViTamin-L`               | 1,024          | `datacomp1b` | 457M       | 768                      |
| `vitamin_l_256`          | `ViTamin-L-256`           | 1,024          | `datacomp1b` | 457M       | 768                      |
| `vitamin_l_336`          | `ViTamin-L-336`           | 512            | `datacomp1b` | 457M       | 768                      |
| `vitamin_l_384`          | `ViTamin-L-384`           | 512            | `datacomp1b` | 457M       | 768                      |
| `vitamin_l2`             | `ViTamin-L2`              | 1,024          | `datacomp1b` | 688M       | 1,024                    |
| `vitamin_l2_384`         | `ViTamin-L2-384`          | 512            | `datacomp1b` | 688M       | 1,024                    |
| `vitamin_xl_384`         | `ViTamin-XL-384`          | 512            | `datacomp1b` | 925M       | 1,152                    |

## More Models

More model types can be viewed via running the following at the command line or in a jupyter notebook:
```python
import open_clip
open_clip.list_pretrained()
```

(Only CLIP, SigLIP, and ViTamin variants are currently supported. `open_clip` models outside these families likely won't work)

# Notes

## Batching
During training, partial batches are dropped by default. For more granular batching tactics e.g. dorsal/ventral, partial batches are dropped
from each category, which may result in fewer batches per epoch. Dorsal/ventral batching can only be toggled for train.
For eval, loss is computed for full batches only, although performance computation includes partial batches.