# (IN PROGRESS)

# Setup

Setup is intended for use with B200s on HiPerGator.

Pull repo and navigate:
```
git clone https://github.com/odobon1/biocosmos.git
cd biocosmos
```

Create and activate env:
```
conda env create -f environment_b200.yaml
conda activate biocosmos_b200
```
Note: `environment.yaml` can be used for non-B200 sessions.


Run setup script (this takes about an hour to run):
```
./setup.sh
```

`setup.sh` generates metadata, including split S29-42 (by default), generates various data indexing structures needed for train and eval. See [metadata/README.md](metadata/README.md) for details.

To link jupyter notebooks with environment via ipykernel:
```
python -m ipykernel install --user --name biocosmos_b200 --display-name "Python (biocosmos_b200)"
```


<br>

# Supported Model Types

**Max Batch Size** indicates max batch size whilst training on a single B200 using mixed precision and activation checkpointing.

## CLIP - ViT Variants

| Model ID <br> (Internal) | Model ID <br> (open_clip)     | Max Batch Size | Pretrain  | Num Params |
|--------------------------|-------------------------------|----------------|-----------|------------|
| `bioclip`                | `hf-hub:imageomics/bioclip`   |                | NA        |            |
| `bioclip2`               | `hf-hub:imageomics/bioclip-2` |                | NA        |            |
| `clip_vitb32`            | `ViT-B-32`                    |                | **(\*1)** |            |
| `clip_vitb16`            | `ViT-B-16`                    |                | **(\*2)** |            |
| `clip_vitl14`            | `ViT-L-14`                    |                | **(\*3)** |            |
| `clip_vitl14_336`        | `ViT-L-14-336`                |                | `openai`  |            |

**(\*1)** `ViT-B-32` Pretrain-Dataset Model Weights Available:
`openai`, `laion400m_e31`, `laion400m_e32`, `laion2b_e16`, `laion2b_s34b_b79k`, `datacomp_xl_s13b_b90k`, `datacomp_m_s128m_b4k`, `commonpool_m_clip_s128m_b4k`, `commonpool_m_laion_s128m_b4k`, `commonpool_m_image_s128m_b4k`, `commonpool_m_text_s128m_b4k`, `commonpool_m_basic_s128m_b4k`, `commonpool_m_s128m_b4k`, `datacomp_s_s13m_b4k`, `commonpool_s_clip_s13m_b4k`, `commonpool_s_laion_s13m_b4k`, `commonpool_s_image_s13m_b4k`, `commonpool_s_text_s13m_b4k`, `commonpool_s_basic_s13m_b4k`, `commonpool_s_s13m_b4k`, `metaclip_400m`, `metaclip_fullcc`

**(\*2)** `ViT-B-16` Pretrain-Dataset Model Weights Available: `openai`, `laion400m_e31`, `laion400m_e32`, `laion2b_s34b_b88k`, `datacomp_xl_s13b_b90k`, `datacomp_l_s1b_b8k`, `commonpool_l_clip_s1b_b8k`, `commonpool_l_laion_s1b_b8k`, `commonpool_l_image_s1b_b8k`, `commonpool_l_text_s1b_b8k`, `commonpool_l_basic_s1b_b8k`, `commonpool_l_s1b_b8k`, `dfn2b`, `metaclip_400m`, `metaclip_fullcc`

**(\*3)** `ViT-L-14` Pretrain-Dataset Model Weights Available: `openai`, `laion400m_e31`, `laion400m_e32`, `laion2b_s32b_b82k`, `datacomp_xl_s13b_b90k`, `commonpool_xl_clip_s13b_b90k`, `commonpool_xl_laion_s13b_b90k`, `commonpool_xl_s13b_b90k`, `metaclip_400m`, `metaclip_fullcc`, `dfn2b`, `dfn2b_s39b`

As we can see the CLIP ViT-series have received quite a lot of attention (no pun attended).

Note: despite the name, both the open-source OpenCLIP and the not-so-open OpenAI CLIP model weights are available through `open_clip`. Pretrain dataset = "openai" --> **OpenAI CLIP** (a.k.a. "original CLIP"), trained on private dataset. Pretrain dataset = <anything other than "openai"> --> **OpenCLIP** ~ community replications of CLIP trained on open sourced datasets as specified in the Pretrain column.

## CLIP - ResNet Variants

| Model ID <br> (Internal) | Model ID <br> (open_clip)     | Max Batch Size | Pretrain                     | Num Params |
|--------------------------|-------------------------------|----------------|------------------------------|------------|
| `clip_rn50`              | `RN50`                        |                | `openai`, `yfcc15m`, `cc12m` |            |
| `clip_rn101`             | `RN101`                       |                | `openai`, `yfcc15m`          |            |
| `clip_rn50x4`            | `RN50x4`                      |                | `openai`                     |            |
| `clip_rn50x16`           | `RN50x16`                     |                | `openai`                     |            |
| `clip_rn50x64`           | `RN50x64`                     |                | `openai`                     |            |

## SigLIP Variants

| Model ID <br> (Internal) | Model ID <br> (open_clip)     | Max Batch Size | Pretrain | Num Params |
|--------------------------|-------------------------------|----------------|----------|------------|
| `siglip_vitb16`          | `ViT-B-16-SigLIP`             |                | `webli`  |            |
| `siglip_vitb16_384`      | `ViT-B-16-SigLIP-384`         |                | `webli`  |            |
| `siglip_vitl16_384`      | `ViT-L-16-SigLIP-384`         |                | `webli`  |            |
| `siglip_vitso400m14`     | `ViT-SO400M-14-SigLIP`        |                | `webli`  |            |
| `siglip2_vitb16`         | `ViT-B-16-SigLIP2`            |                | `webli`  |            |
| `siglip2_vitb16_384`     | `ViT-B-16-SigLIP2-384`        |                | `webli`  |            |
| `siglip2_vitl16_384`     | `ViT-L-16-SigLIP2-384`        |                | `webli`  |            |
| `siglip2_vitso400m14`    | `ViT-SO400M-14-SigLIP2`       |                | `webli`  |            |
| `siglip2_vitgopt16_384`  | `ViT-gopt-16-SigLIP2-384`     |                | `webli`  |            |

## ViTamin Variants

| Model ID <br> (Internal) | Model ID <br> (open_clip)     | Max Batch Size | Pretrain     | Num Params |
|--------------------------|-------------------------------|----------------|--------------|------------|
| `vitamin_s`              | `ViTamin-S`                   |                | `datacomp1b` |            |
| `vitamin_s_ltt`          | `ViTamin-S-LTT`               |                | `datacomp1b` |            |
| `vitamin_b`              | `ViTamin-B`                   |                | `datacomp1b` |            |
| `vitamin_b_ltt`          | `ViTamin-B-LTT`               |                | `datacomp1b` |            |
| `vitamin_l`              | `ViTamin-L`                   |                | `datacomp1b` |            |
| `vitamin_l_256`          | `ViTamin-L-256`               |                | `datacomp1b` |            |
| `vitamin_l_336`          | `ViTamin-L-336`               |                | `datacomp1b` |            |
| `vitamin_l_384`          | `ViTamin-L-384`               |                | `datacomp1b` |            |
| `vitamin_l2`             | `ViTamin-L2`                  |                | `datacomp1b` |            |
| `vitamin_l2_384`         | `ViTamin-L2-384`              |                | `datacomp1b` |            |
| `vitamin_xl_384`         | `ViTamin-XL-384`              |                | `datacomp1b` |            |

## And Many More

More model types can be viewed via running the following at the command line or in a jupyter notebook:
```python
import open_clip
open_clip.list_pretrained()
```

# Supported Loss Types (outdated)

| Loss Type                   | Notes                                          |
|-----------------------------|------------------------------------------------|
| `infonce`                   | InfoNCE; Standard CLIP loss                    |
| `pairwise_sigmoid`          | Standard SigLIP loss                           |
| `pairwise_sigmoid_upwtdpos` | Standard SigLIP loss with upweighted positives |
| `multipos_sigmoid`          | Custom SigLIP loss (multi-positive)            |
