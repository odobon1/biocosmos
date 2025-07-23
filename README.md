# metadata_o Data Structure Generation

All commands must be executed from the project root e.g.

```bash
python -m metadata_o.gen_tax_gbif
```

---

## 1. Generate Species IDs

```bash
python -m metadata_o.gen_species_ids
```

**Requires:**
- Nymphalidae data

**Produces:**
- `metadata_o/species_ids/all`
- `metadata_o/species_ids/known`
- `metadata_o/species_ids/unknown`

---

## 2. Generate Taxonomic structure

```bash
python -m metadata_o.gen_tax_nymph
```

**Requires:**
- `metadata_o/species_ids/known`

**Produces:**
- `metadata_o/tax/nymph`

---

## 3. Generate Rank Keys & Splits
_Run in any order:_

```bash
python -m metadata_o.gen_rank_keys
python -m metadata_o.gen_splits  # make sure to set split parameters first
```

**Requires:**
- `metadata_o/tax/nymph`

**Produces:**
- `metadata_o/rank_keys/nymph`
- `metadata_o/splits/*`

---

## 4. Generate Data Indexes

```bash
python -m metadata_o.gen_data_indexes  # designate split first
```

**Requires:**
- `metadata_o/splits/<SPLIT_NAME>/*`

**Produces:**
- `metadata_o/data_indexes/<SPLIT_NAME>/*`

---

<br>

# Supported Model Types

| Model Type              | Current Max Batch Size (1xB200 Train w/ MP) |
|-------------------------|---------------------------------------------|
| `bioclip`               | 2048                                        |
| `bioclip2`              | 512                                         |
| `clip_vitb32`           | 2048                                        |
| `clip_vitb16`           | 1024                                        |
| `clip_vitl14`           | 512                                         |
| `clip_vitl14_336`       | 256                                         |
| `clip_rn50`             | 2048                                        |
| `clip_rn101`            | 1024                                        |
| `clip_rn101_yfcc15m`    | 1024                                        |
| `clip_rn50x4`           | 1024                                        |
| `clip_rn50x16`          | 256                                         |
| `clip_rn50x64`          | 128                                         |
| `siglip_vitb16`         | 1024                                        |
| `siglip_vitb16_384`     | -                                           |
| `siglip_vitl16_384`     | -                                           |
| `siglip_vitso400m14`    | -                                           |
| `siglip2_vitb16`        | -                                           |
| `siglip2_vitb16_384`    | -                                           |
| `siglip2_vitl16_384`    | -                                           |
| `siglip2_vitso400m14`   | -                                           |
| `siglip2_vitgopt16_384` | -                                           |
| `vitamin_s`             | -                                           |
| `vitamin_s_ltt`         | -                                           |
| `vitamin_b`             | -                                           |
| `vitamin_b_ltt`         | -                                           |
| `vitamin_l`             | -                                           |
| `vitamin_l_256`         | -                                           |
| `vitamin_l_336`         | -                                           |
| `vitamin_l_384`         | -                                           |
| `vitamin_l2`            | -                                           |
| `vitamin_l2_384`        | -                                           |
| `vitamin_xl_384`        | -                                           |

# Supported Loss Types

| Loss Type                   | Notes                                          |
|-----------------------------|------------------------------------------------|
| `infonce`                   | InfoNCE; Standard CLIP loss                    |
| `pairwise_sigmoid`          | Standard SigLIP loss                           |
| `pairwise_sigmoid_upwtdpos` | Standard SigLIP loss with upweighted positives |
| `multipos_sigmoid`          | Custom SigLIP loss (multi-positive)           |
