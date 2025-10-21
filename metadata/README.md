# Metadata Generation

Note: everything described in this file is executed when `./setup.sh` is run from the root as described in the setup procedure.

## Species ID Generation

**metadata/gen_species_ids.py**

To execute from root:
```
python -m metadata.gen_species_ids
```

**Requires:**
- \<secret> data on HiPerGator

**Produces:**
- `metadata/species_ids/all.pkl`
- `metadata/species_ids/known.pkl`
- `metadata/species_ids/unknown.pkl`
- `metadata/species_ids/sids2commons.pkl`

Despite the name, this file is also currently rigged up to generate the mapping from species IDs to common names (`sids2commons.pkl`).


## Generate Taxonomic structure
**metadata/gen_tax_nymph.py**

To execute from root:
```
python -m metadata.gen_tax_nymph
```
**Requires:**
- `metadata/species_ids/known.pkl`

**Produces:**
- `metadata/tax/nymph.pkl`


## Generate Data Split
**metadata/gen_split.py**

To execute from root:
```
python -m metadata.gen_split
```
**Requires:**
- `metadata/tax/nymph.pkl`

**Produces:**
- `metadata/splits/<split_name>/*`

This is configured to generate split S29-42 by default, but can be adjusted in `config/gen_split.yaml`.

Conventions used for split naming, using S29-42 as an example:
* **29** refers to 29% of the data being split out (equally distributed between ID/OOD eval/test)
* **42** refers to the seed used for split generation


## Generate Rank Keys
**metadata/gen_rank_keys.py**

To execute from root:
```
python -m metadata.gen_rank_keys
```
**Requires:**
- `metadata/tax/nymph.pkl`

**Produces:**
- `metadata/rank_keys/nymph.pkl`

Rank keys are used for hierarchical loss.
