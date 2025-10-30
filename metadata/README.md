# Metadata Generation

Note: everything described in this file is executed when `./setup.sh` is run from the root as described in the setup procedure.

## Species ID Generation

**metadata/gen_species_ids.py**

To execute from root:
```
python -m metadata.gen_species_ids
```

**Requires:**
- Butterflies data on HiPerGator

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

This script generates in-distribution (ID) and out-of-distribution (OOD) stratified splits for train/validation/test.

First, entire species are split out for OOD zero-shot evaluation. Selection is stratified by genus and tuned to hit a target OOD proportion by number of species and total samples within a specified tolerance. Next, ID eval splits are sampled from the remaining pool. Singleton classes are temporarily excluded so they don't leak pseudo-OOD examples into ID eval (selection of such samples for ID eval would effectively render them OOD). Samples corresponding to non-singleton species are split into ID val/test sets, stratified by species. The remainder of the samples are joined with the singletons to produce the train set.

![System diagram](../images/strat_process.png)

The sklearn stratified splitter (train_test_split) errors out if the number of classes exceeds the number of test samples, which was the case for the OOD splits. Also, in the ID case, because our dataset is extremely long-tailed, with many classes that have very few instances, the sklearn stratified split yields a significant number of species that are completely unsampled from. To address these problems, each stratified split is performed in 2 stages:
* Sparse Stratified Split (custom sampling method)
* Standard Stratified Split (sklearn)

For the OOD splits, classes are genera and instances are species. For the ID splits, classes are species and instances are samples. Stratification process: Overall, the classes are first grouped by their instance-counts (1-instance classes, 2-instance classes, etc.). The sparse stratified splitter algorithm then samples from the classes with the fewest amount of instances in a way that abides by the target split percentage. For example, if the target OOD split percentage is 10%, the sparse splitter will first iterate through classes containing 1 instance and select 1 class for val/test from every ten 1-instance classes. It will then iterate through 2-instance classes and select 1 instance for val/test from every five 2-instance classes. This pattern continues until n (for n-instances) becomes high enough that the sklearn splitter behaves as desired, at which point the sklearn stratified splitter is used for the remainder of the larger classes.
Note: The ID split begins at 2-instance classes (2-sample species) because the singletons are first removed for reasons previously described.
Note: The first stage of stratification (sparse) is done to ensure a well-distributed split among classes with fewer instances. By default, the sklearn stratified splitter (and all the other numerous stratified splitters I tried) will leave these unassigned, effectively leaving the tail end of the distribution completely unsampled from. In other words, the standard sklearn splitting process biases evaluation away from measuring few-shot learning.

Datastructures are created for tracking n-shot subsets of ID splits for monitoring robustness to class imbalance and the assessing the effectiveness of class imbalance methods utilized. Stats regarding the variety (num. species) and volume (num. samples) of n-shot ID eval subsets are produced to get an idea of the statistical significance of the different n-shot subsets (subsets with lower volume are less significant) and are used to adjust the bounds of the n-shot buckets such that they are more statistically significant.

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

Rank keys are used for generating intermediate targets for use with hierarchical loss. Future work involves experimentation with phylogenetic distance metrics to provide a higher fidelity learning signal.
