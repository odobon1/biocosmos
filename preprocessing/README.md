# Nymphalidae Metadata Generation

Note: everything described in this file is executed when `./setup.sh` is run from the root as described in the setup procedure.

## Generate common names
**preprocessing/nymph/cids2commons.py**

To execute from root:
```
python -m preprocessing.nymph.cids2commons
```
**Requires:**
- Butterflies data on HiPerGator

**Produces:**
- `preprocessing/nymph/intermediaries/cids2commons.pkl`


## Generate class data
**preprocessing/nymph/class_data.py**

To execute from root:
```
python -m preprocessing.nymph.class_data
```
**Requires:**
- `preprocessing/nymph/intermediaries/cids2commons.pkl`

**Produces:**
- `metadata/nymph/class_data.pkl`


## Generate Data Split
**preprocessing/nymph/splits.py**

To execute from root:
```
python -m preprocessing.nymph.splits
```
**Requires:**
- `metadata/nymph/class_data.pkl`

**Produces:**
- `metadata/nymph/splits/<split_name>/split.pkl`
- `metadata/nymph/splits/<split_name>/figures/*`
- `metadata/nymph/splits/dev/split.pkl`
- `metadata/nymph/splits/dev/figures/*`

This is configured to generate split S29-42 by default, but can be adjusted in `config/splits.yaml`.

Conventions used for split naming, using S29-42 as an example:
* **29** refers to 29% of the data being split out (equally distributed between ID/OOD eval/test)
* **42** refers to the seed used for split generation

This script generates in-distribution (ID) and out-of-distribution (OOD) stratified splits for train/validation/test.
In addition to the primary split, it also generates a default `dev` split where every partition mirrors the primary partition keys and contains the first `size_dev` samples from the primary train partition.

First, entire species are split out for OOD zero-shot evaluation. Selection is stratified by genus and tuned to hit a target OOD proportion by number of species and total samples within a specified tolerance. Next, ID eval splits are sampled from the remaining pool. Singleton classes are temporarily excluded so they don't leak pseudo-OOD examples into ID eval (selection of such samples for ID eval would effectively render them OOD). Samples corresponding to non-singleton species are split into ID val/test sets, stratified by species. The remainder of the samples are joined with the singletons to produce the train set.

![System diagram](../images/strat_process.png)

The sklearn stratified splitter (train_test_split) errors out if the number of classes exceeds the number of test samples, which was the case for the OOD splits. Also, in the ID case, because our dataset is extremely long-tailed, with many classes that have very few instances, the sklearn stratified split yields a significant number of species that are completely unsampled from. To address these problems, each stratified split is performed in 2 stages:
* Sparse Stratified Split (custom sampling method)
* Standard Stratified Split (sklearn)

For the OOD splits, classes are genera and instances are species. For the ID splits, classes are species and instances are samples. Stratification process: Overall, the classes are first grouped by their instance-counts (1-instance classes, 2-instance classes, etc.). The sparse stratified splitter algorithm then samples from the classes with the fewest amount of instances in a way that abides by the target split percentage. For example, if the target OOD split percentage is 10%, the sparse splitter will first iterate through classes containing 1 instance and select 1 class for val/test from every ten 1-instance classes. It will then iterate through 2-instance classes and select 1 instance for val/test from every five 2-instance classes. This pattern continues until n (for n-instances) becomes high enough that the sklearn splitter behaves as desired, at which point the sklearn stratified splitter is used for the remainder of the larger classes.
Note: The ID split begins at 2-instance classes (2-sample species) because the singletons are first removed for reasons previously described.
Note: The first stage of stratification (sparse) is done to ensure a well-distributed split among classes with fewer instances. By default, the sklearn stratified splitter (and all the other numerous stratified splitters I tried) will leave these unassigned, effectively leaving the tail end of the distribution completely unsampled from. In other words, the standard sklearn splitting process biases evaluation away from measuring few-shot learning.

Datastructures are created for tracking n-shot subsets of ID splits for monitoring robustness to class imbalance and the assessing the effectiveness of class imbalance methods utilized. Stats regarding the variety (num. species) and volume (num. samples) of n-shot ID eval subsets are produced to get an idea of the statistical significance of the different n-shot subsets (subsets with lower volume are less significant) and are used to adjust the bounds of the n-shot buckets such that they are more statistically significant.

## Generate Phylogenetic Tree
**preprocessing/nymph/phylo.py**

To execute from root:
```
python -m preprocessing.nymph.phylo
```
**Requires:**
- Nymphalidae phylogenetic tree data (`paths["nymph_tree_raw"]`)
- `metadata/nymph/class_data.pkl`

**Produces:**
- `metadata/nymph/tree.pkl`

## Generate Rank Encodings
**preprocessing/nymph/rank_encs.py**

To execute from root:
```
python -m preprocessing.nymph.rank_encs
```
**Requires:**
- `metadata/nymph/class_data.pkl`

**Produces:**
- `metadata/nymph/rank_encs.pkl`

Rank keys are used for generating intermediate targets for use with hierarchical loss. Future work involves experimentation with phylogenetic distance metrics to provide a higher fidelity learning signal.


# Lepidoptera Metadata Generation

## Generate common names
**preprocessing/lepid/cids2commons.py**

To execute from root:
```
python -m preprocessing.lepid.cids2commons
```
**Requires:**
- Lepidoptera image data on HiPerGator

**Produces:**
- `preprocessing/lepid/intermediaries/cids2commons.pkl`


## Generate class data
**preprocessing/lepid/class_data.py**

To execute from root:
```
python -m preprocessing.lepid.class_data
```
**Requires:**
- `preprocessing/lepid/intermediaries/cids2commons.pkl`
- `paths["lepid_metadata_tax"]` CSV (taxonomic metadata)

**Produces:**
- `metadata/lepid/class_data.pkl`


## Generate Phylogenetic Tree
**preprocessing/lepid/phylo.py**

To execute from root:
```
python -m preprocessing.lepid.phylo
```
**Requires:**
- Lepidoptera phylogenetic tree data (`paths["lepid_tree_raw"]`)
- `metadata/nymph/tree.pkl` (Nymphalidae tree, for merging)
- `metadata/lepid/class_data.pkl`

**Produces:**
- `metadata/lepid/tree.pkl`

The Lepid tree is used as the global backbone. The Nymphalidae subtree is merged into it, preserving the full Nymph tree exactly without branch-length scaling, then attaching it at the appropriate anchor point on the Lepid Nymphalidae path.


## Generate Data Split
**preprocessing/lepid/splits.py**

To execute from root:
```
python -m preprocessing.lepid.splits
```
**Requires:**
- `metadata/lepid/class_data.pkl`
- `metadata/lepid/tree.pkl`

**Produces:**
- `metadata/lepid/splits/<split_name>/split.pkl`
- `metadata/lepid/splits/<split_name>/figures/*`
- `metadata/lepid/splits/dev/split.pkl`
- `metadata/lepid/splits/dev/figures/*`

This is configured via `config/splits.yaml`. Lepid now follows the same standard split setup as Nymph: ID plus OOD-species partitions for validation/test (no separate OOD-family or OOD-genus partitions).


## Generate Rank Encodings
**preprocessing/lepid/rank_encs.py**

To execute from root:
```
python -m preprocessing.lepid.rank_encs
```
**Requires:**
- `metadata/lepid/class_data.pkl`

**Produces:**
- `metadata/lepid/rank_encs.pkl`

Encodes ranks: `family`, `genus`, `species`.


# Bryozoa Metadata Generation

## Generate class data and phylogenetic tree
**preprocessing/bryo/class_data_phylo.py**

Note: Class data and phylo tree generation are inherently coupled for this one. Scientific names are needed to grab taxonomic info from GBIF used to construct class data. Image directories do not provide scientific names, only genus names. Using genus names of image directories, scientific names are harvested from phylo tree and used to query GBIF for taxonomic info. Tree tip names are then converted from sci-name to genus and all other species in that genus pruned.

To execute from root:
```
python -m preprocessing.bryo.class_data_phylo
```
**Requires:**
- Bryozoa image data on HiPerGator (genus-level subdirectories)
- `metadata/bryo/SI_Fig1(BIG).newick` (Bryozoa phylogenetic tree)
- Internet access (GBIF API used to resolve taxonomic info per genus)

**Produces:**
- `metadata/bryo/class_data.pkl`
- `metadata/bryo/tree.pkl`

Class data and the phylogenetic tree are generated together because the tree is used to identify which genera have resolvable taxonomy. The newick tree is pruned to one tip per genus (removing duplicates, unknowns, and taxonomically conflicted entries), genus names are normalized to lowercase, and GBIF is queried to resolve family-level taxonomy. The tree is then further pruned to retain only genera present in both the image data and the resolved class data.


## Generate Data Split
**preprocessing/bryo/splits.py**

To execute from root:
```
python -m preprocessing.bryo.splits
```
**Requires:**
- `metadata/bryo/class_data.pkl`
- Bryozoa image data on HiPerGator

**Produces:**
- `metadata/bryo/splits/<split_name>/split.pkl`
- `metadata/bryo/splits/<split_name>/figures/*`
- `metadata/bryo/splits/dev/split.pkl`
- `metadata/bryo/splits/dev/figures/*`

OOD genera are defined as genera present in the image data but absent from `class_data` (i.e. genera without resolved taxonomy). ID/OOD splits are otherwise generated using the same sparse + standard stratified splitting approach as the other datasets.


## Generate Rank Encodings
**preprocessing/bryo/rank_encs.py**

To execute from root:
```
python -m preprocessing.bryo.rank_encs
```
**Requires:**
- `metadata/bryo/class_data.pkl`

**Produces:**
- `metadata/bryo/rank_encs.pkl`

Encodes ranks: `family`, `genus`.