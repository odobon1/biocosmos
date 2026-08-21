# Metadata Generation

Preprocessing turns raw image directories, taxonomy CSVs, and phylogenetic trees
into the per-dataset metadata artifacts consumed by training: `class_data.pkl`,
`tree.pkl`, `rank_encs.pkl`, and the split bundles under `splits/`.

The pipeline runs in five steps, in dependency order:

```
cids2commons  →  class_data  →  phylo  →  rank_encs  →  split_gen
```

(Lepidoptera runs an extra [synonyms](#synonyms) step between `class_data` and
`phylo`.)

Each step is implemented in `preprocessing/common/` and invoked by a thin
per-dataset entry point under `preprocessing/<dataset>/` that supplies
dataset-specific parameters and helpers (e.g. `species_ids.py`,
`split_gen_utils.py`). The sections below are organized by step; dataset-specific
nuances are called out within each section.

## Running

All metadata artifacts are committed to the repo, so the pipeline does not need to
be run for normal use. To regenerate everything from scratch, run the full pipeline
end-to-end for all four datasets (in dependency order) with:
```
./setup.sh
```
To regenerate a single step on its own, run the `python -m ...` command listed in
that step's section below.

All commands below are run from the repo root.


# cids2commons

Resolves a common (vernacular) name for each class via the GBIF API.

**Shared:** `preprocessing/common/cid2commons.py` (`build_cids2commons`)

`build_cids2commons` matches each scientific name with GBIF
(`species/match` → `species/{key}/vernacularNames`) and keeps the first English
vernacular name.

**Only `nymph` and `lepid` run this step.** CUB resolves common names inside its
`class_data` step (from the dataset's `.mat` file + GBIF), and Bryozoa stores
`common_name = None` — common names pertain to species, but Bryozoa's leaf-level
classes are at the genus level.

### Nymphalidae
```
python -m preprocessing.nymph.cids2commons
```
- **Requires:** Nymphalidae image data on HiPerGator
- **Produces:** `preprocessing/nymph/intermediaries/cids2commons.pkl`

Class IDs come from listing the image directory and filtering to well-formed
alphabetic species names (`species_ids.get_cids_nymph`), with a few hyphenated
exceptions (e.g. `polygonia_c-album`).

### Lepidoptera
```
python -m preprocessing.lepid.cids2commons
```
- **Requires:** Lepidoptera image data on HiPerGator
- **Produces:** `preprocessing/lepid/intermediaries/cids2commons.pkl`

Class IDs come from per-family image subdirectories (`species_ids.get_cids_lepid`),
with subspecies names truncated to species level, then flattened across families.


# class_data

Builds `class_data.pkl`: a mapping `cid -> {rank: value, ..., "common_name": ...}`
that is the source of truth for class taxonomy throughout preprocessing and
training. Available taxonomic ranks vary by dataset.

> Bryozoa generates `class_data.pkl` jointly with its tree — see the
> [phylo](#phylo) section.

### Nymphalidae
```
python -m preprocessing.nymph.class_data
```
- **Requires:** `preprocessing/nymph/intermediaries/cids2commons.pkl`; Nymphalidae
  taxonomy CSV (`/lustre/blue2/arthur.porto/data/datasets/nymphalidae_whole_specimen-v250613/metadata/data_meta-nymphalidae_whole_specimen-v250613.csv`)
- **Produces:** `metadata/nymph/class_data.pkl`

Ranks: `subfamily`, `genus`, `species`, `common_name`. Genus is parsed from the
class ID; subfamily comes from the taxonomy CSV (the placeholder `"moth"` →
`None`).

### Lepidoptera
```
python -m preprocessing.lepid.class_data
```
- **Requires:** `preprocessing/lepid/intermediaries/cids2commons.pkl`; Lepidoptera
  taxonomy CSV (`/lustre/blue2/arthur.porto/data/datasets/butterflies_whole_specimen-clean_rot_512-v2025_05_07/metadata/data_tree_meta.csv`)
- **Produces:** `metadata/lepid/class_data.pkl`

Ranks: `family`, `subfamily`, `tribe`, `genus`, `species`, `common_name`. Family
comes from the image subdirectory; subfamily/tribe are looked up per genus from
the taxonomy CSV.

### CUB
```
python -m preprocessing.cub.class_data
```
- **Requires:** CUB `att_splits.mat` (`data/cub/xlsa17/data/CUB/`);
  internet access (GBIF API)
- **Produces:** `metadata/cub/class_data.pkl`

Ranks: `order`, `family`, `genus`, `species`, `common_name`. CUB has no separate
`cids2commons` step: common names are parsed from the class names in the `.mat`
file, corrected via a small manual map (`COMMON_NAME_CORRECTIONS`) for names GBIF
can't resolve or resolves to the wrong species (ambiguous vernaculars like
"Nighthawk"), then queried against GBIF (iNaturalist backbone first, falling back
to the whole general-backbone result if the backbone hit is unusable — results
are never merged field-by-field) to resolve `order`/`family`/`genus`/`species`.
Bird hits (`class == "Aves"`) are preferred and extinct taxa are dropped.


# synonyms

**Only `lepid` runs this step.** Builds
`preprocessing/lepid/intermediaries/synonyms.pkl`: `{"rename": {cid: tip},
"sister": {cid: host_cid}}` maps consumed by the phylo step.

```
python -m preprocessing.lepid.synonyms
```
- **Requires:** both raw trees (`data/lepid/tree_renamed_full.tre`,
  `data/nymph/tree_nymphalidae_chazot2021_all.tree`); `metadata/lepid/class_data.pkl`;
  Lepidoptera taxonomy CSV; internet access (GBIF API)
- **Produces:** `preprocessing/lepid/intermediaries/synonyms.pkl`

The image directories and the raw trees sometimes use different generic
combinations for the same species (e.g. class `chilasa_clytia` is on the lepid
tree as `papilio_clytia`). For each class absent from both raw trees, the class's
GBIF accepted name and synonyms are searched among the raw tree tips. Hits go into
two maps consumed by the phylo step: `"rename"` (the tip is not a class — it gets
renamed to the class id, giving the class its real placement) and `"sister"` (the
tip is another class, i.e. the dataset holds duplicate classes for one species —
the class is grafted as a zero-length sister of that host tip). Guards: a hit is
rejected when the tip name's own GBIF resolution points at a different species
than the class resolves to (one-way chains toward a separately-accepted species
are almost always GBIF backbone errors); tips found only on the nymph tree are
usable only for nymphalid classes (the merge attaches the whole nymph tree at
the Nymphalidae anchor, so its outgroup tips carry no usable placement for
other families); mappings that contradict the tax CSV's family for the tip's
genus are dropped; and when one tip's claimants resolve to different species,
the tip's own epithet arbitrates. Classes that then share one rename tip are
conspecifics of one another: the first takes the tip, the rest join it as
zero-length sisters. `class_data` and the CSV are untouched — ranks keep coming
from the CSV only.


# phylo

Builds `tree.pkl`: a `Bio.Phylo` tree whose terminals are class IDs, used to
derive phylogenetic learning signal. Classes present in `class_data` but missing
from the raw tree are grafted on. Each pipeline also saves `tree_prepoly.pkl`
alongside it: the trained-on tree minus the polytomy grafts (real — and for
Lepid, conspecific-sister — placements only, pruned to `class_data`), kept for
visualization.

**Shared:** `preprocessing/common/phylo.py`
- `augment_tree_with_polytomies` — inserts each missing class as a polytomy, preferring
  its genus' divergence anchor (the most recent divergence between a represented
  congener — or same-higher-rank representative — and its nearest represented
  sibling-taxon tip); classes with no usable anchor fall back to a lineage search that
  grafts at the deepest MRCA between their most specific represented taxon and a
  represented sibling taxon. Every inserted tip is extended to the mean pre-graft tip
  depth of the clade it joins.
- `prune_tree` — drops tips not in `class_data`.
- `augment_class_data` — infers taxonomy for tree tips absent from `class_data`
  using a same-genus representative (skipping genera with conflicting ranks).
  Used only for butterfly datasets preprocessing (`nymph`, `lepid`).

### Nymphalidae
```
python -m preprocessing.nymph.phylo
```
- **Requires:** Nymphalidae raw tree (`data/nymph/tree_nymphalidae_chazot2021_all.tree`);
  `metadata/nymph/class_data.pkl`
- **Produces:** `metadata/nymph/tree.pkl`, `metadata/nymph/tree_prepoly.pkl`

The raw newick is read and a few stray tips pruned. `class_data` is augmented from
the tree, the tree is pruned to that augmented set, missing classes are added as
polytomies, then the tree is pruned back to the original `class_data`.

### Lepidoptera
```
python -m preprocessing.lepid.phylo
```
- **Requires:** Lepidoptera raw tree (`data/lepid/tree_renamed_full.tre`); the
  Nymphalidae tree (rebuilt from the raw nymph tree); `metadata/lepid/class_data.pkl`;
  `preprocessing/lepid/intermediaries/synonyms.pkl`
- **Produces:** `metadata/lepid/tree.pkl`, `metadata/lepid/tree_prepoly.pkl`

The Lepid tree is the global backbone. Subspecies tips are truncated to species
level. Tips in the synonym map's `"rename"` half are renamed to their class ids, so
those classes take their real tree placements instead of being polytomy-grafted;
after the merge, classes in its `"sister"` half are grafted as zero-length sisters
of their conspecific host tips. The intact Nymphalidae tree is merged in
(`combine_trees_lepid_nymph`)
without branch-length scaling: the shared Lepid Nymphalidae tips are pruned and
the Nymph subtree is attached at the lowest ancestor on the Lepid Nymphalidae path
that keeps terminal depth ultrametric. Lepid-only tips and non-Nymphalidae genera
sourced from the Nymph tree are then rehomed onto their nearest retained anchor.
The merged tree is augmented with polytomies and pruned, as above.

### CUB
```
python -m preprocessing.cub.phylo
```
- **Requires:** CUB raw tree (`data/cub/1_tree-consensus-Hacket-AllSpecies-modified_cub-names_v1.phy`);
  Jetz et al. posterior sample (`data/cub/AllBirdsHackett1.tre`, fetched by
  `data_setup/cub/dataset.sh`); `metadata/cub/class_data.pkl`; internet access
  (GBIF API, only when the Jetz cache needs rebuilding)
- **Produces:** `metadata/cub/tree.pkl`, `metadata/cub/tree_prepoly.pkl`,
  `preprocessing/cub/intermediaries/jetz_cache.pkl.gz`

The consensus backbone's tip names are normalized to class ID formatting (strip the
leading prefix, lowercase, spaces → underscores). The backbone covers 190 of the 200
classes; the 10 missing species are filled in from the Jetz et al. (birdtree.org)
Hackett Stage2 posterior sample: each is attached as sister to its majority-rule
attachment clade across the 1000 posterior trees, at the median attachment age
(class IDs are bridged to Jetz's 2012-era tip names via GBIF synonyms). Backbone
placements and branch lengths are untouched. Any class that still can't be placed
falls back to polytomy grafting, then the tree is pruned to `class_data`.

The GBIF cid→Jetz mapping and the pruned posterior sample are cached together in
`preprocessing/cub/intermediaries/jetz_cache.pkl.gz` (git-tracked, ~3.5 MB) and
rebuilt automatically when the class set changes, so re-runs skip both the GBIF
calls and the 464 MB posterior parse. Delete the file to force a rebuild.

### Bryozoa (class_data + phylo, combined)
```
python -m preprocessing.bryo.class_data_phylo
```
- **Requires:** Bryozoa image data on HiPerGator (genus-level subdirectories);
  Bryozoa raw tree (`data/bryo/SI_Fig1(BIG).newick`); internet access (GBIF API)
- **Produces:** `metadata/bryo/class_data.pkl`, `metadata/bryo/tree.pkl`,
  `metadata/bryo/tree_prepoly.pkl`

Class data and the tree are generated together because they are mutually
dependent: image directories provide only genus names, while scientific names
(needed to resolve taxonomy from GBIF) come from the tree. The newick is pruned to
one tip per genus (dropping duplicates, `UNKNOWN`s, and taxonomically conflicted
tips) and tip names lowercased. For each genus present in the image data, a
species name from the tree is used to query GBIF for family-level taxonomy (a
small `MISSING_GENUS_2_FAMILY` map covers genera GBIF can't resolve). The tree is
then pruned/renamed to one tip per genus retained in `class_data`, and missing
classes are added as polytomies. Bryozoa classes are **genera**.


# rank_encs

Builds `rank_encs.pkl`: per-rank `bidict`s mapping each rank value to a contiguous
integer key, used to generate intermediate hierarchical targets during training when discrete
hierarchical-taxonomic loss is used.

**Shared:** `preprocessing/common/rank_encs.py` (`build_rank_encs`)

Encodings are deterministic: `species` keys follow the sorted class IDs, and every
other rank's keys follow its sorted distinct values. Each dataset's entry point
just declares which ranks to encode.

| dataset | command                                   | ranks                              |
|---------|-------------------------------------------|------------------------------------|
| `nymph` | `python -m preprocessing.nymph.rank_encs` | `genus`, `species`                 |
| `lepid` | `python -m preprocessing.lepid.rank_encs` | `family`, `genus`, `species`       |
| `cub`   | `python -m preprocessing.cub.rank_encs`   | `order`, `family`, `genus`, `species` |
| `bryo`  | `python -m preprocessing.bryo.rank_encs`  | `family`, `genus`                  |

- **Requires:** `metadata/<dataset>/class_data.pkl`
- **Produces:** `metadata/<dataset>/rank_encs.pkl`


# split_gen

Generates the train/validation/test split bundle for each dataset, with both
in-distribution (ID) and out-of-distribution (OOD) evaluation partitions.

**Shared:** `preprocessing/common/split_gen.py` (`GenSplitDataManager`,
stratified samplers, `generate_splits`)

**Config:** `config/split_gen.yaml`
- `split` — label for the output directory (default `D10`)
- `seed` — RNG seed
- `pct_partition` — fraction of classes/samples drawn into each eval partition
- `pct_ood_tol` — tolerance on OOD sample volume vs. target; the OOD draw is a
  random search over seeds until within tolerance (too small ⇒ never terminates)
- `size_dev` — partition size of the `dev` split
- `nst_names` / `nst_seps` — n-shot bucket names and separators
- `pos_filter` — `{null, dorsal}`; only affects `nymph` and `lepid`
- `dev.debug_mode` — when true, applies `dev.debug` overrides

Each dataset is invoked the same way and produces the same outputs:
```
python -m preprocessing.<dataset>.split_gen
```
- **Requires:** `metadata/<dataset>/class_data.pkl`; dataset image data on HiPerGator
- **Produces:**
  - `metadata/<dataset>/splits/<split>/split.pkl` + `figures/*`
  - `metadata/<dataset>/splits/dev/split.pkl` + `figures/*`

A `split.pkl` bundles the per-partition data indexes (`train`, `val.id`,
`val.ood`, `trainval`, `test.id`, `test.ood`), the class-ID↔encoding map, n-shot
tracking buckets, and per-class counts (train and trainval). The `dev` split
mirrors the partition keys and takes the first `size_dev` samples of each.

### Stratified splitting

For ID/OOD partitions that aren't predefined by the dataset, each partition is
drawn with a two-stage stratified sampler:

1. **Sparse stratified split** (custom). Classes are grouped by instance count.
   The sampler draws proportionally from the smallest groups first (e.g. 1 of
   every ten 1-instance classes at a 10% target), ensuring the long tail is
   represented. Without this, standard stratified splitters leave low-count
   classes completely unsampled, biasing evaluation away from few-shot
   performance.
2. **Standard stratified split** (sklearn `train_test_split`) for the remaining
   larger classes, where it behaves well.

Within a single partition draw, OOD is drawn first (whole penultimate-rank groups
held out as unseen classes), then ID (samples held out from seen classes;
singletons are excluded from the ID draw so they don't leak into eval, which would
effectively render them OOD samples, then rejoined into train). For datasets that
draw their own test partition, this whole procedure runs twice — once for test,
once for val — so the overall sampling order is:

1. OOD test
2. ID test
3. OOD validation
4. ID validation

Each draw partitions off a fraction of what remains, so earlier draws take
priority: test is carved out before validation, and unseen (OOD) classes before
held-out (ID) samples. The exception is CUB, whose ID/OOD test partitions are
predefined and well-established in the relevant literature (taken from
`att_splits.mat`); for CUB, only the validation partitions are sampled (steps 3–4).

n-shot buckets track how many classes fall into each shot range (for monitoring
robustness to class imbalance), and summary/distribution figures are written
alongside each split. Two bucket sets are built, each pairing a shot-count
partition with the ID eval partition it scores:

- **`train/val`** — ID-`val` classes, bucketed by their **`train`**-partition
  sample cardinality.
- **`trainval/test`** — ID-`test` classes, bucketed by their **`trainval`**-partition
  sample cardinality.

### Dataset nuances

- **Nymphalidae / Lepidoptera** — penultimate rank is `genus`. Image pointers are
  built from the image directories and honor `pos_filter` (e.g. `dorsal`-only).
  Per-sample metadata (`pos`, `sex`) is attached from the images CSV. Both test
  and val partitions are sampled.
- **CUB** — penultimate rank is `genus`. The ID/OOD **test** partitions are taken
  from CUB's predefined `att_splits.mat` (`test_seen` / `test_unseen`), so only
  the validation partition is sampled (with `ood_tol_flag=False`).
- **Bryozoa** — classes are genera; penultimate rank is `family`. ID/OOD test and
  val partitions are both sampled with the standard stratified procedure.
