All files must be executed from the top-level e.g.
`python -m metadata_o.gen_tax_gbif`


`metadata_o` data structure generation:

1. Generate Species ID's
`python -m metadata_o.gen_species_ids`
Requires:
- Nymph data
Produces:
- metadata_o/species_ids/all
- metadata_o/species_ids/known
- metadata_o/species_ids/unknown

2. Generate Taxonomic structure
`python -m metadata_o.gen_tax_nymph`
Requires:
- metadata_o/species_ids/known
Produces:
- metadata_o/tax/nymph

3. Generate Base Labels, Rank Keys, Splits
Run in any order:
`python -m metadata_o.gen_base_labels_nymph`
`python -m metadata_o.gen_rank_keys`
`python -m metadata_o.gen_splits`
Requires:
- metadata_o/tax/nymph
Produces:
- metadata_o/base_labels/nymph_sci
- metadata_o/base_labels/nymph_tax
- metadata_o/rank_keys/nymph
- metadata_o/splits/*
- figures/splits/*
