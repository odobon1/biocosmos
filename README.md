All files must be executed from the top-level e.g.
`python -m metadata_o.gen_tax_gbif`


`metadata_o/` data structure generation:

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

3. Generate Rank Keys & Splits
Run in any order:
`python -m metadata_o.gen_rank_keys`
`python -m metadata_o.gen_splits` (set split params first)
Requires:
- metadata_o/tax/nymph
Produces:
- metadata_o/rank_keys/nymph
- metadata_o/splits/*
- figures/splits/*

4. Generate Data Indexes
`python -m metadata_o.gen_data_indexes` (designate split first)
Requires:
- metadata_o/splits/<SPLIT_NAME>/
Produces:
- metadata_o/data_indexes/<SPLIT_NAME>/
