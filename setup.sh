#!/bin/bash

python -m metadata.gen_species_ids
echo "gen_species_ids complete"
python -m metadata.gen_tax_nymph
echo "gen_tax_nymph complete"
python -m metadata.gen_rank_keys
echo "gen_rank_keys complete"
python -m metadata.gen_split
