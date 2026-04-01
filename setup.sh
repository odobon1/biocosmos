#!/bin/bash

echo "Generating Nymphalidae metadata..."
python -m preprocessing.nymph.gen_species_ids
echo "gen_species_ids complete"
python -m preprocessing.nymph.gen_tax
echo "gen_tax complete"
python -m preprocessing.nymph.gen_rank_keys
echo "gen_rank_keys complete"
python -m preprocessing.nymph.gen_split