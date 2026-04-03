#!/bin/bash

echo "Generating Nymphalidae metadata..."
python -m preprocessing.nymph.gen_sids2commons
echo "gen_sids2commons.py complete"
python -m preprocessing.nymph.gen_class_data
echo "gen_class_data.py complete"
python -m preprocessing.nymph.gen_rank_keys
echo "gen_rank_keys.py complete"
python -m preprocessing.nymph.gen_split