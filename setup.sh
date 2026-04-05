#!/bin/bash

echo "Generating Nymphalidae metadata..."
python -m preprocessing.nymph.sids2commons
echo "sids2commons complete"
python -m preprocessing.nymph.class_data
echo "class_data complete"
python -m preprocessing.nymph.gen_rank_keys
echo "gen_rank_keys complete"
python -m preprocessing.nymph.gen_split