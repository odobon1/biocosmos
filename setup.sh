#!/bin/bash

echo "===================== Nymphalidae metadata generation ====================="
python -m preprocessing.nymph.sids2commons
python -m preprocessing.nymph.class_data
python -m preprocessing.nymph.phylo
python -m preprocessing.nymph.gen_rank_keys
python -m preprocessing.nymph.gen_split

echo "===================== Lepidoptera metadata generation ====================="
python -m preprocessing.lepid.sids2commons
python -m preprocessing.lepid.class_data
python -m preprocessing.lepid.phylo