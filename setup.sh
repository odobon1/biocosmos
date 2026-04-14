#!/bin/bash

echo "===================== Nymphalidae metadata generation ====================="
python -m preprocessing.nymph.sids2commons
python -m preprocessing.nymph.class_data
python -m preprocessing.nymph.phylo
python -m preprocessing.nymph.rank_encs
python -m preprocessing.nymph.splits

echo "===================== Lepidoptera metadata generation ====================="
python -m preprocessing.lepid.sids2commons
python -m preprocessing.lepid.class_data
python -m preprocessing.lepid.phylo
python -m preprocessing.lepid.rank_encs
python -m preprocessing.lepid.splits