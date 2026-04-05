#!/bin/bash

echo "======================================"
echo "Generating Nymphalidae metadata..."
echo ""
python -m preprocessing.nymph.sids2commons
echo "sids2commons complete"
python -m preprocessing.nymph.class_data
echo "class_data complete"
python -m preprocessing.nymph.phylo
echo "phylo complete"
python -m preprocessing.nymph.gen_rank_keys
echo "gen_rank_keys complete"
python -m preprocessing.nymph.gen_split
echo "gen_split complete"
echo ""

echo "======================================"
echo "Generating Lepidoptera metadata..."
echo ""
# python -m preprocessing.lepid.sids2commons
echo "sids2commons complete"
python -m preprocessing.lepid.class_data
echo "class_data complete"
python -m preprocessing.lepid.phylo
echo "phylo complete"