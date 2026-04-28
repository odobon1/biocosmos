#!/bin/bash

echo "============================= CUB data setup =============================="
./data_setup/cub/dataset.sh
python -m data_setup.cub.split_info

echo "======================= Bryozoa metadata generation ======================="
python -m preprocessing.bryo.class_data_phylo
python -m preprocessing.bryo.rank_encs
python -m preprocessing.bryo.splits

echo "========================= CUB metadata generation ========================="
python -m preprocessing.cub.class_data
python -m preprocessing.cub.phylo
python -m preprocessing.cub.rank_encs
python -m preprocessing.cub.splits

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