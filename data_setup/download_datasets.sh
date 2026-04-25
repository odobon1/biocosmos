#!/bin/bash

# TODO: get datasets

# CUB_200_2011 dataset
echo "Getting CUB_200 dataset"
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz

mkdir -p data/cub
tar -xzf CUB_200_2011.tgz -C data/cub
rm CUB_200_2011.tgz  
