#!/bin/bash

# TODO: get datasets

# CUB_200_2011 dataset
echo "Getting CUB-200-2011 dataset"
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz

tar -xzf CUB_200_2011.tgz -C data/cub
rm CUB_200_2011.tgz