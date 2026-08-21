#!/bin/bash

echo "Getting CUB-200-2011 dataset"
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz

tar -xzf CUB_200_2011.tgz -C data/cub
rm CUB_200_2011.tgz

echo "Getting Jetz et al. (birdtree.org) Hackett Stage2 posterior tree sample"
wget https://data.vertlife.org/birdtree/Stage2/HackettStage2_0001_1000.zip

unzip -o -j HackettStage2_0001_1000.zip -d data/cub
rm HackettStage2_0001_1000.zip
