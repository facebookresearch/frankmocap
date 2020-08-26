#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.

set -ex


[ -d extradata ] || mkdir extradata
cd extradata

echo "Downloading extra data from SPIN"
wget http://visiondata.cis.upenn.edu/spin/data.tar.gz && tar -xvf data.tar.gz && rm data.tar.gz
mv data data_from_spin
cd ..

echo "Downloading pretrained model"
[ -d models_eft ] || mkdir models_eft
cd models_eft
wget dl.fbaipublicfiles.com/eft/2020_05_31-00_50_43-best-51.749683916568756.pt
cd ..

echo "Downloading sample videos"
wget s3://dl.fbaipublicfiles.com/eft/sampledata.tar && tar -xvf sampledata.tar && rm sampledata.tar

echo "Done"
