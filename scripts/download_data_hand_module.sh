#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.

set -ex

[ -d extra_data ] || mkdir extra_data
cd extra_data

[ -d hand_module ] || mkdir hand_module
cd hand_module

echo "Downloading other data"
wget https://dl.fbaipublicfiles.com/eft/fairmocap_data/hand_module/SMPLX_HAND_INFO.pkl
wget https://dl.fbaipublicfiles.com/eft/fairmocap_data/hand_module/mean_mano_params.pkl

echo "Downloading pretrained hand model"
[ -d pretrained_weights ] || mkdir pretrained_weights
cd pretrained_weights
wget https://dl.fbaipublicfiles.com/eft/fairmocap_data/hand_module/checkpoints_best/pose_shape_best.pth

#Go to root directory
cd ../../../        

echo "Downloading sample videos"
wget https://dl.fbaipublicfiles.com/eft/sample_data_frank.tar && tar -xvf sample_data_frank.tar && rm sample_data_frank.tar
echo "Done"