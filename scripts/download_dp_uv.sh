#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# Script from Densepose repo: https://github.com/facebookresearch/DensePose/blob/master/DensePoseData/get_densepose_uv.sh

cd extradata
mkdir densepose_uv_data
cd densepose_uv_data
wget https://dl.fbaipublicfiles.com/densepose/densepose_uv_data.tar.gz
tar xvf densepose_uv_data.tar.gz
rm densepose_uv_data.tar.gz