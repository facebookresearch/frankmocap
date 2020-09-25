#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.

mkdir -p detectors
cd detectors

git clone git@github.com:jhugestar/lightweight-human-pose-estimation.pytorch.git
if [ ! -d lightweight-human-pose-estimation.pytorch ]; then
    git clone https://github.com/jhugestar/lightweight-human-pose-estimation.pytorch.git
fi
mv lightweight-human-pose-estimation.pytorch body_pose_estimator

#Download pretrained model
wget https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth
mkdir -p ../extra_data/body_module/body_pose_estimator
mv *.pth ../extra_data/body_module/body_pose_estimator