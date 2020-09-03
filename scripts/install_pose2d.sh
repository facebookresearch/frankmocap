#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.

mkdir -p detec
cd detectors
git clone https://github.com/jhugestar/lightweight-human-pose-estimation.pytorch.git
mv lightweight-human-pose-estimation.pytorch body_pose_estimator

#Download pretrained model
wget https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth
mkdir -p ../data/weights/body_pose_estimator
mv *.pth ../data/weights/body_pose_estimator