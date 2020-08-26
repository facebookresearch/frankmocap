#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.

git clone https://github.com/jhugestar/lightweight-human-pose-estimation.pytorch.git
cd lightweight-human-pose-estimation.pytorch

#Download pretrained model
wget https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth

