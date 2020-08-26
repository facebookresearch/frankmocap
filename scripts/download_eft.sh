#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.

set -ex

echo "Downloading EFT fitting data"
mkdir eft_fit
cd eft_fit

#COCO2014-All ver0.1
wget https://dl.fbaipublicfiles.com/eft/eft_fit_ver01/COCO2014-All-ver01.json

#COCO2014-Part ver0.1
wget https://dl.fbaipublicfiles.com/eft/eft_fit_ver01/COCO2014-Part-ver01.json

#LSPet ver0.1
wget https://dl.fbaipublicfiles.com/eft/eft_fit_ver01/LSPet_ver01.json

#MPII ver0.1
wget https://dl.fbaipublicfiles.com/eft/eft_fit_ver01/MPII_ver01.json

echo "Done"
