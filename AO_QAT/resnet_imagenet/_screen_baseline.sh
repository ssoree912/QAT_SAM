#!/bin/bash
# Launched inside a detached `screen` session so training survives any client
# session ending. Sets the env, then runs run.sh (which tees to log/.../training.txt).
source /opt/conda/etc/profile.d/conda.sh
conda activate aoq
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export USE_KD=False
export USE_QSAM=False
cd /workspace/AOQ-main/AO_QAT/resnet_imagenet
exec bash run.sh
