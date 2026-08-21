#!/bin/bash
# qSAM run (single-step S2-SAM + RA-qSAM p=2), identical harness to the baseline.
# Detached `screen` so it survives any client session ending.
source /opt/conda/etc/profile.d/conda.sh
conda activate aoq
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export USE_KD=False
export USE_QSAM=True
cd /workspace/AOQ-main/AO_QAT/resnet_imagenet
exec bash run.sh
