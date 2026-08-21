#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate aoq
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export USE_KD=False
export USE_QSAM=True
export QSAM_WARMUP=0
cd /workspace/AOQ-main/AO_QAT/resnet_imagenet
exec bash run.sh
