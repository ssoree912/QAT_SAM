#!/bin/bash
# CIFAR-100 ResNet-50 4-bit run with the CURRENT (fixed) code.
# Usage: METHOD=lsq|lsq_qsam GPU=0 SEED=42 bash _screen_cifar.sh
source /opt/conda/etc/profile.d/conda.sh
conda activate aoq
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
METHOD=${METHOD:-lsq}
SEED=${SEED:-42}
EPOCHS=${EPOCHS:-200}
cd /workspace/AOQ-main/AO_QAT/resnet_cifar100
TAG=${METHOD}_resnet50_4bit_seed${SEED}
mkdir -p log/${TAG}
exec python3 train.py \
    --method=${METHOD} \
    --model=resnet50 \
    --n_bit=4 \
    --epochs=${EPOCHS} \
    --seed=${SEED} \
    --data=./data \
    2>&1 | tee -a log/${TAG}/training.txt
