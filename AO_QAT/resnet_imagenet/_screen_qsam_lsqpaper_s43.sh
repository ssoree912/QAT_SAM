#!/bin/bash
# qSAM on the LSQ-paper recipe (no-KD), SEED=43 -- reproducibility check of the
# +0.09 qSAM effect seen at seed 42. Separate checkpoints via _seed43 tag.
source /opt/conda/etc/profile.d/conda.sh
conda activate aoq
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU:-2}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export USE_KD=False
export USE_QSAM=True
export OPTIMIZER=sgd
export LR_SCHED=cosine
export LR=0.01
export EPOCHS=90
export WD=1e-4
export FIRST_LAST_BIT=8
export QSAM_WARMUP=0
export QSAM_RHO=1.0
export QSAM_RATIO=0.001
export SEED=43
export SAVE=./models_lsqpaper
cd /workspace/AOQ-main/AO_QAT/resnet_imagenet
exec bash run.sh
