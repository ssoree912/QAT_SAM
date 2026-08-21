#!/bin/bash
# LSQ-paper recipe + single-step quantized SAM (S2-SAM efficiency, RA-qSAM p=2).
# Same hyperparameters as _screen_lsq_paper.sh (SGD mom0.9, cosine, lr0.01,
# 90ep, wd1e-4, first/last 8-bit, NO KD) so the ONLY difference vs the baseline
# is qSAM -> a clean baseline-vs-qSAM comparison.
#   qSAM knobs: ratio 0.001 (K/d), rho 1.0, warmup 0 (on from epoch 0).
# Runs in a detached screen so it survives client sessions.
source /opt/conda/etc/profile.d/conda.sh
conda activate aoq
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU:-2}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
# LSQ-paper recipe (identical to baseline)
export USE_KD=False
export OPTIMIZER=sgd
export LR_SCHED=cosine
export LR=0.01
export EPOCHS=90
export WD=1e-4
export FIRST_LAST_BIT=8
# qSAM ON
export USE_QSAM=True
export QSAM_RATIO=0.001
export QSAM_RHO=1.0
export QSAM_WARMUP=0
export SAVE=./models_lsqpaper_qsam
cd /workspace/AOQ-main/AO_QAT/resnet_imagenet
exec bash run.sh
