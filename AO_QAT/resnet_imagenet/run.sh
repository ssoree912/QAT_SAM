#!/bin/bash
# Train a 4-bit ResNet-50 on ImageNet with the LSQ baseline, optionally adding
# single-step quantized SAM (S2-SAM efficiency + RA-qSAM p=2 geometry).
# Usage:
#   bash run.sh                              # LSQ baseline: resnet50, 4-bit
#   USE_QSAM=True bash run.sh                # LSQ + single-step qSAM
#   bash run.sh resnet50 4 True
#   CUDA_VISIBLE_DEVICES=2 bash run.sh       # pin to physical GPU 2
#   DATA_DIR=/path/to/imagenet bash run.sh
set -e

NETWORK=${1:-resnet50}
N_BIT=${2:-4}
QUAN_DOWNSAMPLE=${3:-True}
# ImageNet root containing train/ and val/ subfolders (ImageFolder layout).
DATA_DIR=${DATA_DIR:-/workspace/hd/data/ImageNet}
USE_KD=${USE_KD:-False}
# qSAM toggle + knobs.
USE_QSAM=${USE_QSAM:-False}
QSAM_RATIO=${QSAM_RATIO:-0.001}   # fraction K/d of weights shifted per step
QSAM_RHO=${QSAM_RHO:-1.0}
QSAM_WARMUP=${QSAM_WARMUP:-0}     # pure-LSQ epochs before qSAM turns on
BN_RECAL=${BN_RECAL:-200}         # BN recalibration batches before each validate
# Optimizer / schedule / first-last precision (defaults = current harness).
# LSQ-paper recipe: OPTIMIZER=sgd LR_SCHED=cosine LR=0.01 EPOCHS=90 WD=1e-4 FIRST_LAST_BIT=8
OPTIMIZER=${OPTIMIZER:-adam}
LR_SCHED=${LR_SCHED:-linear}
LR=${LR:-1e-3}
EPOCHS=${EPOCHS:-50}
WD=${WD:-0}
FIRST_LAST_BIT=${FIRST_LAST_BIT:-0}
SAVE=${SAVE:-./models}
SEED=${SEED:-42}

LR_TAG=$(printf '%g' "${LR}")
WD_TAG=$(printf '%g' "${WD}")
QSAM_RATIO_TAG=$(printf '%g' "${QSAM_RATIO}")
QSAM_RHO_TAG=$(printf '%g' "${QSAM_RHO}")
TAG=${NETWORK}_${N_BIT}bit_qd_${QUAN_DOWNSAMPLE}_fl${FIRST_LAST_BIT}_opt_${OPTIMIZER}_sched_${LR_SCHED}_lr${LR_TAG}_ep${EPOCHS}_wd${WD_TAG}_qsam_${USE_QSAM}_kd_${USE_KD}
if [ "${USE_QSAM}" = "True" ]; then TAG=${TAG}_w${QSAM_WARMUP}_r${QSAM_RATIO_TAG}_rho${QSAM_RHO_TAG}; fi
TAG=${TAG}_seed${SEED}
LOG_DIR=log/${TAG}
mkdir -p ${LOG_DIR}

# Training harness (kept from prior setup so baseline and qSAM share it):
#   - Adam, betas (0.9, 0.999), linear LR decay from 1e-3
#   - batch size 128 (fits one 48GB GPU; QAT activation overhead), wd 0, 50 epochs
# Quantizer: faithful LSQ (learned step size + gradient scale, Esser et al. 2020).
# SAM: single-step S2-SAM (prior-gradient perturbation, zero extra cost).
python3 train.py \
    --data=${DATA_DIR} \
    --batch_size=128 \
    --learning_rate=${LR} \
    --epochs=${EPOCHS} \
    --weight_decay=${WD} \
    --momentum=0.9 \
    --optimizer=${OPTIMIZER} \
    --lr_scheduler=${LR_SCHED} \
    --first_last_n_bit=${FIRST_LAST_BIT} \
    --save=${SAVE} \
    --student=${NETWORK} \
    --teacher=resnet101 \
    --use_kd=${USE_KD} \
    --n_bit=${N_BIT} \
    --use_qsam=${USE_QSAM} \
    --qsam_ratio=${QSAM_RATIO} \
    --qsam_rho=${QSAM_RHO} \
    --qsam_warmup_epochs=${QSAM_WARMUP} \
    --bn_recal_batches=${BN_RECAL} \
    --quantize_downsample=${QUAN_DOWNSAMPLE} \
    --workers=8 \
    --seed=${SEED} \
    2>&1 | tee -a ${LOG_DIR}/training.txt
