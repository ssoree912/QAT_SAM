#!/bin/bash
set -e

METHOD=${1:-all}
N_BIT=${2:-4}
QSAM_RATIO=${3:-0.01}
EPOCHS=${EPOCHS:-200}
DATA=${DATA:-./data}
MODEL=${MODEL:-resnet18}
SAM_RHO=${SAM_RHO:-0.05}

run_one() {
    local m=$1
    local tag="${m}_${MODEL}_${N_BIT}bit"
    mkdir -p "log/${tag}"
    echo "=========================================="
    echo "Method: ${m}  n_bit=${N_BIT}  qsam_ratio=${QSAM_RATIO}  sam_rho=${SAM_RHO}  epochs=${EPOCHS}"
    echo "=========================================="
    python3 train.py \
        --method=${m} \
        --model=${MODEL} \
        --n_bit=${N_BIT} \
        --epochs=${EPOCHS} \
        --batch_size=128 \
        --learning_rate=1e-3 \
        --weight_decay=0 \
        --qsam_ratio=${QSAM_RATIO} \
        --sam_rho=${SAM_RHO} \
        --lambda_dampen=1e-3 \
        --workers=4 \
        --data=${DATA} \
        --seed=42 \
        2>&1 | tee -a "log/${tag}/training.txt"
}

if [ "$METHOD" = "all" ]; then
    for m in lsq lsq_sam lsq_aoq lsq_qsam lsq_aoq_qsam; do
        run_one $m
    done
else
    run_one $METHOD
fi
