#!/usr/bin/env bash
set -e

METHOD=${1:-qvit}
GPU=${GPU:-2}
DATA=${DATA:-/workspace/AOQ-main/AO_QAT/data}
VARIANT=${VARIANT:-B_32}
NBITS=${NBITS:-4}
EPOCHS=${EPOCHS:-100}
BS=${BS:-64}
LR=${LR:-5e-4}
WORKERS=${WORKERS:-8}
OUT=${OUT:-log/google_vit_${VARIANT}_${NBITS}bit_${METHOD}}
PYTHON=${PYTHON:-/opt/conda/envs/aoq/bin/python}
DEVICE=${DEVICE:-cuda}
FINETUNE=${FINETUNE:-auto}
NO_FINETUNE=${NO_FINETUNE:-0}
DEBUG_TRAIN_BATCHES=${DEBUG_TRAIN_BATCHES:-0}
DEBUG_VAL_BATCHES=${DEBUG_VAL_BATCHES:-0}
SAM_RHO=${SAM_RHO:-0.05}

finetune_args=(--finetune "${FINETUNE}")
if [ "${NO_FINETUNE}" = "1" ]; then
  finetune_args=(--no-finetune)
fi

# wandb is opt-in: WANDB=1 bash run_google_vit_cifar100.sh   (needs `wandb login` first)
wandb_args=()
if [ "${WANDB:-0}" = "1" ]; then
  wandb_args=(--wandb --wandb-project "${WANDB_PROJECT:-aoq-vit-qat}")
  [ -n "${WANDB_RUN_NAME:-}" ] && wandb_args+=(--wandb-run-name "${WANDB_RUN_NAME}")
fi

CUDA_VISIBLE_DEVICES=${GPU} "${PYTHON}" -u google_vit_cifar100.py \
  --method "${METHOD}" \
  --variant "${VARIANT}" \
  --nbits "${NBITS}" \
  --data-path "${DATA}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BS}" \
  --lr "${LR}" \
  --workers "${WORKERS}" \
  --device "${DEVICE}" \
  --output-dir "${OUT}" \
  --debug-train-batches "${DEBUG_TRAIN_BATCHES}" \
  --debug-val-batches "${DEBUG_VAL_BATCHES}" \
  --sam-rho "${SAM_RHO}" \
  "${finetune_args[@]}" \
  "${wandb_args[@]}"
