#!/usr/bin/env bash
# Milestone 1: ViT QAT + qSAM on CIFAR-100, DeiT-Small, 4-bit, KD-off.
# Quantizer = original Q-ViT (LSQ, per-channel alpha), UNCHANGED. We only add qSAM.
#
# NON-NEGOTIABLE (see plan): the 4-bit student is INITIALIZED from an ImageNet-pretrained
# DeiT-S (--finetune) and QAT-fine-tuned on CIFAR-100 at 224x224. Never train from scratch.
# distilled is OFF (single head) -> pair with the NON-distilled DeiT-S checkpoint below.
set -e

GPU=${GPU:-2}
DATA=${DATA:-/workspace/AOQ-main/AO_QAT/data}
EPOCHS=${EPOCHS:-100}          # ViT@224 finetune is heavy (~8-9 min/epoch); ResNet used 200 @32px
BS=${BS:-128}
LR=${LR:-5e-4}
# qSAM hyperparams: match the ResNet-CIFAR100 ablation defaults (qsam_ratio=0.01, sam_rho=0.05).
# NOTE: rho scales one quant level; rho=1.0 (a full level shift) is too aggressive -> use 0.05.
QSAM_RATIO=${QSAM_RATIO:-0.01}
QSAM_RHO=${QSAM_RHO:-0.05}
QSAM_WARMUP=${QSAM_WARMUP:-5}
MODEL=${MODEL:-fourbits_deit_small_patch16_224}   # 4-bit DeiT-S; 2/3-bit: twobits_/threebits_
CKPT=https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth

# wandb is opt-in: WANDB=1 bash run.sh lsq   (needs `wandb login` first)
WANDB_ARGS=""
if [ "${WANDB:-0}" = "1" ]; then
  WANDB_ARGS="--wandb --wandb-project ${WANDB_PROJECT:-aoq-vit-qat}"
fi

common="--model $MODEL --data-set CIFAR --data-path $DATA --input-size 224 \
  --batch-size $BS --epochs $EPOCHS --lr $LR --finetune $CKPT --num_workers 8 $WANDB_ARGS"

case "${1:-lsq}" in
  lsq)        # baseline: LSQ ViT QAT, no qSAM
    CUDA_VISIBLE_DEVICES=$GPU python -u main.py $common \
      --output_dir log/qvit_s_4bit_lsq ;;
  lsq_qsam)   # + qSAM
    CUDA_VISIBLE_DEVICES=$GPU python -u main.py $common \
      --use-qsam --qsam-ratio $QSAM_RATIO --qsam-rho $QSAM_RHO --qsam-warmup-epochs $QSAM_WARMUP \
      --output_dir log/qvit_s_4bit_lsq_qsam ;;
  *) echo "usage: bash run.sh [lsq|lsq_qsam]"; exit 1 ;;
esac
