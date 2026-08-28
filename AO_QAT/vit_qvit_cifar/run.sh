#!/usr/bin/env bash
# Milestone 1: ViT QAT on CIFAR-100, DeiT-Small, 4-bit, KD-off.
# Quantizer = original Q-ViT per-channel learned step size.
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
MODEL=${MODEL:-fourbits_deit_small_patch16_224}   # 4-bit DeiT-S; 2/3-bit: twobits_/threebits_
CKPT=https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth

# wandb is opt-in: WANDB=1 bash run.sh qvit   (needs `wandb login` first)
WANDB_ARGS=""
if [ "${WANDB:-0}" = "1" ]; then
  WANDB_ARGS="--wandb --wandb-project ${WANDB_PROJECT:-aoq-vit-qat}"
fi

common="--model $MODEL --data-set CIFAR --data-path $DATA --input-size 224 \
  --batch-size $BS --epochs $EPOCHS --lr $LR --finetune $CKPT --num_workers 8 $WANDB_ARGS"

case "${1:-qvit}" in
  qvit)        # baseline: QViT QAT, no qSAM
    CUDA_VISIBLE_DEVICES=$GPU python -u main.py $common \
      --output_dir log/qvit_s_4bit ;;
  *) echo "usage: bash run.sh [qvit]"; exit 1 ;;
esac
