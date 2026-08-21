#!/bin/bash
# CIFAR-100 ResNet-50 4-bit, seed 43: baseline then qSAM, SEQUENTIALLY on GPU2.
# Second paired seed for the qSAM-vs-baseline reproducibility check (seed42 done).
source /opt/conda/etc/profile.d/conda.sh
conda activate aoq
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
cd /workspace/AOQ-main/AO_QAT/resnet_cifar100

run() {  # $1 = method
  local TAG=$1_resnet50_4bit_seed43
  mkdir -p log/$TAG
  echo "=== [$(date '+%H:%M:%S')] START $1 seed43 ===" | tee -a log/$TAG/training.txt
  python3 train.py --method=$1 --model=resnet50 --n_bit=4 --epochs=200 \
      --seed=43 --data=./data 2>&1 | tee -a log/$TAG/training.txt
}

run lsq          # baseline first
run lsq_qsam     # then qSAM
echo "=== [$(date '+%H:%M:%S')] BOTH seed43 DONE ==="
