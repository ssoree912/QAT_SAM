#!/bin/bash
# Faithful LSQ-paper reproduction (Esser et al., ICLR 2020), NO knowledge distil.
#   - SGD momentum 0.9, cosine LR decay, lr 0.01, 90 epochs, weight decay 1e-4
#   - first conv & last fc quantized to 8-bit (rest 4-bit), no qSAM, no KD
# Target: ResNet-50 4-bit ~76.7% (paper Table 1, no-KD).
# Runs in a detached screen so it survives client sessions.
source /opt/conda/etc/profile.d/conda.sh
conda activate aoq
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
# LSQ-paper recipe
export USE_KD=False
export USE_QSAM=False
export OPTIMIZER=sgd
export LR_SCHED=cosine
export LR=0.01
export EPOCHS=90
export WD=1e-4
export FIRST_LAST_BIT=8
export SAVE=./models_lsqpaper
cd /workspace/AOQ-main/AO_QAT/resnet_imagenet
exec bash run.sh
