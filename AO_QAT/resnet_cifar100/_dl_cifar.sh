#!/bin/bash
cd /workspace/AOQ-main/AO_QAT/resnet_cifar100
echo "[$(date +%H:%M:%S)] download start" > _dl.log
curl -L -s -o data/cifar-100-python.tar.gz "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
sz=$(stat -c %s data/cifar-100-python.tar.gz)
echo "[$(date +%H:%M:%S)] downloaded $sz bytes (target 169001437)" >> _dl.log
if [ "$sz" = "169001437" ]; then
  tar -xzf data/cifar-100-python.tar.gz -C data/ && echo "[$(date +%H:%M:%S)] EXTRACT OK" >> _dl.log
else
  echo "[$(date +%H:%M:%S)] SIZE MISMATCH" >> _dl.log
fi
ls data/cifar-100-python/ >> _dl.log 2>&1
