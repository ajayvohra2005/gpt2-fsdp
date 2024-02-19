#!/bin/bash

ulimit -n 65536

GPU_NODES=1
GPU_NUM_DEVICES=$(nvidia-smi --list-gpus | wc -l)

PJRT_DEVICE=CUDA XLA_USE_BF16=1 torchrun \
--rdzv_backend=c10d \
--rdzv_endpoint=localhost:0 \
--rdzv_id=gpt2-fsdp-xla \
--nnodes=$GPU_NODES \
--nproc_per_node=$GPU_NUM_DEVICES \
  train_fsdp.py \
--device_type='xla' \
--dataset_dir='data' \
--log_dir='logs' \
--checkpoint_dir='checkpoints' \
--cache_dir="cache" \
--batch_size=8 \
--mixed_precision='True' \
--hf_model="gpt2"
