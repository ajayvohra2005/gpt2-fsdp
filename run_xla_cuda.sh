#!/bin/bash

ulimit -n 65536

GPU_NODES=1
GPU_NUM_DEVICES=$(nvidia-smi --list-gpus | wc -l)
export PT_XLA_DEBUG=0 
export PJRT_DEVICE=CUDA 

torchrun \
--rdzv_backend=c10d \
--rdzv_endpoint=localhost:0 \
--rdzv_id=gpt2-fsdp-xla \
--nnodes=$GPU_NODES \
--nproc_per_node=$GPU_NUM_DEVICES \
  train_fsdp.py \
--dataset_dir='data' \
--log_dir='logs' \
--checkpoint_dir='checkpoints' \
--cache_dir="cache/xla-cuda" \
--batch_size=4 \
--hf_model="gpt2"
