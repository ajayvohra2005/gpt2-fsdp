#!/bin/bash

ulimit -n 65536
NCCL_DEBUG=Info
NCCL_SOCKET_IFNAME=lo
GPU_NODES=1
GPU_NUM_DEVICES=$(nvidia-smi --list-gpus | wc -l)

torchrun \
--nnode=$GPU_NODES \
--nproc-per-node=$GPU_NUM_DEVICES \
train_fsdp.py \
--dataset_dir='data' \
--log_dir='logs' \
--checkpoint_dir='checkpoints' \
--hf_model="gpt2" \
--batch_size=8
