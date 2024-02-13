#!/bin/bash

ulimit -n 65536
NCCL_DEBUG=Info
NCCL_SOCKET_IFNAME=lo
torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
--nnodes=1 --nproc-per-node=4 train_fsdp.py \
--device_type='cuda' \
--dataset_dir='data' \
--log_dir='logs' \
--checkpoint_dir='checkpoints' \
--mixed_precision='True' \
--hf_model="gpt2"
