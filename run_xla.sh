#!/bin/bash

ulimit -n 65536

PJRT_DEVICE=CUDA XLA_USE_BF16=1 torchrun \
--rdzv_backend=c10d \
--rdzv_endpoint=localhost:0 \
--nnodes=1 --nproc_per_node=4 train_fsdp.py \
--device_type='xla' \
--dataset_dir='data' \
--log_dir='logs' \
--checkpoint_dir='checkpoints' \
--cache_dir="cache" \
--batch_size=8 \
--mixed_precision='True' \
--hf_model="gpt2"