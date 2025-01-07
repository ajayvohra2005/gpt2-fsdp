#!/bin/bash

ulimit -n 65536

export CACHE_DIR=/cache/xla-neuron
export XDG_CACHE_HOME=/cache
export PJRT_DEVICE=NEURON
export NEURON_CC_FLAGS="--cache_dir=$CACHE_DIR --model-type=transformer --optlevel=1"
export NEURON_RT_STOCHASTIC_ROUNDING_EN="1"
export XLA_IR_SHAPE_CACHE_SIZE="20480"
export CCOM_SOCKET_IFNAME="ens32"
export FI_EFA_USE_DEVICE_RDMA="1"
export FI_PROVIDER="efa"
export FI_EFA_FORK_SAFE="1"

NUM_EPOCHS=2
if [ "$NEURON_EXTRACT_GRAPHS_ONLY" = "1" ]
then
  EXTRA_ARGS="--epochs=1 --save_ckpt='False' --max_dataset_len=40960"
else
  EXTRA_ARGS="--epochs=$NUM_EPOCHS --save_ckpt='True'"
fi

torchrun \
--rdzv_backend=c10d \
--rdzv_endpoint=localhost:0 \
--rdzv_id=gpt2-fsdp-neuron \
--nnodes=1 \
--nproc_per_node=32 \
  train_fsdp.py \
--dataset_dir='data' \
--log_dir='logs' \
--checkpoint_dir='checkpoints' \
--cache_dir="${CACHE_DIR}" \
--batch_size=2 \
--hf_model="gpt2" \
${EXTRA_ARGS}
