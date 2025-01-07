#!/bin/bash

mkdir -p /home/ubuntu/cache
mkdir -p /home/ubuntu/tmp

docker run -t -d -v /home/ubuntu/gpt2-fsdp:/app \
    -v /home/ubuntu/cache:/cache \
    -v /home/ubuntu/tmp:/tmp \
    --shm-size=16g --net=host \
    --device=/dev/neuron0 \
    --device=/dev/neuron1 \
    --device=/dev/neuron2 \
    --device=/dev/neuron3 \
    --device=/dev/neuron4 \
    --device=/dev/neuron5 \
    --device=/dev/neuron6 \
    --device=/dev/neuron7 \
    --device=/dev/neuron8 \
    --device=/dev/neuron9 \
    --device=/dev/neuron10 \
    --device=/dev/neuron11 \
    --device=/dev/neuron12 \
    --device=/dev/neuron13 \
    --device=/dev/neuron14 \
    --device=/dev/neuron15 \
    docker.io/library/gpt2-fsdp-xla-neuron   sleep infinity