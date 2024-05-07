
# Fully Sharded Data Parallel (FSDP) GPT2

The primary objective of this project is to demonstrate how to develop LLMs that can run and scale on PyTorch/CUDA, and PyTorch/XLA, without modifications to the model. The secondary objectives of this project are as follows:

1. Provide an easy to understand LLM model for learning GPT2 model architecture details
2. Demonstrate use of FSDP and activation checkpointing to scale LLM training by wrapping appropriate transformer layer class
3. Provide an easy to understand FSDP training loop that uses TensorBoard for training metrics
4. Demonstrate how to debug LLM code in Visual Studio Code for CUDA/XLA using Docker extensions

To keep things simple, we started with [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) repository, and simplified and adapted the model and the training loop to achieve the objectives outlined above.

For pre-training, the GPT2 model can be initialized from scratch, or loaded from the following pre-trained Hugging Face models: `gpt2`, `gpt2-medium`, `gpt2-large`,  and `gpt2-xl`. To load from a pre-trained Hugging Face model, use the command line argument `--hf_model=model-name`.  Command line arguments to the training script are automatically parsed and mapped to the training configuration. 

## Step-by-step Tutorial

Below we provide a step-by-step tutorial on how to debug, and pre-train the model using [AWS Deep Learning Desktop](https://github.com/aws-samples/aws-deep-learning-ami-ubuntu-dcv-desktop). Any other equivalent or better Nvidia GPU machine can be used for walking through the tutorial, as well.

### Launch AWS Deep Learning Desktop

Follow the instructions at this [repository](https://github.com/aws-samples/aws-deep-learning-ami-ubuntu-dcv-desktop) to launch a `g5.12xlarge` desktop with 4 NVIDIA A10G Tensor Core GPUs with 24GB memory per GPU. Specify `EbsVolumeSize` parameter at least 500 GB when launching the dekstop.

### Development Setup

Clone this repository under the home directory on the launched desktop, and `cd` into the cloned repository . Activate `pytorch` conda environment:

    conda activate pytorch

### Download and Convert Hugging Face Dataset

Install `tiktoken==0.5.2` in your `pytorch` conda environment:

    pip3 install tiktoken==0.5.2
    pip3 install -r requirements.txt

Convert Hugging Face `openwebtext` dataset into a dataset we can use with GPT2 by running:

    python3 gpt_data_converter.py

### Debug in Visual Studio Code

Launch pre-installed Visual Studio Code and open this repository in Code. Install [Python](https://code.visualstudio.com/docs/languages/python) and [Docker](https://code.visualstudio.com/docs/containers/overview) extensions for Visual Studio Code. Select `pytorch` conda environment Python interpreter (`Use Shift + CMD + P > Python: Select Interpreter`). 

There are three options for debugging the current file `train_fsdp.py` in Code:

1. To debug with CUDA running on the desktop use `Python: CurrentFile` debugger configuration in Code
2. To debug in a docker container running CUDA, use `Docker: Python Debug CUDA` debugger configuration. For this option, first run following command in a Code Terminal:

    `aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-west-2.amazonaws.com`

3. To debug in a docker container running XLA on top of CUDA, use `Docker: Python Debug XLA` debugger configuration 

### Distributed FSDP Training using CUDA on Desktop

To run distributed FSDP training on CUDA on the desktop, execute:

    ./run_cuda.sh 1>run_cuda.out 2>&1 &

### Distributed FSDP Training using XLA/CUDA in Docker Container

First  use `Docker: Python Debug XLA` debugger configuration in Code: This will build the `gp2-fsdp-xla:latest` Docker image locally on the desktop. Next, start the docker container for running distributed training on XLA running on top of CUDA:

    docker run -t -d -v /home/ubuntu/gpt2-fsdp:/app --shm-size=16g --net=host --gpus all docker.io/library/gp2-fsdp-xla:latest  sleep infinity

Next `exec` into the running Docker container using the docker container short id:

    docker exec -it CONTAINER_ID /bin/bash

Next, launch distributed training within the docker container:

    ./run_xla.sh 1>run_xla.out 2>&1 &

**Note:** In this case we are mounting the cloned repository on the `/app` directory of the Docker container. 

## Acknowledgements

[karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
