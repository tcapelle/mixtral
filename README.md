# mixtral

This repo aims to fine-tune Mixtral model.

## Requirements

I provided a small Docker image to run the code. You can build it or pull [from this repo](https://github.com/tcapelle/mixtral/pkgs/container/mixtral).

I have not been able to install latest `flash_attn` in any container that has the necessary PyTorch and Cuda versions. This image is made to run on H100 GPUs.

## Run

- Run the simple_inference.py script to test the model. It actually runs on an A100 with 40GB of memory!

## Train

Test you system by running the `simple_train.py` script. It will train a model on a small dataset. It takes around 1 hour on a 8xH100 machine.