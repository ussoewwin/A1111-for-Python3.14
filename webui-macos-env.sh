#!/bin/bash
####################################################################
#                          macOS defaults                          #
# Please modify webui-user.sh to change these instead of this file #
####################################################################

export install_dir="$HOME"
export COMMANDLINE_ARGS="--skip-torch-cuda-test --upcast-sampling --no-half-vae --use-cpu interrogate"
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Version-aligned with Windows/Linux CUDA default (torch 2.13 / torchvision 0.28); FA2 is skipped on macOS.
export TORCH_COMMAND="pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu"

####################################################################
