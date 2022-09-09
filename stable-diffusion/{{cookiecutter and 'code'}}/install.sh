#!/bin/bash -x

sudo apt -qq update
sudo apt -qq install git
pip3 install -q diffusers transformers ftfy gdown av

[[ ! -f ~/.huggingface/token ]]  && { huggingface-cli login; }

git clone https://github.com/google-research/frame-interpolation ../film_net
gdown https://drive.google.com/drive/folders/1q8110-qp225asX3DQvZnfLfJPkCHmDpy?usp=sharing -O ../film_net_models --folder