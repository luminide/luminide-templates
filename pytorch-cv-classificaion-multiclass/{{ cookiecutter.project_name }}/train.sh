#!/bin/bash -x

export TF_CPP_MIN_LOG_LEVEL=2

python3 ../code/main.py --epochs 2 -b 16 --seed 0 -q
