#!/bin/bash -x

export TF_CPP_MIN_LOG_LEVEL=2

python3 ../code/main.py --epochs 10 -b 16 --seed 0 -q
