#!/bin/bash 

export TF_CPP_MIN_LOG_LEVEL=2

python3 ../code/train.py --epochs 3 --seed 0
