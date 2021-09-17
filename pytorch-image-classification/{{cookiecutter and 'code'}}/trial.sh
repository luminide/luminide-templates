#!/bin/bash 

export TF_CPP_MIN_LOG_LEVEL=2

python3 ../code/main.py --epochs 3 -b 16 --seed 0
