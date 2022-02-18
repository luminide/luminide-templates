#!/bin/bash 

{%- if cookiecutter.data_subset_percentage == '100' %}

python3 ../code/train.py --epochs 3 --seed 0
{%- else %}

python3 ../code/train.py --epochs 3 --seed 0 --subset {{ cookiecutter.data_subset_percentage }}
{%- endif %}
