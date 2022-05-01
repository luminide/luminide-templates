#!/bin/bash 
#
# train for a few epochs while performing "fast sweeps"
#

{%- if cookiecutter.data_subset_percentage == '100' %}

python3 ../code/train.py --epochs 5
{%- else %}

python3 ../code/train.py --epochs 5 --subset {{ cookiecutter.data_subset_percentage }}
{%- endif %}
