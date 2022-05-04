#!/bin/bash 
#
# train for a few epochs while performing "fast sweeps"
#

pip3 install -q -r ../code/requirements.txt

{%- if cookiecutter.data_subset_percentage == '100' %}

python3 ../code/train.py --epochs 15
{%- else %}

python3 ../code/train.py --epochs 15 --subset {{ cookiecutter.data_subset_percentage }}
{%- endif %}
