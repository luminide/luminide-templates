#!/bin/bash -x
#
# resume training from a saved model
#

{%- if cookiecutter.data_subset_percentage == '100' %}

/usr/bin/time -f "Time taken: %E" python3 ../code/train.py --epochs 2000 --resume latest.pth
{%- else %}

/usr/bin/time -f "Time taken: %E" python3 ../code/train.py --epochs 1000 --resume latest.pth --subset {{ cookiecutter.data_subset_percentage }}
{%- endif %}
