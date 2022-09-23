#!/bin/bash

# Common packages are already installed on the compute server
# Need an additional package? Install it here via:
#  pip3 install package-name

# Edit the line below to run your experiment (this is just an example). Note:
#  - This script will be run from your output directory
#  - Imported Data is accessible via the relative path ../input/

set +x
rm *.png
set -x
{%- if cookiecutter.data_subset_percentage == '100' %}

/usr/bin/time -f "Time taken: %E" python3 ../code/train.py --epochs 200
{%- else %}

/usr/bin/time -f "Time taken: %E" python3 ../code/train.py --epochs 200 --subset {{ cookiecutter.data_subset_percentage }}
{%- endif %}
