#!/bin/bash -x

# Common packages are already installed on the compute server
# Need an additional package? Install it here via:
#  pip3 install package-name

# Edit the line below to run your experiment (this is just an example). Note:
#  - This script will be run from your output directory
#  - Imported Data is accessible via the relative path ../input/

pip3 install -q -r ../code/requirements.txt
{%- if cookiecutter.data_subset_percentage == '100' %}

/usr/bin/time -f "Time taken: %E" python3 ../code/train.py --epochs 2000 --fold 0
{%- else %}

/usr/bin/time -f "Time taken: %E" python3 ../code/train.py --epochs 2000 --subset {{ cookiecutter.data_subset_percentage }} --fold 0
{%- endif %}
