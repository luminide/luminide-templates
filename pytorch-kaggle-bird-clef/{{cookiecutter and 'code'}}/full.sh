#!/bin/bash -x

# Common packages are already installed on the compute server
# Need an additional package? Install it here via:
#  pip3 install package-name

pip3 install -q -r ../code/requirements.txt

# Edit the line below to run your experiment (this is just an example). Note:
#  - This script will be run from your output directory
#  - Imported Data is accessible via the relative path ../input/

{%- if cookiecutter.data_subset_percentage == '100' %}

/usr/bin/time -f "Time taken: %E" python3 ../code/train.py --epochs 60
{%- else %}

/usr/bin/time -f "Time taken: %E" python3 ../code/train.py --epochs 60 --subset {{ cookiecutter.data_subset_percentage }}
{%- endif %}
