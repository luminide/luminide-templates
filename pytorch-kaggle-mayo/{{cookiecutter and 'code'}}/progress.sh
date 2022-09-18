#!/bin/bash
#
# generate an HTML report of the last SSL training session
#

jupyter nbconvert --execute ../code/progress.ipynb --no-input --to html --output-dir ./
