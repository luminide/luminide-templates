#!/bin/bash -ex

# move image files to a flat directory
cd ../input
mkdir train_images
mv -vi train_images_*/*.jpg train_images/
rm -rf train_images_*
