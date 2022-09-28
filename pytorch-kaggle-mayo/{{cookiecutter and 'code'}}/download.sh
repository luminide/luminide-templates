#!/bin/bash

kaggle competitions download -c mayo-clinic-strip-ai -f train.csv
kaggle competitions download -c mayo-clinic-strip-ai -f test.csv
kaggle competitions download -c mayo-clinic-strip-ai -f sample_submission.csv
kaggle competitions download -c mayo-clinic-strip-ai -f test/008e5c_0.tif
kaggle competitions download -c mayo-clinic-strip-ai -f test/00c058_0.tif
kaggle competitions download -c mayo-clinic-strip-ai -f test/006388_0.tif
kaggle competitions download -c mayo-clinic-strip-ai -f test/01adc5_0.tif
unzip *.zip
mkdir test
mv *.tif test/
