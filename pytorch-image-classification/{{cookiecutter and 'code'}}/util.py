import os
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_class_names(df):
    labels = df['{{ cookiecutter.label_column }}']
    return np.unique(' '.join(labels.unique()).split()).tolist()

{% if cookiecutter.augmentation == "True" -%}
def make_test_augmenter(conf):
    crop_size = round(conf.image_size*conf.crop_size)
    return  A.Compose([
        A.CenterCrop(height=crop_size, width=crop_size),
        A.Normalize(),
        ToTensorV2()
    ])

{% elif cookiecutter.augmentation == "False" -%}
def make_test_augmenter(conf):
    aug_list = [
        A.Normalize(),
        ToTensorV2()
    ]

    return A.Compose(aug_list)
{%- endif %}