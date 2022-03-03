import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_class_names(df):
    labels = df['{{ cookiecutter.label_column }}']
    return np.unique(' '.join(labels.unique()).split())

def search_layer(module, layer_type, reverse=True):
    if isinstance(module, layer_type):
        return module

    if not hasattr(module, 'children'):
        return None

    children = list(module.children())
    if reverse:
        children = reversed(children)
    # search for the first occurence recursively
    for child in children:
        res = search_layer(child, layer_type)
        if res:
            return res
    return None

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