import os
import cv2
import numpy as np
import pandas as pd
import torch.utils.data as data


class VisionDataset(data.Dataset):
    def __init__(
            self, meta_df, conf, input_dir, imgs_dir,
            num_classes, transform, training, quick=False):
        self.conf = conf
        self.num_classes = num_classes
        self.transform = transform

        if training and quick:
            # train on a subset
            split = meta_df.shape[0]//10
            meta_df = self.meta_df.iloc[:split].reset_index(drop=True)

        files = meta_df['{{ cookiecutter.image_column }}']
        assert isinstance(files[0], str), (
            f'column {meta_df.columns[0]} must be of type str')
        self.files = [os.path.join(input_dir, imgs_dir, f) for f in files]

        labels = np.int32(meta_df['{{ cookiecutter.label_column }}'].values)
        num_samples = len(files)
        self.onehot_labels = np.zeros((num_samples, num_classes), dtype=np.float32)
        self.onehot_labels[np.arange(num_samples), labels] = 1.0

    def __getitem__(self, index):
        conf = self.conf
        filename = self.files[index]
        assert os.path.isfile(filename)
        img = cv2.imread(filename)
{%- if cookiecutter.augmentation == "True" %}
        # increase crop size by around (margin*4%)
        img_width = conf.crop_width + round(conf.margin*conf.crop_width*0.01)*4
        img_height = conf.crop_height + round(conf.margin*conf.crop_height*0.01)*4
        img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_AREA)
{%- elif cookiecutter.augmentation == "False" %}
        img = cv2.resize(img, (conf.crop_width, conf.crop_height), interpolation=cv2.INTER_AREA)
{%- endif %}
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)['image']
        label = self.onehot_labels[index]
        return img, label

    def __len__(self):
        return len(self.files)
