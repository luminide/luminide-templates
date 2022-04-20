import os
import cv2
import numpy as np
from skimage import exposure
import torch.utils.data as data


class VisionDataset(data.Dataset):
    def __init__(
            self, df, conf, input_dir, imgs_dir,
            class_names, transform, subset=100):
        self.conf = conf
        self.transform = transform

        if subset != 100:
            assert subset < 100
            # train and validate on subsets
            num_rows = df.shape[0]*subset//100
            df = df.iloc[:num_rows]

        files = df['{{ cookiecutter.image_column }}']
        assert isinstance(files[0], str), (
            f'column {df.columns[0]} must be of type str')
        self.files = [os.path.join(input_dir, imgs_dir, f.replace('.png', '.jpg')) for f in files]

        labels = df['{{ cookiecutter.label_column }}']
        num_samples = len(files)
        num_classes = len(class_names)
        class_map = {class_names[i]: i for i in range(num_classes)}
        self.labels = np.zeros((num_samples, num_classes), dtype=np.float32)
        for i in range(num_samples):
            row_labels = [class_map[token] for token in labels[i].split(' ')]
            self.labels[i, row_labels] = 1.0

    def __getitem__(self, index):
        conf = self.conf
        filename = self.files[index]
        assert os.path.isfile(filename)
        img = cv2.imread(filename)
{%- if cookiecutter.augmentation == "True" %}
        img = cv2.resize(
            img, (conf.image_size, conf.image_size),
            interpolation=cv2.INTER_AREA)
{%- elif cookiecutter.augmentation == "False" %}
        crop_size = round(conf.image_size*conf.crop_size)
        img = cv2.resize(img, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
{%- endif %}
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if conf.equalize_hist:
            img = exposure.equalize_adapthist(img, nbins=255).astype(np.float32)
            img = (img*255).round().astype(np.uint8)
        if self.transform:
            img = self.transform(image=img)['image']
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.files)
