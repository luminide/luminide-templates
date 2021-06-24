import os
import cv2
import numpy as np
import pandas as pd
import torch.utils.data as data


class VisionDataset(data.Dataset):
    def __init__(self, training, conf, input_dir, imgs_dir, num_classes, transform, quick):
        self.conf = conf
        self.num_classes = num_classes
        self.transform = transform
        meta_file = os.path.join(input_dir, '{{ cookiecutter.train_metadata }}')

        self.meta_df = pd.read_csv(meta_file)
        if training:
            # shuffle the dataset
            self.meta_df = self.meta_df.sample(frac=1, random_state=0).reset_index(drop=True)
            if quick:
                # train on a subset
                split = self.meta_df.shape[0]//10
                self.meta_df = self.meta_df.iloc[:split].reset_index(drop=True)

        files = self.meta_df['{{ cookiecutter.image_column }}']
        assert isinstance(files[0], str), (
            f'column {self.meta_df.columns[0]} must be of type str')
        self.files = [os.path.join(input_dir, imgs_dir, f) for f in files]

        labels = np.int32(self.meta_df['{{ cookiecutter.label_column }}'].values)
        num_samples = len(files)
        self.onehot_labels = np.zeros((num_samples, num_classes), dtype=np.float32)
        self.onehot_labels[np.arange(num_samples), labels] = 1.0
        print(f'{num_samples} examples')

    def __getitem__(self, index):
        conf = self.conf
        filename = self.files[index]
        assert os.path.isfile(filename)
        img = cv2.imread(filename)
        # increase crop size by around (margin*4%)
        img_width = conf.crop_width + round(conf.margin*conf.crop_width*0.01)*4
        img_height = conf.crop_height + round(conf.margin*conf.crop_height*0.01)*4
        img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)['image']
        label = self.onehot_labels[index]
        return img, label

    def __len__(self):
        return len(self.files)