import os
import cv2
import numpy as np
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

        files = df['img_files']
        self.files = [os.path.join(input_dir, imgs_dir, f) for f in files]
        self.masks = [os.path.join('../masks', f) for f in files]

    def resize(self, img):
        return  cv2.resize(
            img, (self.conf.image_size, self.conf.image_size),
            interpolation=cv2.INTER_AREA)

    def __getitem__(self, index):
        conf = self.conf
        img_file = self.files[index]
        assert os.path.isfile(img_file)
        # read image
        img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
        img = np.stack((img, img, img), axis=2)
        img = img.astype(np.float32)
        max_val = img.max()
        if max_val != 0:
            img /= max_val
        img = self.resize(img)

        # read mask
        msk_file = self.masks[index]
        msk = cv2.imread(msk_file, cv2.IMREAD_UNCHANGED)
        msk = self.resize(msk)
        msk = msk.astype(np.float32)
        if self.transform:
            result = self.transform(image=img, mask=msk)
            img, msk = result['image'], result['mask']
        return img, msk

    def __len__(self):
        return len(self.files)
