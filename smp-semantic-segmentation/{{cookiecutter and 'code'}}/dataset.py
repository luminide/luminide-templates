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

    def load_slice(self, img_file, diff):
        slice_num = os.path.basename(img_file).split('_')[1]
        filename = img_file.replace('slice_' + slice_num, 'slice_' + str(int(slice_num) + diff).zfill(4))
        if not os.path.exists(filename):
            return None
        return cv2.imread(filename, cv2.IMREAD_UNCHANGED)

    def __getitem__(self, index):
        conf = self.conf
        img_file = self.files[index]
        assert os.path.isfile(img_file)
        # read 5 slices into one image
        imgs = [self.load_slice(img_file, i) for i in range(-2, 3)]
        assert imgs[2] is not None
        if imgs[3] is None:
            imgs[3] = imgs[2]
        if imgs[4] is None:
            imgs[4] = imgs[3]
        if imgs[1] is None:
            imgs[1] = imgs[2]
        if imgs[0] is None:
            imgs[0] = imgs[1]
        img = np.stack(imgs, axis=2)

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
