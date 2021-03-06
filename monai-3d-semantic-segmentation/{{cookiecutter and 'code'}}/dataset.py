import os
import cv2
import numpy as np
import torch
import torch.utils.data as torchdata
from monai.data import Dataset as MonaiDataset


class VisionDataset(torchdata.Dataset):
    def __init__(
            self, df, conf, input_dir, imgs_dir,
            class_names, transform, is_test=False, subset=100):
        self.conf = conf
        self.transform = transform
        self.is_test = is_test
        self.num_classes = len(class_names)

        if 'num_slices' not in self.conf._params:
            self.conf['num_slices'] = 5
        if subset != 100:
            assert subset < 100
            # train and validate on subsets
            num_rows = df.shape[0]*subset//100
            df = df.iloc[:num_rows]

        files = df['img_files']
        self.files = [os.path.join(input_dir, imgs_dir, f) for f in files]
        self.masks = [os.path.join('../masks', f) for f in files]

    def resize(self, img, interp):
        return  cv2.resize(
            img, (self.conf.image_size, self.conf.image_size), interpolation=interp)

    def load_slice(self, img_file, diff, interp):
        slice_num = os.path.basename(img_file).split('_')[1]
        filename = (
            img_file.replace(
                'slice_' + slice_num,
                'slice_' + str(int(slice_num) + diff).zfill(4)))
        if os.path.exists(filename):
            img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            img = self.resize(img, interp)
            return img.astype(np.float32)
        return None

    def __getitem__(self, index):
        conf = self.conf
        num_slices = conf.num_slices
        assert num_slices % 2 == 1
        img_file = self.files[index]
        # read multiple slices into one image
        # in HWD format
        img = np.zeros(
            (conf.image_size, conf.image_size, num_slices), dtype=np.float32)
        for i, diff in enumerate(range(-(num_slices//2), num_slices//2 + 1)):
            slc =  self.load_slice(img_file, diff, cv2.INTER_AREA)
            if slc is None:
                continue
            img[:, :, i] = slc

        max_val = img.max()
        if max_val != 0:
            img /= max_val

        if self.is_test:
            msk = 0
            result = self.transform(image=img)
            img = result['image']
        else:
            # read mask
            msk_file = self.masks[index]
            if conf.multi_slice_label:
                msk = np.zeros(
                    (conf.image_size, conf.image_size, self.num_classes*num_slices), dtype=np.float32)
                for i, diff in enumerate(range(-(num_slices//2), num_slices//2 + 1)):
                    slc =  self.load_slice(msk_file, diff, cv2.INTER_NEAREST)
                    if slc is None:
                        continue
                    start = i*self.num_classes
                    msk[:, :, start:start + self.num_classes] = slc
            else:
                # in HWC format
                msk = cv2.imread(msk_file, cv2.IMREAD_UNCHANGED)
                msk = self.resize(msk, cv2.INTER_NEAREST)
                msk = msk.astype(np.float32)
            result = self.transform(image=img, mask=msk)
            img, msk = result['image'], result['mask']
        return img, msk

    def __len__(self):
        return len(self.files)

class VisionDataset3D(MonaiDataset):
    def __init__(
            self, df, conf, input_dir, imgs_dir,
            class_names, transform, is_test=False, subset=100):
        self.conf = conf
        self.transform = transform
        self.is_test = is_test
        self.num_classes = len(class_names)

        if subset != 100:
            assert subset < 100
            # train and validate on subsets
            num_rows = df.shape[0]*subset//100
            df = df.iloc[:num_rows]

        imgs = [f'../nifti/{f}' for f in df['img_files']]
        msks = [f'../nifti/{f}' for f in df['msk_files']]
        data = [
            {'img': img, 'msk': msk} for img, msk in zip(imgs, msks)
        ]

        super().__init__(data=data, transform=transform)

    def __getitem__(self, index):
        item =  super().__getitem__(index)
        # in CHWD format
        return item

    def set_random_state(self, seed):
        pass


def create_dataset(
        df, conf, input_dir, imgs_dir,
        class_names, transform, is_test=False, subset=100):
    ds = VisionDataset3D if '3D' in conf.arch else VisionDataset
    return ds(
        df, conf, input_dir, imgs_dir,
        class_names, transform, is_test, subset)
