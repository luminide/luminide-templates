import os
from glob import glob
import cv2
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from monai.data import NiftiSaver
from monai.metrics import compute_hausdorff_distance
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    ScaleIntensityd,
    Spacingd,
    EnsureTyped,
)


class LossHistory:
    def __init__(self):
        self.data = []

    def add_val_loss(self, epoch, sample_count, val):
        self.data.append(
            [len(self.data), epoch, sample_count, np.nan, val, np.nan])

    def add_train_loss(self, epoch, sample_count, val):
        self.data.append(
            [len(self.data), epoch, sample_count, val, np.nan, np.nan])

    def add_epoch_val_loss(self, epoch, sample_count, val):
        self.data.append(
            [len(self.data), epoch, sample_count, np.nan, np.nan, val])

    def save(self):
        columns = [
            'index', 'epoch', 'sample_count',
            'train_loss', 'val_loss', 'epoch_val_loss']
        df = pd.DataFrame(self.data, columns=columns)
        df.to_csv('history.csv', index=False)

def get_class_names(df):
    labels = df['{{ cookiecutter.label_column }}']
    return labels.unique()

def get_id(filename):
    # e.g. filename: case123_day20/scans/slice_0001_266_266_1.50_1.50.png
    # id: case123_day20_slice_0001
    tokens = filename.split('/')
    return tokens[-3] + '_' + '_'.join(tokens[-1].split('_')[:2])

# adapted from https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

def get_mask(filename, meta_df, class_names):
    img_id = get_id(filename)
    annos = meta_df[meta_df['{{ cookiecutter.image_column }}'] == img_id].reset_index(drop=True)

    basename = os.path.basename(filename)
    tokens = basename.split('_')
    height, width = int(tokens[3]), int(tokens[2])
    mask = np.zeros((height, width, len(class_names)), dtype=np.uint8)
    class_map = {name: i for i, name in enumerate(class_names)}
    for anno_id, row in annos.iterrows():
        anno = row['{{ cookiecutter.annotation_column }}']
        if pd.isnull(anno):
            continue

        class_name = row['{{ cookiecutter.label_column }}']
        class_id = class_map[class_name]
        mask[:, :, class_id] = rle_decode(anno, (height, width))
    return mask

def dice_coeff(labels, preds):
    scores = []
    for idx in range(labels.shape[0]):
        scores.append(2*(labels[idx]*preds[idx]).sum()/(labels[idx].sum() + preds[idx].sum() + 1e-6))
    return torch.stack(scores).mean()

def hausdorff_score(labels, preds):
    #XXX return 0 for now
    return 0.0
    dist = compute_hausdorff_distance(preds, labels).mean().item()
    # labels are in NCHWD format
    max_dist = np.linalg.norm(labels.shape[2:])
    if dist > max_dist:
        return 0
    return 1.0 - dist/max_dist

def process_files(conf, input_dir, img_dir, meta_df, class_names):
    filename = 'train_processed.csv'
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        # delete bad data
        df = df[~df['img_files'].str.contains('case7_day0') & ~df['img_files'].str.contains('case81_day30')].reset_index(drop=True)
    else:
        img_files = sorted(glob(f'{input_dir}/{img_dir}/**/*.{{ cookiecutter.file_extension }}', recursive=True))
        mask_dir = '../masks'
        for img_file in img_files:
            mask = get_mask(img_file, meta_df, class_names)
            basename = img_file.replace(f'{input_dir}/{img_dir}/', '')
            mask_filename = f'{mask_dir}/{basename}'
            dirname = os.path.dirname(mask_filename)
            os.makedirs(dirname, exist_ok=True)
            cv2.imwrite(mask_filename, mask)
        df = pd.DataFrame()
        # delete common prefix from paths
        img_files = [f.replace(f'{input_dir}/{img_dir}/', '') for f in img_files]
        df['img_files'] = img_files
        df.to_csv(filename, index=False)
    if conf.arch == 'Unet3D':
        return process_files_3d(input_dir, img_dir)
    return df

def save_nifti(output_dir, output_file, files):
    tokens = os.path.basename(files[0][:-4]).split('_')
    _, w, h, x, y = [float(item) for item in tokens[1:]]
    print(files[0], w, h, x, y)
    z = 3.0
    # read all files and stack them
    imgs = [cv2.imread(img_file, cv2.IMREAD_UNCHANGED) for img_file in files]
    data = np.stack(imgs)
    if len(data.shape) < 4:
        # add C dimension
        data = np.expand_dims(data, 3)
    # convert from DHWC to CHWD that NiftiSaver expects
    data = data.transpose((3, 1, 2, 0))
    print('before - CHWD', data.shape)

    # save as nifti
    saver = NiftiSaver(output_dir=output_dir, output_postfix='', resample=True, separate_folder=False)
    affine = np.eye(4)
    affine[0, 0] = x
    affine[1, 1] = y
    affine[2, 2] = z

    #target_affine = np.eye(4)*1.5
    target_affine = affine
    saver.save(
        data,
        meta_data={'filename_or_obj': output_file, 'affine': affine,
                   'original_affine': target_affine, 'dtype': data.dtype})

    # sanity check
    #img = nib.load(f'{output_dir}/{output_file}')
    #print('after-HWDC', img.shape)

def process_files_3d(input_dir, img_dir):
    filename = 'train_processed_3d.csv'
    if os.path.exists(filename):
        return  pd.read_csv(filename)

    msk_dir = '../masks'
    nii_dir = '../nifti'
    os.makedirs(nii_dir, exist_ok=True)
    case_list = sorted(glob(f'{input_dir}/{img_dir}/*/*'))
    print(f'{len(case_list)} cases')
    images = []
    masks = []
    for case_dir in case_list:
        case_name = case_dir.split('/')[-1]
        if case_name in ['case7_day0', 'case81_day30']:
            # skip bad data
            print(f'skipping {case_name}')
            continue
        img_files = sorted(glob(f'{case_dir}/**/*.{{ cookiecutter.file_extension }}', recursive=True))
        msk_files = [item.replace(f'{input_dir}/{img_dir}', msk_dir) for item in img_files]
        nifti_img_file = f'{case_name}.nii.gz'
        nifti_msk_file = f'{case_name}.msk.nii.gz'
        save_nifti(nii_dir, nifti_img_file, img_files)
        save_nifti(nii_dir, nifti_msk_file, msk_files)
        images.append(nifti_img_file)
        masks.append(nifti_msk_file)

    df = pd.DataFrame()
    df['img_files'] = images
    df['msk_files'] = masks
    df.to_csv(filename, index=False)
    return df

def visualize(**images):
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        image = image.transpose((1, 2, 0))
        image -= image.min()
        max_val = image.max()
        if max_val != 0:
            image /= max_val
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(name)
        plt.imshow(image)
    plt.show()

def make_val_augmenter_3d(conf):
    aug_list = [
        LoadImaged(keys=['img', 'msk']),
        EnsureChannelFirstd(keys=['img', 'msk']),
        #Spacingd(
        #    keys=['img', 'msk'], pixdim=(
        #    1.5, 1.5, 3.0), mode=('bilinear', 'nearest')),
        ScaleIntensityd(keys=['img'], minv=0, maxv=1),
        EnsureTyped(keys=['img', 'msk']),
    ]
    return Compose(aug_list)

def make_test_augmenter_3d(conf):
    aug_list = [
        LoadImaged(keys=['img']),
        EnsureChannelFirstd(keys=['img']),
        ScaleIntensityd(keys=['img'], minv=0, maxv=1),
        EnsureTyped(keys=['img']),
    ]
    return Compose(aug_list)

def make_val_augmenter(conf):
    if '3D' in conf.arch:
        return make_val_augmenter_3d(conf)

    return  A.Compose([
        ToTensorV2(transpose_mask=True)
    ])
