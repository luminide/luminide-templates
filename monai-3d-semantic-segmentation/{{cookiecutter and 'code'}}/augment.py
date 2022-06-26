import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch

from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    RandSpatialCropd,
    ResizeWithPadOrCropd,
    RandScaleIntensityd,
    RandGaussianSmoothd,
    RandGaussianSharpend,
    RandShiftIntensityd,
    ScaleIntensityd,
    Spacingd,
    EnsureTyped,
    RandAffined,
    RandBiasFieldd,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandCoarseDropoutd,
    RandGridDistortiond,
    OneOf,
    RandFlipd,
)

def make_train_augmenter_3d(conf):
    side = conf.train_roi
    depth = conf.train_depth
    p = conf.aug_prob
    angle = np.pi/36
    aug_list = [
        LoadImaged(keys=['img', 'msk']),
        EnsureChannelFirstd(keys=['img', 'msk']),
        #Spacingd(
        #    keys=['img', 'msk'], pixdim=(
        #    1.5, 1.5, 3.0), mode=('bilinear', 'nearest')),
        RandSpatialCropd(
            keys=['img', 'msk'],
            roi_size=(side, side, depth)),
        ScaleIntensityd(keys=['img'], minv=0, maxv=1),
        RandFlipd(keys=('img', 'msk'), prob=0.5, spatial_axis=[0]),
        RandFlipd(keys=('img', 'msk'), prob=0.5, spatial_axis=[1]),
        ResizeWithPadOrCropd(keys=['img', 'msk'], spatial_size=(side, side, depth)),
        RandAffined(
             keys=['img', 'msk'],
             mode=('bilinear', 'nearest'),
             prob=p,
             rotate_range=3*(angle,),
             scale_range=3*(0.1,)),
        OneOf([
            RandCoarseDropoutd(
                keys=['img', 'msk'],
                holes=5,
                max_holes=8,
                spatial_size=(1, 1, 1),
                max_spatial_size=(12, 12, 12),
                fill_value=0.0,
                prob=p),
            RandGridDistortiond(keys=('img', 'msk'), prob=p, distort_limit=(-0.05, 0.05)),
        ]),
    ]

    if conf.strong_aug:
        aug_list.append(
            OneOf([
                RandShiftIntensityd(keys='img', offsets=(-0.1, 0.1), prob=p),
                RandBiasFieldd(keys='img', prob=p),
                RandGaussianNoised(keys='img', prob=0.5*p),
                RandAdjustContrastd(keys='img', prob=0.5*p),
            ]))
        aug_list.append(
            OneOf([
                RandScaleIntensityd(keys='img', factors=(-0.2, 0.2), prob=p),
                RandGaussianSmoothd(keys='img', prob=0.5*p),
                RandGaussianSharpend(keys='img', prob=0.5*p),
            ]))

    aug_list.append(EnsureTyped(keys=['img', 'msk']))
    return Compose(aug_list)

def make_train_augmenter(conf):
    if '3D' in conf.arch:
        return make_train_augmenter_3d(conf)

    p = conf.aug_prob
    crop_size = conf.train_roi
    if p <= 0:
        return A.Compose([
            A.RandomCrop(height=crop_size, width=crop_size, always_apply=True),
            ToTensorV2(transpose_mask=True)
        ])

    aug_list = []
    if conf.max_cutout > 0:
        aug_list.extend([
            A.CoarseDropout(
                max_holes=conf.max_cutout, min_holes=1,
                max_height=crop_size//10, max_width=crop_size//10,
                min_height=4, min_width=4, mask_fill_value=0, p=0.2*p),
        ])

    aug_list.extend([
        A.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.2, rotate_limit=25,
            interpolation=cv2.INTER_AREA, p=p),
        A.RandomCrop(height=crop_size, width=crop_size, always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.OneOf([
            A.MotionBlur(p=0.2*p),
            A.MedianBlur(blur_limit=3, p=0.1*p),
            A.Blur(blur_limit=3, p=0.1*p),
        ], p=0.2*p),
        A.Perspective(p=0.2*p),
    ])

    if conf.strong_aug:
        aug_list.extend([
            A.GaussNoise(var_limit=0.001, p=0.2*p),
            A.OneOf([
                A.OpticalDistortion(p=0.3*p),
                A.GridDistortion(p=0.1*p),
                A.PiecewiseAffine(p=0.3*p),
            ], p=0.2*p),
            A.OneOf([
                A.Sharpen(p=0.2*p),
                A.Emboss(p=0.2*p),
                A.RandomBrightnessContrast(p=0.2*p),
            ], p=0.3*p),
        ])

    aug_list.extend([
        ToTensorV2(transpose_mask=True)
    ])

    return A.Compose(aug_list)
