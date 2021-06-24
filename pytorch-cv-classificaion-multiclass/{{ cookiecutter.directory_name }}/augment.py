import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def make_augmenters(conf):
    p = conf.aug_prob
    aug_list = [
        A.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.2, rotate_limit=25,
            interpolation=cv2.INTER_CUBIC, p=p),
        A.RandomCrop(height=conf.crop_height, width=conf.crop_width, always_apply=True),
        A.Flip(p=0.5*p),
        A.OneOf([
            A.MotionBlur(p=0.2*p),
            A.MedianBlur(blur_limit=3, p=0.1*p),
            A.Blur(blur_limit=3, p=0.1*p),
        ], p=0.2*p),
        A.Perspective(p=0.2*p),
    ]

    if conf.strong_aug:
        aug_list.extend([
            A.GaussNoise(p=0.2*p),
            A.OneOf([
                A.OpticalDistortion(p=0.3*p),
                A.GridDistortion(p=0.1*p),
                A.PiecewiseAffine(p=0.3*p),
            ], p=0.2*p),
            A.OneOf([
                A.CLAHE(clip_limit=2, p=0.2*p),
                A.Sharpen(p=0.2*p),
                A.Emboss(p=0.2*p),
                A.RandomBrightnessContrast(p=0.2*p),
            ], p=0.3*p),
        ])

    aug_list.extend([
            A.Normalize(),
            ToTensorV2()
    ])

    train_aug = A.Compose(aug_list)
    test_aug = A.Compose([
        A.CenterCrop(height=conf.crop_height, width=conf.crop_width),
        A.Normalize(),
        ToTensorV2()
    ])

    return train_aug, test_aug
