import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def make_augmenters(conf):
    aug_list = [
        A.Normalize(),
        ToTensorV2()
    ]

    train_aug = A.Compose(aug_list)
    test_aug = train_aug

    return train_aug, test_aug
