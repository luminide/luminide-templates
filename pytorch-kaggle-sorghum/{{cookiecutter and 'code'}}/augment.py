import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

{% if cookiecutter.augmentation == "True" -%}

class SegmentGreen(A.ImageOnlyTransform):

    def __init__(self, always_apply=False, p=1.0):
        super(SegmentGreen, self).__init__(always_apply, p)

    def apply(self, img, **params):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hmin = random.randrange(25, 35)
        vmin = random.randrange(30, 70)
        smax = random.randrange(150, 180)
        mask = cv2.inRange(hsv, (hmin, 0, vmin) , (179, smax, 255))
        img = cv2.bitwise_and(img, img, mask=mask)
        return img

def make_train_augmenter(conf):
    p = conf.aug_prob
    if p <= 0:
        return A.Compose([
            A.Normalize(),
            ToTensorV2()
        ])

    crop_size = round(conf.image_size*conf.crop_size)
    aug_list = []
    if conf.max_cutout > 0:
        aug_list.append(
            A.CoarseDropout(
                max_holes=conf.max_cutout, min_holes=1,
                max_height=crop_size//10, max_width=crop_size//10,
                min_height=4, min_width=4, p=0.2*p))

    if conf.segment_green > 0:
        aug_list.append(SegmentGreen(p=conf.segment_green))

    aug_list.extend([
        A.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.2, rotate_limit=25,
            interpolation=cv2.INTER_AREA, p=p),
        A.RandomCrop(height=crop_size, width=crop_size, always_apply=True),
        A.Flip(p=0.5*p),
        A.OneOf([
            A.MotionBlur(p=0.2*p),
            A.MedianBlur(blur_limit=3, p=0.1*p),
            A.Blur(blur_limit=3, p=0.1*p),
        ], p=0.2*p),
        A.Perspective(p=0.2*p),
    ])

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
                A.ColorJitter(p=0.2*p),
            ], p=0.3*p),
        ])

    aug_list.extend([
        A.Normalize(),
        ToTensorV2()
    ])

    return A.Compose(aug_list)

{% elif cookiecutter.augmentation == "False" -%}
def make_train_augmenter(conf):
    aug_list = [
        A.Normalize(),
        ToTensorV2()
    ]

    return A.Compose(aug_list)

{%- endif %}
