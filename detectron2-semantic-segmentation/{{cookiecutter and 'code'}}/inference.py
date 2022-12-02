import argparse
import glob
import cv2
import numpy as np
import pandas as pd
import torch

import detectron2
from detectron2.config import CfgNode as CN
from detectron2.engine import DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

from util import get_class_names
from dataset import get_id


parser = argparse.ArgumentParser()
parser.add_argument(
    '--input', default='../input', metavar='DIR', help='input directory')
parser.add_argument(
    '--model_dir', default='.', type=str, metavar='PATH',
    help='path to saved models')


def rle_encode(img):
    '''
    this function is adapted from
    https://www.kaggle.com/code/stainsby/fast-tested-rle/notebook

    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def get_mask(pred, class_id):
    # aggregate instances of the given class
    instances = pred['instances']
    if len(instances) == 0:
        return None

    result = np.zeros(instances.image_size, dtype=np.uint8)
    found = False
    for idx in range(len(instances)):
        instance = instances[idx]
        if instance.pred_classes == class_id:
            found = True
            result += instance.pred_masks.cpu().numpy().squeeze()

    result[result > 1] = 1
    if found:
        return result
    return None

def run(input_dir, model_dir):
    meta_file = f'{input_dir}/{{ cookiecutter.train_metadata }}'
    meta_df = pd.read_csv(meta_file)
    class_names = get_class_names(meta_df)

    cfg = CN.load_cfg(open(f'{model_dir}/cfg.yaml'))
    print('processing test set...')
    model_file = f'{model_dir}/model.pth'
    cfg.MODEL.WEIGHTS = model_file
    print(f'loading {model_file}')

    model = build_model(cfg)
    DetectionCheckpointer(model).load(model_file)

    predictor = DefaultPredictor(cfg)
    test_names = []
    subdir = ''
    while len(test_names) == 0 and len(subdir) < 10:
        test_names = sorted(glob.glob(f'{input_dir}/test/{subdir}*.{{ cookiecutter.file_extension }}'))
        subdir += '*/'
    subm = pd.read_csv(f'{input_dir}/sample_submission.csv')
    del subm['predicted']
    ids = []
    classes = []
    masks = []
    for img_file in test_names:
        img_data = cv2.imread(img_file)
        pred = predictor(img_data)
        img_id = get_id(img_file)
        for class_id, class_name in enumerate(class_names):
            img_mask = get_mask(pred, class_id)
            enc_mask = '' if img_mask is None else rle_encode(img_mask)
            ids.append(img_id)
            classes.append(class_name)
            masks.append(enc_mask)

    pred_df = pd.DataFrame({'id': ids, 'class': classes, 'predicted': masks})
    if pred_df.shape[0] > 0:
        # sort according to the given order and save to a csv file
        subm = subm.merge(pred_df, on=['id', 'class'])
    subm.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args.input, args.model_dir)
