import numpy as np
import torch
import pycocotools.mask as mask_util

import detectron2
from detectron2.data import DatasetCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator

from inference import get_mask


def dice_coef(pred, targ):
    ious = []
    for targ_item in targ:
        class_id = targ_item['category_id']
        img_mask = get_mask(pred, class_id)
        if img_mask is not None:
            enc_targ = targ_item['segmentation']
            targ_mask = mask_util.decode(enc_targ)
            iou = (targ_mask*img_mask).sum()/(targ_mask.sum() + img_mask.sum() + 1e-6)
            ious.append(iou)
        else:
            ious.append(0)
    if len(ious) == 0:
        return 0
    return 2*np.mean(ious)

class DiceEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, thresh=0.5):
        self.thresh = thresh
        self.dataset_name = dataset_name
        self.dataset_dicts = DatasetCatalog.get(dataset_name)
        self.annotations_cache = {item['image_id']:item['annotations'] for item in self.dataset_dicts}

    def reset(self):
        self.scores = []

    def process(self, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            if len(out['instances']) == 0:
                self.scores.append(0)
            else:
                targ = self.annotations_cache[inp['image_id']]
                res = dice_coef(out, targ)
                self.scores.append(res)

    def evaluate(self):
        score = round(np.mean(self.scores), 4)
        return {f'{self.dataset_name} Dice coeff': score}
