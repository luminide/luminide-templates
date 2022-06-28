import os
import cv2
from glob import glob
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.utils.data as data
from monai.data import Dataset as MonaiDataset
from monai.data import DataLoader
from monai.inferers import sliding_window_inference

from util import get_class_names, make_val_augmenter, make_test_augmenter_3d, get_id
from dataset import VisionDataset
from models import ModelWrapper
from config import Config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on {device}')

def create_test_loader(conf, input_dir, class_names):
    test_aug = make_val_augmenter(conf)
    test_df = pd.DataFrame()
    img_dir = 'test'
    img_files = sorted(glob(f'{input_dir}/{img_dir}/**/*.{{ cookiecutter.file_extension }}', recursive=True))

    # delete common prefix from paths
    img_files = [f.replace(f'{input_dir}/{img_dir}/', '') for f in img_files]

    test_df['img_files'] = img_files
    test_df['study'] = test_df['img_files'].apply(lambda x: x.split('/')[-3])
    # sort files according to studies
    studies = [group['img_files'] for _, group in test_df.groupby('study')]
    new_img_files = []
    for study in studies:
        new_img_files.extend(study)
    test_df['img_files'] = new_img_files
    test_df['study'] = test_df['img_files'].apply(lambda x: x.split('/')[-3])
    test_dataset = VisionDataset(
        test_df, conf, input_dir, img_dir,
        class_names, test_aug, is_test=True)
    print(f'{len(test_dataset)} examples in test set')
    loader = data.DataLoader(
        test_dataset, batch_size=conf.batch_size, shuffle=False,
        num_workers=mp.cpu_count(), pin_memory=False)
    return loader, test_df

def create_test_loader_3d(conf, input_dir, class_names):
    test_df = pd.DataFrame()
    img_dir = 'test'
    img_files = sorted(glob(f'{input_dir}/{img_dir}/**/*.{{ cookiecutter.file_extension }}', recursive=True))

    # delete common prefix from paths
    #img_files = [f.replace(f'{input_dir}/{img_dir}/', '') for f in img_files]

    test_df['img_files'] = img_files
    test_df['study'] = test_df['img_files'].apply(lambda x: x.split('/')[-3])
    studies = [group['img_files'] for _, group in test_df.groupby('study')]
    data_dicts = [{'img' : list(study)} for study in studies]

    test_aug = make_test_augmenter_3d(conf)
    test_dataset = MonaiDataset(data=data_dicts, transform=test_aug)
    print(f'{len(test_dataset)} examples in test set')
    loader = DataLoader(
        test_dataset, batch_size=conf.batch_size, shuffle=False,
        num_workers=mp.cpu_count(), pin_memory=False)
    return loader, test_df

def create_model(model_file, num_classes):
    checkpoint = torch.load(model_file, map_location=device)
    conf = Config(checkpoint['conf'])
    conf.pretrained = False
    model = ModelWrapper(conf, num_classes)
    model = model.to(device)
    model.load_state_dict(checkpoint['model'])
    return model, conf

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

def get_img_shape(filename):
    basename = os.path.basename(filename)
    tokens = basename.split('_')
    height, width = int(tokens[3]), int(tokens[2])
    return (height, width)

def pad_mask(conf, mask):
    # pad image to conf.image_size
    num_channels, height, width = mask.shape
    padded = np.zeros((num_channels, conf.image_size, conf.image_size), dtype=mask.dtype)
    dh = conf.image_size - height
    dw = conf.image_size - width

    top = dh//2
    left = dw//2
    padded[:, top:top + height, left:left + width] = mask
    return padded

def resize_mask(mask, height, width):
    mask = mask.transpose((1, 2, 0))
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask.transpose((2, 0, 1))

def test(confs, loaders, models, df, class_names, thresh):
    ids = []
    classes = []
    masks = []
    sigmoid = nn.Sigmoid()
    num_classes = len(class_names)
    for model in models:
        model.eval()
    iters = [iter(loader) for loader in loaders]
    studies = df.groupby('study')
    sw_batch_size = 4
    flip_dims = [[], [2], [3], [2, 3]]
    overlap = 0.5
    with torch.no_grad():
        for _, study in studies:
            study_len = len(study['img_files'])
            print('study', study['study'].unique())
            for i, it in enumerate(iters):
                conf = confs[i]
                model = models[i]
                if '3D' in conf.arch:
                    is_3d = True
                    roi_size = (conf.test_roi, conf.test_roi, conf.test_depth)
                else:
                    is_3d = False
                    roi_size = (conf.test_roi, conf.test_roi)
                study_preds = []
                pred_len = 0
                while True:
                    batch = it.next()
                    images = batch['img'].to(device)
                    if is_3d:
                        # convert from CDWH to CHWD
                        images = images.permute((0, 3, 2, 1)).unsqueeze(0)
                    outputs = None
                    for tta in flip_dims:
                        tta_images = torch.flip(images, tta)
                        tta_outputs = sliding_window_inference(
                            tta_images, roi_size, sw_batch_size, model,
                            mode='gaussian', overlap=overlap)
                        tta_outputs = torch.flip(tta_outputs, tta)
                        if outputs == None:
                            outputs = tta_outputs
                        else:
                            outputs += tta_outputs
                    outputs /= len(flip_dims)
                    pred = sigmoid(outputs).cpu().numpy()[0]
                    study_preds.append(pred)
                    if is_3d:
                        assert pred.shape[3] == len(study['img_files'])
                        break
                    pred_len += 1
                    if pred_len == study_len:
                        break
                if len(study_preds) > 1:
                    assert not is_3d
                    study_preds = np.stack(study_preds, 3)
                else:
                    assert is_3d
                    study_preds = study_preds[0]
                if i == 0:
                    mean_pred = study_preds
                else:
                    mean_pred += study_preds
            mean_pred /= len(iters)
            assert mean_pred.shape[3] == study_len
            for slice_idx, img_file in enumerate(study['img_files']):
                height, width = get_img_shape(img_file)
                img_id = get_id(img_file)
                for class_id, class_name in enumerate(class_names):
                    mask = mean_pred[class_id, :, :, slice_idx]
                    assert mask.shape[0] == height
                    assert mask.shape[1] == width
                    mask[mask >= thresh] = 1
                    mask[mask < thresh] = 0
                    enc_mask = '' if mask.sum() == 0 else rle_encode(mask)
                    ids.append(img_id)
                    classes.append(class_name)
                    masks.append(enc_mask)
    return ids, classes, masks

def save_results(input_dir, ids, classes, masks):
    pred_df = pd.DataFrame({'id': ids, 'class': classes, 'predicted': masks})
    subm = pd.read_csv(f'{input_dir}/sample_submission.csv')
    del subm['predicted']

    if pred_df.shape[0] > 0:
        # sort according to the given order and save to a csv file
        subm = subm.merge(pred_df, on=['id', 'class'])
    subm.to_csv('submission.csv', index=False)

def run(input_dir, model_files, thresh):
    meta_file = os.path.join(input_dir, '{{ cookiecutter.train_metadata }}')
    train_df = pd.read_csv(meta_file, dtype=str)
    class_names = np.array(get_class_names(train_df))
    num_classes = len(class_names)
    batch_size = 1

    models = []
    confs = []
    loaders = []
    for i, model_file in enumerate(model_files):
        print(model_file)
        model, conf = create_model(model_file, num_classes)
        print(conf)
        conf['batch_size'] = batch_size
        if '3D' in conf.arch:
            loader, df = create_test_loader_3d(conf, input_dir, class_names)
        else:
            loader, df = create_test_loader(conf, input_dir, class_names)
        models.append(model)
        confs.append(conf)
        loaders.append(loader)
    # average predictions from multiple models
    ids, classes, masks = test(confs, loaders, models, df, class_names, thresh)
    save_results(input_dir, ids, classes, masks)

if __name__ == '__main__':
    test_thresh = 0.5
    model_dir = './'
    model_files = sorted(glob(f'{model_dir}/model*.pth'))
    run('../input', model_files, test_thresh)
