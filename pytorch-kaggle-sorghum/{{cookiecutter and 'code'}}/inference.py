import os
import argparse
from glob import glob
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.utils.data as data

from util import get_class_names, make_test_augmenter, save_dist
from dataset import VisionDataset
from models import ModelWrapper
from config import Config

parser = argparse.ArgumentParser()
parser.add_argument(
    '-n', '--num-patches', default=9, type=int, metavar='N',
    help='number of patches per image')
parser.add_argument(
    '-e', '--ensemble', dest='ensemble', action='store_true',
    help='ensemble predictions')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on {device}')

def create_test_loader(conf, input_dir, class_names):
    test_aug = make_test_augmenter(conf)

    image_dir = '{{ cookiecutter.test_image_dir }}'
    test_df = pd.DataFrame()
    image_files = sorted(glob(f'{input_dir}/{image_dir}/*.*'))
    assert len(image_files) > 0, f'No files inside {input_dir}/{image_dir}'
    image_files = [os.path.basename(filename) for filename in image_files]
    test_df['{{ cookiecutter.image_column }}'] = image_files
    test_df['{{ cookiecutter.label_column }}'] = class_names[0]
    test_dataset = VisionDataset(
        test_df, conf, input_dir, image_dir,
        class_names, test_aug)
    print(f'{len(test_dataset)} examples in test set')
    loader = data.DataLoader(
        test_dataset, batch_size=conf.batch_size, shuffle=False,
        num_workers=mp.cpu_count(), pin_memory=False)
    return loader, test_df

def create_model(model_dir, num_classes):
    checkpoint = torch.load(f'{model_dir}/model.pth', map_location=device)
    conf = Config(checkpoint['conf'])
    conf.pretrained = False
    model = ModelWrapper(conf, num_classes)
    model = model.to(device)
    model.load_state_dict(checkpoint['model'])
    return model, conf

def test(loader, model, num_classes, num_patches):
    preds = np.zeros((len(loader.dataset), num_classes), dtype=np.float32)
    start_idx = 0
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            pred_batch = outputs.cpu().numpy()
            num_rows = pred_batch.shape[0]
            end_idx = start_idx + num_rows
            preds[start_idx:end_idx] = pred_batch
            start_idx = end_idx

    # average predictions from patches
    assert preds.shape[0] % num_patches == 0
    preds = preds.reshape((preds.shape[0]//num_patches, num_patches, -1))
    return preds.mean(axis=1)

def collapse(filenames, num_patches):
    filenames = filenames.iloc[::num_patches]
    # rename from xxx_x.jpg to xxx.png
    return [f'{fn[:-6]}.png' for fn in filenames]

def save_results(df, preds, class_names, num_patches):
    pred_names = [class_names[int(pred)] for pred in preds]
    results = pd.DataFrame()
    results['filename'] = collapse(df['{{ cookiecutter.image_column }}'], num_patches)
    results['{{ cookiecutter.label_column }}'] = pred_names
    results.to_csv('submission.csv', index=False)
    print('Saved submission.csv')

    dist_file = 'predicted-distribution.png'
    save_dist(results['{{ cookiecutter.label_column }}'].value_counts(), dist_file)
    print(f'\nSaved predicted class distribution to {dist_file}')

def ensemble():
    pred_files = sorted(glob('*.npy'))
    if len(pred_files) == 0:
        print('error: no npy files found')
        return None
    basenames = [os.path.basename(f) for f in pred_files]
    print(f'ensembling predictions: {basenames}...')
    preds_list = []
    for filename in pred_files:
        preds_list.append(np.load(filename))
    return np.stack(preds_list).mean(axis=0)

def run(args, input_dir, model_dir):
    meta_file = os.path.join(input_dir, '{{ cookiecutter.train_metadata }}')
    train_df = pd.read_csv(meta_file, dtype=str)
    train_df.dropna(inplace=True)
    class_names = np.array(get_class_names(train_df))
    num_classes = len(class_names)

    model, conf = create_model(model_dir, num_classes)
    loader, df = create_test_loader(conf, input_dir, class_names)
    if args.ensemble:
        preds = ensemble()
        if preds is None:
            return
    else:
        print('running inference on test set...')
        print(conf)
        preds = test(loader, model, num_classes, args.num_patches)
        i = 0
        while os.path.exists(f'preds{i}.npy'):
            i += 1
        np.save(f'preds{i}.npy', preds)
    pred_labels = preds.argmax(axis=1)
    save_results(df, pred_labels, class_names, args.num_patches)

if __name__ == '__main__':
    args = parser.parse_args()
    input_dir = '../input'
    run(args, input_dir, './')
