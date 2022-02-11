import os
import numpy as np
import pandas as pd
from glob import glob
import multiprocessing as mp
import torch
from torch import nn
import torch.utils.data as data

from util import get_class_names, make_test_augmenter
from dataset import VisionDataset
from models import ModelWrapper
from config import Config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on {device}')

def create_test_loader(conf, input_dir, class_names):
    test_aug = make_test_augmenter(conf)

    image_dir = 'test_images'
    test_df = pd.DataFrame()
    image_files = sorted(glob(f'{input_dir}/{image_dir}/*.*'))
    assert len(image_files) > 0, f'No files inside {input_dir}/{image_dir}'
    image_files = [os.path.basename(filename) for filename in image_files]
    test_df['image'] = image_files
    test_df['labels'] = class_names[0]
    test_dataset = VisionDataset(
        test_df, conf, input_dir, image_dir,
        class_names, test_aug, training=False)
    print(f'{len(test_dataset)} examples in test set')
    loader = data.DataLoader(
        test_dataset, batch_size=conf.batch_size, shuffle=False,
        num_workers=mp.cpu_count(), pin_memory=True)
    return loader, test_df

def create_model(model_dir, num_classes):
    checkpoint = torch.load(f'{model_dir}/model.pth')
    conf = Config(checkpoint['conf'])
    conf.pretrained = False
    model = ModelWrapper(conf, num_classes)
    model = model.to(device)
    model.load_state_dict(checkpoint['model'])
    return model, conf

def test(loader, model, num_classes):
    sigmoid = nn.Sigmoid()
    preds = np.zeros((len(loader.dataset), num_classes), dtype=np.float32)
    start_idx = 0
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            end_idx = start_idx + outputs.shape[0]
            preds[start_idx:end_idx] = sigmoid(outputs).round().cpu().numpy()
            start_idx = end_idx
    return preds

def save_results(df, preds, class_names):
    pred_names = []
    for row in preds:
        pred_names.append(' '.join(class_names[np.where(row)]))
    df['labels'] = pred_names
    df.to_csv('submission.csv', index=False)
    print('Saved submission.csv')

def run(input_dir, model_dir):
    meta_file = os.path.join(input_dir, '{{ cookiecutter.train_metadata }}')
    train_df = pd.read_csv(meta_file, dtype=str)
    class_names = np.array(get_class_names(train_df))
    num_classes = len(class_names)

    model, conf = create_model(model_dir, num_classes)
    loader, df = create_test_loader(conf, input_dir, class_names)
    preds = test(loader, model, num_classes)
    save_results(df, preds, class_names)

if __name__ == '__main__':
    run('../input', './')

