import os
from glob import glob
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.utils.data as data

from util import get_class_names, make_test_augmenter
from dataset import VisionDataset
from models import SelfSupervisedModel
from prep import save_dataset
from config import Config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on {device}')


def create_test_loader(conf, input_dir, class_names, test_df):
    test_aug = make_test_augmenter(conf)

    image_dir = 'test_images'
    test_df['label'] = class_names[0]
    test_dataset = VisionDataset(
        test_df, conf, input_dir, image_dir,
        class_names, test_aug)
    print(f'{len(test_dataset)} examples in test set')
    loader = data.DataLoader(
        test_dataset, batch_size=conf.batch_size, shuffle=False,
        num_workers=mp.cpu_count(), pin_memory=False)
    return loader

def create_model(model_dir, num_classes):
    checkpoint = torch.load(f'{model_dir}/model.pth', map_location=device)
    conf = Config(checkpoint['conf'])
    conf.pretrained = False
    model = SelfSupervisedModel(conf, num_classes)
    model = model.to(device)
    model.load_state_dict(checkpoint['model'])
    return model, conf

def test(loader, model, num_classes):
    act = nn.Sigmoid()
    preds = np.zeros((len(loader.dataset), num_classes), dtype=np.float32)
    start_idx = 0
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            # multi-label classification:
            pred_batch = act(model.bin_outputs).cpu().numpy()
            num_rows = pred_batch.shape[0]
            end_idx = start_idx + num_rows
            preds[start_idx:end_idx] = pred_batch
            start_idx = end_idx
    return preds

def save_results(input_dir, df, preds, class_names):
    subm = pd.DataFrame(columns=['patient_id', 'CE', 'LAA'])

    subm['patient_id'] = df['patient_id']
    subm['CE'] = preds[:, 0]
    subm['LAA'] = preds[:, 1]

    # group predictions according to patient IDs
    subm = subm.groupby(['patient_id']).mean().reset_index()
    subm.to_csv('submission.csv', index=False)
    print('Saved submission.csv')
    sample_subm = pd.read_csv(f'{input_dir}/sample_submission.csv')
    assert sample_subm.shape[0] == subm.shape[0]

def run(input_dir, model_dir):
    meta_file = os.path.join(input_dir, '{{ cookiecutter.train_metadata }}')
    train_df = pd.read_csv(meta_file, dtype=str)
    class_names = np.array(get_class_names(train_df))
    num_classes = len(class_names)

    model, conf = create_model(model_dir, num_classes)

    test_df = pd.read_csv(f'{input_dir}/test.csv')
    save_dataset(test_df, image_dir=f'{input_dir}/test')

    loader = create_test_loader(conf, './', class_names, test_df)
    preds = test(loader, model, num_classes)
    save_results(input_dir, test_df, preds, class_names)

if __name__ == '__main__':
    run('../input', './')

