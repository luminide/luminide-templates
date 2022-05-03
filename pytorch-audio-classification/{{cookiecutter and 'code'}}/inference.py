import os
from glob import glob
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.utils.data as data

from util import get_class_names, make_test_augmenters
from dataset import AudioDataset
from models import ModelWrapper
from config import Config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on {device}')

def create_test_loader(conf, input_dir, class_names):
    test_audio_aug, test_image_aug = make_test_augmenters(conf)

    data_dir = 'test'
    test_df = pd.DataFrame()
    data_files = sorted(glob(f'{input_dir}/{data_dir}/*.*'))
    assert len(data_files) > 0, f'No files inside {input_dir}/{data_dir}'
    data_files = [os.path.basename(filename) for filename in data_files]
    test_df['filename'] = data_files
    test_df['labels'] = class_names[0]
    test_dataset = AudioDataset(
        test_df, conf, input_dir, data_dir,
        class_names, test_audio_aug, test_image_aug)
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

def predict_batch(outputs, threshold):
    sigmoid = nn.Sigmoid()
    # multi-label classification:
    # first, set prediction to 1 if probability >= threshold
    pred_batch = sigmoid(outputs).cpu().numpy() >= threshold
    num_rows = pred_batch.shape[0]
    # for each example, pick the label that was predicted as most likely
    max_inds = outputs.argmax(axis=1).cpu().numpy()
    # this is to make sure that at least one class is predicted
    pred_batch[range(num_rows), max_inds] = 1
    return pred_batch

def test(loader, model, num_classes, threshold):
    preds = np.zeros((len(loader.dataset), num_classes), dtype=np.float32)
    start_idx = 0
    model.eval()
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            end_idx = start_idx + outputs.shape[0]
            preds[start_idx:end_idx] = predict_batch(outputs, threshold)
            start_idx = end_idx
    return preds

def save_results(df, preds, class_names):
    pred_names = []
    for row in preds:
        pred_names.append(' '.join(class_names[np.where(row)]))
    df['labels'] = pred_names
    df.to_csv('submission.csv', index=False)
    print('Saved submission.csv')

def run(input_dir, model_dir, threshold):
    meta_file = os.path.join(input_dir, '{{ cookiecutter.train_metadata }}')
    train_df = pd.read_csv(meta_file, dtype=str)
    class_names = np.array(get_class_names(train_df))
    num_classes = len(class_names)

    model, conf = create_model(model_dir, num_classes)
    loader, df = create_test_loader(conf, input_dir, class_names)
    preds = test(loader, model, num_classes, threshold)
    save_results(df, preds, class_names)

if __name__ == '__main__':
    run('../input', './', 0.5)

