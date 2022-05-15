import os
import json
import pickle
import librosa
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

def create_test_loader(conf, df, input_dir, class_names):
    test_duration = 5
    max_clips = 10
    data_dir = 'train_audio'
    test_audio_aug, test_image_aug = make_test_augmenters(conf)

    filenames = df['{{ cookiecutter.file_column }}']
    labels = df['{{ cookiecutter.label_column }}'] + ' ' + [' '.join(row) for row in df['secondary_labels'].apply(eval)]
    labels = [label.strip() for label in labels]
    clip_names = []
    clip_offsets = []
    for filename, label in zip(filenames, labels):
        duration = librosa.get_duration(filename=f'{input_dir}/{data_dir}/{filename}')
        num_clips = min(max_clips, max(1, int(duration)//5))
        for idx in range(num_clips):
            clip_names.append(filename)
            clip_offsets.append(idx*test_duration)

    test_df = pd.DataFrame()
    test_df['{{ cookiecutter.file_column }}'] = clip_names
    test_df['{{ cookiecutter.label_column }}'] = class_names[0]
    test_df['secondary_labels'] = '[]'
    test_df['offsets'] = clip_offsets

    test_dataset = AudioDataset(
        test_df, conf, input_dir, data_dir,
        class_names, test_audio_aug, test_image_aug, is_stage2=True)
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
    # return raw probability values
    return sigmoid(outputs).cpu().numpy()

def test(loader, model, num_classes, threshold):
    preds = np.zeros((len(loader.dataset), num_classes), dtype=np.float32)
    start_idx = 0
    model.eval()
    total = len(loader)
    with torch.no_grad():
        for i, (inputs, _) in enumerate(loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            end_idx = start_idx + outputs.shape[0]
            preds[start_idx:end_idx] = predict_batch(outputs, threshold)
            start_idx = end_idx
            if i%100 == 0:
                perc = i*100/total
                print(f'{perc:.2f}% done')
    return preds

def save_results1(input_dir, train_df, test_df, preds, class_names):
    class_map = {name: i for i, name in enumerate(class_names)}
    results_df = test_df.copy()
    results_df['label_strength'] = 0.0
    for idx in range(test_df.shape[0]):
        label = results_df.iloc[idx]['{{ cookiecutter.label_column }}']
        class_id = class_map[label]
        results_df['label_strength'].iloc[idx] = preds[idx][class_id]
    results_df.to_csv('results.csv', index=False)
    print('Saved results.csv')
    offsets_df = pd.DataFrame()
    offsets_df['{{ cookiecutter.file_column }}'] = train_df['{{ cookiecutter.file_column }}']
    offsets_df['best_offset'] = results_df['offsets'].iloc[results_df.groupby(['{{ cookiecutter.file_column }}'])['label_strength'].idxmax()].values
    offsets_df.to_csv('offsets.csv', index=False)
    print('Saved offsets.csv')

def save_results(input_dir, train_df, test_df, preds, class_names):
    results = {filename: [] for filename in train_df['{{ cookiecutter.file_column }}']}
    for idx in range(test_df.shape[0]):
        filename =  test_df['{{ cookiecutter.file_column }}'].iloc[idx]
        results[filename].append(preds[idx])
    with open('pseudo-labels.pkl', 'wb') as fd:
        pickle.dump(results, fd)

def run(input_dir, model_dir, threshold=0.3):
    meta_file = os.path.join(input_dir, '{{ cookiecutter.train_metadata }}')
    train_df = pd.read_csv(meta_file, dtype=str)
    class_names = np.array(get_class_names(train_df))
    num_classes = len(class_names)

    model, conf = create_model(model_dir, num_classes)
    loader, test_df = create_test_loader(conf, train_df, input_dir, class_names)
    preds = test(loader, model, num_classes, threshold)
    save_results(input_dir, train_df, test_df, preds, class_names)

if __name__ == '__main__':
    run('../input', './')
