import os
import json
import pandas as pd

def prep_metadata(input_dir):
    meta_file = f'{input_dir}/{{ cookiecutter.train_metadata }}'
    assert os.path.exists(meta_file), f'{meta_file} not found on Compute Server'
    print('generating train.csv...')
    df = pd.read_csv(meta_file)
    if True:
        # train on scored birds only
        with open(f'{input_dir}/scored_birds.json') as json_file:
            birds = json.load(json_file)

        labels = []
        files = []
        for bird in birds:
            bird_files = df[df['{{ cookiecutter.label_column }}'] == bird]['{{ cookiecutter.file_column }}']
            files.extend(bird_files.tolist())
            labels.extend([bird]*len(bird_files))
        result = pd.DataFrame()
        result['files'] = files
        result['labels'] = labels
        result.to_csv(f'train.csv', index=False)

    else:
        # train on the whole dataset
        result = pd.DataFrame()
        result['files'] = df['{{ cookiecutter.file_column }}']
        result['labels'] = df['{{ cookiecutter.label_column }}']
        result.to_csv(f'train.csv', index=False)

