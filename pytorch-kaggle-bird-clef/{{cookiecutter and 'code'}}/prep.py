import os
import json
import pandas as pd

def prep_metadata(input_dir):
    meta_file = f'{input_dir}/{{ cookiecutter.train_metadata }}'
    assert os.path.exists(meta_file), f'{meta_file} not found on Compute Server'
    print('generating train.csv...')
    df = pd.read_csv(meta_file)
    result = pd.DataFrame()
    result['files'] = df['{{ cookiecutter.file_column }}']
    result['labels'] = df['{{ cookiecutter.label_column }}']
    result.to_csv(f'train.csv', index=False)

