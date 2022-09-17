# {{ cookiecutter.project_description }}

### Introduction
This repository contains source code generated by [Luminide](https://luminide.com). It may be used to train, validate and tune deep learning models for image classification. The following directory structure is assumed:
```
├── code (source code)
├── input (dataset)
└── output (working directory)
```

The dataset should have images inside a directory named `{{ cookiecutter.train_image_dir }}` and a CSV file named `{{ cookiecutter.train_metadata }}`. An example is shown below:

```
input
├── {{ cookiecutter.train_metadata }}
└── {{ cookiecutter.train_image_dir }}
    ├── 800113bb65efe69e.jpg
    ├── 8002cb321f8bfcdf.jpg
    ├── 80070f7fb5e2ccaa.jpg
```

The CSV file is expected to have labels under a column named `{{ cookiecutter.label_column }}` as in the example below:

```
{{ cookiecutter.image_column }},{{ cookiecutter.label_column }}
800113bb65efe69e.jpg,healthy
8002cb321f8bfcdf.jpg,scab frog_eye_leaf_spot complex
80070f7fb5e2ccaa.jpg,scab
```
If an item has multiple labels, they should be separated by a space character as shown.

### Using this repo with Luminide
- Configure your [Kaggle API token](https://github.com/Kaggle/kaggle-api) on the `Import Data` tab.
- Attach a Compute Server with a GPU (e.g. gcp-t4).
- On the `Import Data` data tab, choose Kaggle and then enter `yasufuminakama/mayo-train-images-size1024-n16` (User Dataset).
- For exploratory analysis, run [eda.ipynb](eda.ipynb).
- To train, use the `Run Experiment` menu.
- To monitor training progress, use the `Experiment Visualization` menu.
- To generate a report on the most recent training session, run report.sh from the `Run Experiment` tab. Make sure `Track Experiment` is checked. The results will be copied back to a file called `report.html`.
- To tune the hyperparameters, edit [sweep.yaml](sweep.yaml) as desired and launch a sweep from the `Run Experiment` tab. Tuned values will be copied back to a file called `config-tuned.yaml` along with visualizations in `sweep-results.html`.
- After an experiment is complete, use the file browser on the IDE interface to access the results on the IDE Server.
- Use the `Experiment Tracking` menu to track experiments.
- Run [kaggle.sh](kaggle.sh) as a custom experiment to upload the code to Kaggle.
- To create a submission, copy [kaggle.ipynb](kaggle.ipynb) to a new Kaggle notebook.
- Add the notebook output of `https://www.kaggle.com/luminide/wheels1` as Data.
- Add your dataset at `https://www.kaggle.com/<kaggle_username>/kagglecode` as Data.
- Add the relevant competition dataset as Data.
- Run the notebook after turning off the `Internet` setting.

{%- if cookiecutter.data_subset_percentage != '100' %}

Note: As configured, the code trains on {{ cookiecutter.data_subset_percentage }}% of the data. To train on the entire dataset, edit `full.sh` and `fast.sh` to remove the `--subset` command line parameter so that the default value of 100 is used.
{%- endif %}


For more details on usage, see [Luminide documentation](https://luminide.readthedocs.io)