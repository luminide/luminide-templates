import os
import argparse
import logging
import random
import multiprocessing as mp
import numpy as np
import pandas as pd
import yaml

import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
import torch.utils.data as data
{%- if cookiecutter.AMP == "True" %}
from torch.cuda.amp import GradScaler, autocast
{%- endif %}

from augment import make_augmenters
from dataset import VisionDataset
from models import ModelWrapper
from config import Config
from report import plot_images


NUM_CLASSES = {{ cookiecutter.num_classes }}

logging.basicConfig(
    level=logging.INFO,
    format=('%(asctime)s - %(name)s - ' '%(levelname)s -  %(message)s'))
logger = logging.getLogger('main')
parser = argparse.ArgumentParser(description='Pattern recognition')
parser.add_argument(
    '-j', '--num-workers', default=mp.cpu_count(), type=int, metavar='N',
    help='number of data loading workers')
parser.add_argument(
    '--epochs', default=50, type=int, metavar='N',
    help='number of total epochs to run')
parser.add_argument(
    '--seed', default=None, type=int,
    help='seed for initializing the random number generator')
parser.add_argument(
    '--resume', default='', type=str, metavar='PATH',
    help='path to saved model')
parser.add_argument(
    '-q', '--quick', dest='quick', action='store_true',
    help='use a subset of the data')
parser.add_argument(
    '--input', default='../input', metavar='DIR', help='input directory')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info('Running on %s', device)

class Trainer:
    def __init__(self, model, conf, input_dir, device, num_workers, quick=False):
        self.model = model
        self.conf = conf
        self.input_dir = input_dir
        self.device = device
        self.optimizer = torch.optim.SGD(
            model.parameters(), conf.lr, conf.momentum, conf.nesterov)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=conf.gamma)
{%- if cookiecutter.AMP == "True" %}
        self.scaler = GradScaler()
{%- endif %}

        # data loading code
        self.create_dataloaders(num_workers, quick)

    def create_dataloaders(self, num_workers, quick):
        conf = self.conf
        meta_file = os.path.join(self.input_dir, '{{ cookiecutter.train_metadata }}')
        meta_df = pd.read_csv(meta_file)
        # shuffle
        meta_df = meta_df.sample(frac=1, random_state=0).reset_index(drop=True)
        train_aug, test_aug = make_augmenters(conf)

        # split into train and validation sets
        split = meta_df.shape[0]*80//100
        train_df = meta_df.iloc[:split].reset_index(drop=True)
        val_df = meta_df.iloc[split:].reset_index(drop=True)
        train_dataset = VisionDataset(
            train_df, conf, self.input_dir, '{{ cookiecutter.train_image_dir }}',
            NUM_CLASSES, train_aug, training=True, quick=quick)
        print(f'{len(train_dataset)} examples in training set')
        drop_last = True if len(train_dataset) % conf.batch_size == 1 else False
        # FIXME: set pin_memory to True when spurious warnings are fixed in pytorch
        self.train_loader = data.DataLoader(
            train_dataset, batch_size=conf.batch_size,
            num_workers=num_workers, pin_memory=False,
            worker_init_fn=worker_init_fn, drop_last=drop_last)

    def fit(self, epochs):
        best_loss = None
        history = []
        writer = SummaryWriter()
        for epoch in range(epochs):
            # train for one epoch
            train_loss, val_loss = self.train_epoch(epoch, history)
            self.scheduler.step()
            writer.add_scalar('training loss', train_loss, epoch)
            writer.add_scalar('validation loss', val_loss, epoch)
            writer.flush()
            print(
                f'Epoch {epoch + 1}: training loss {train_loss:.4f} '
                f' validation loss {val_loss:.4f}')
            if best_loss is None or val_loss < best_loss:
                best_loss = val_loss
                state = {
                    'epoch': epoch, 'model': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                    'conf': self.conf.get()
                }
                torch.save(state, 'model.pth')

        df = pd.DataFrame(history, columns=['epoch', 'iter', 'train_loss', 'val_loss'])
        df.to_csv('history.csv')
        writer.close()
        self.make_report(self.train_loader)

    def train_epoch(self, epoch, history):
        loss_func = nn.BCEWithLogitsLoss()
        model = self.model
        optimizer = self.optimizer
{%- if cookiecutter.AMP == "True" %}
        scaler = self.scaler
{%- endif %}
        # skip bprop on some minibatches
        val_percent = 10
        skip_interval = 100 // val_percent
        loss_sums = [0, 0]
        batch_counts = [0, 0]
        model.train()
        for i, (images, labels) in enumerate(self.train_loader):
            images = images.to(device)
            labels = labels.to(device)
            # compute output
{%- if cookiecutter.AMP == "True" %}
            # use AMP
            with autocast():
                outputs = model(images)
                loss = loss_func(outputs, labels)
{%- elif cookiecutter.AMP == "False" %}
            outputs = model(images)
            loss = loss_func(outputs, labels)
{%- endif %}

            if i % skip_interval == 0:
                # skip this minibatch while tuning
                history.append([epoch, i, np.nan, loss.item()])
                loss_sums[1] += loss.item()
                batch_counts[1] += 1
                continue

            history.append([epoch, i, loss.item(), np.nan])
            loss_sums[0] += loss.item()
            batch_counts[0] += 1
            # compute gradient and do SGD step
{%- if cookiecutter.AMP == "True" %}
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
{%- elif cookiecutter.AMP == "False" %}
            loss.backward()
            optimizer.step()
{%- endif %}
            optimizer.zero_grad()

        train_loss = loss_sums[0]/batch_counts[0]
        val_loss = loss_sums[1]/batch_counts[1]
        return train_loss, val_loss

    def make_report(self, loader):
        images, labels = iter(loader).next()
        with torch.no_grad():
            outputs = self.model(images.to(device)).cpu()
        plot_images(images, labels, outputs, self.input_dir)


def worker_init_fn(worker_id):
    random.seed(random.randint(0, 2**32) + worker_id)
    np.random.seed(random.randint(0, 2**32) + worker_id)


def main(args_list):
    if args_list is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args=args_list)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    input_dir = args.input
    model_file = args.resume
    if model_file:
        logger.info('Loading model from %s', model_file)
        checkpoint = torch.load(model_file)
        conf = Config(checkpoint['conf'])
    else:
        conf = Config()

    conf_file = 'config.yaml'
    if os.path.exists(conf_file):
        logger.info('Updating config from %s', conf_file)
        # read in hyperparameter values
        with open(conf_file) as fd:
            updates = yaml.safe_load(fd)
        conf.update(updates)
    else:
        logger.info('Using default config')
    logger.info(conf.get())
    model = ModelWrapper(NUM_CLASSES, conf)
    model = model.to(device)
    if model_file:
        model.load_state_dict(checkpoint['model'])

    trainer = Trainer(
        model, conf, input_dir, device, args.num_workers, quick=args.quick)
    trainer.fit(args.epochs)

if __name__ == '__main__':
    main(None)
    logger.info('Done')
