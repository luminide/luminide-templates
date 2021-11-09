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
    '-p', '--print-interval', default=200, type=int,
    metavar='N', help='print interval in batches (default: 200)')
parser.add_argument(
    '--seed', default=0, type=int,
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
    def __init__(
            self, model, conf, input_dir, device, num_workers,
            checkpoint, print_interval, quick=False):
        self.model = model
        self.conf = conf
        self.input_dir = input_dir
        self.device = device
        self.print_interval = print_interval
        self.optimizer = torch.optim.SGD(
            model.parameters(), conf.lr, conf.momentum, conf.nesterov)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=conf.gamma)
{%- if cookiecutter.AMP == "True" %}
        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler()
{%- endif %}

        if checkpoint:
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

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
        val_dataset = VisionDataset(
            val_df, conf, self.input_dir, '{{ cookiecutter.train_image_dir }}',
            NUM_CLASSES, test_aug, training=False)
        print(f'{len(train_dataset)} examples in training set')
        print(f'{len(val_dataset)} examples in validation set')
        drop_last = True if len(train_dataset) % conf.batch_size == 1 else False
        # FIXME: set pin_memory to True when spurious warnings are fixed in pytorch
        self.train_loader = data.DataLoader(
            train_dataset, batch_size=conf.batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=False,
            worker_init_fn=worker_init_fn, drop_last=drop_last)
        self.val_loader = data.DataLoader(
            val_dataset, batch_size=conf.batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=False)

    def fit(self, epochs):
        best_loss = None
        history = []
        writer = SummaryWriter()
        for epoch in range(epochs):
            # train for one epoch
            train_loss = self.train_epoch(epoch, history)
            val_loss, val_acc = self.val_epoch()
            self.scheduler.step()
            writer.add_scalar('training loss', train_loss, epoch)
            writer.add_scalar('validation loss', val_loss, epoch)
            writer.flush()
            print(
                f'Epoch {epoch + 1}: training loss {train_loss:.4f} '
                f' validation loss {val_loss:.4f} validation accuracy {val_acc:.2f}%')
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

        val_iter = iter(self.val_loader)
        val_interval = len(self.train_loader)//len(self.val_loader)
        train_loss_list = []
        val_loss_list = []
        model.train()
        for step, (images, labels) in enumerate(self.train_loader):
            if (step + 1) % val_interval == 0:
                model.eval()
                # collect validation history for tuning
                try:
                    with torch.no_grad():
                        val_images, val_labels = next(val_iter)
                        val_images = val_images.to(device)
                        val_labels = val_labels.to(device)
{%- if cookiecutter.AMP == "True" %}
                        with autocast(enabled=self.use_amp):
                            val_outputs = model(val_images)
{%- elif cookiecutter.AMP == "False" %}
                        val_outputs = model(val_images)
{%- endif %}
                        val_loss = loss_func(val_outputs, val_labels)
                        history.append([epoch, step, np.nan, val_loss.item()])
                        val_loss_list.append(val_loss.item())
                except StopIteration:
                    pass
                # switch back to training mode
                model.train()

            images = images.to(device)
            labels = labels.to(device)
            # compute output
{%- if cookiecutter.AMP == "True" %}
            # use AMP
            with autocast(enabled=self.use_amp):
                outputs = model(images)
                loss = loss_func(outputs, labels)
{%- elif cookiecutter.AMP == "False" %}
            outputs = model(images)
            loss = loss_func(outputs, labels)
{%- endif %}

            train_loss_list.append(loss.item())
            history.append([epoch, step, loss.item(), np.nan])
            if (step + 1) % self.print_interval == 0:
                print(f'batch {step + 1}: training loss {loss.item():.4f}')
            # compute gradient and do SGD step
{%- if cookiecutter.AMP == "True" %}
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()
{%- elif cookiecutter.AMP == "False" %}
            loss.backward()
            optimizer.step()
{%- endif %}
            optimizer.zero_grad()

        mean_train_loss = np.array(train_loss_list).mean()
        return mean_train_loss

    def val_epoch(self):
        loss_func = nn.BCEWithLogitsLoss()
        losses = []
        correct = 0
        self.model.eval()
        for images, labels in self.val_loader:
            images = images.to(device)
            labels = labels.to(device)
{%- if cookiecutter.AMP == "True" %}
            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
{%- elif cookiecutter.AMP == "False" %}
            outputs = self.model(images)
{%- endif %}
            preds = outputs.argmax(axis=1)
            correct += (preds == labels[:, 1]).sum()
            losses.append(loss_func(outputs, labels).item())
        accuracy = correct*100./len(self.val_loader.dataset)
        return np.mean(losses), accuracy

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
        checkpoint = None
        conf = Config()

    logger.info(conf.get())
    model = ModelWrapper(NUM_CLASSES, conf)
    model = model.to(device)

    trainer = Trainer(
        model, conf, input_dir, device, args.num_workers,
        checkpoint, args.print_interval, quick=args.quick)
    trainer.fit(args.epochs)

if __name__ == '__main__':
    main(None)
    logger.info('Done')
