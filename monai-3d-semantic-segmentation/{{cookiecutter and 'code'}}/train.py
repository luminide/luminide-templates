import os
import argparse
import random
import re
import multiprocessing as mp
from datetime import datetime
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
from glob import glob

import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
import torch.utils.data as data
{%- if cookiecutter.AMP == "True" %}
from torch import autocast
from torch.cuda.amp import GradScaler
{%- endif %}
import monai
from monai.data import DataLoader
from monai.inferers import sliding_window_inference

from augment import make_train_augmenter
from dataset import create_dataset
from models import ModelWrapper
from config import Config
import util

import warnings
warnings.filterwarnings("ignore")

# workaround for https://github.com/Project-MONAI/MONAI/issues/701
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

parser = argparse.ArgumentParser()
parser.add_argument(
    '-j', '--num-workers', default=mp.cpu_count(), type=int, metavar='N',
    help='number of data loading workers')
parser.add_argument(
    '--epochs', default=40, type=int, metavar='N',
    help='number of total epochs to run')
parser.add_argument(
    '-p', '--print-interval', default=100, type=int, metavar='N',
    help='print-interval in batches')
parser.add_argument(
    '--seed', default=0, type=int,
    help='seed for initializing the random number generator')
parser.add_argument(
    '--resume', default='', type=str, metavar='PATH',
    help='path to saved model')
parser.add_argument(
    '--validate', default='', type=str, metavar='PATH',
    help='path to saved model to validate')
parser.add_argument(
    '-s', '--subset', default=100, type=int, metavar='N',
    help='use a percentage of the data for training and validation')
parser.add_argument(
    '--input', default='../input', metavar='DIR',
    help='input directory')

device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_type)

class Trainer:
    def __init__(
            self, conf, input_dir, device, num_workers,
            checkpoint, print_interval=100, subset=100):
        self.conf = conf
        self.input_dir = input_dir
        self.device = device
        self.max_patience = 200
        self.print_interval = print_interval
{%- if cookiecutter.AMP == "True" %}
        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler()
{%- endif %}

        self.create_dataloaders(num_workers, subset, random_split=False)

        self.model = ModelWrapper(conf, self.num_classes)
        self.model = self.model.to(device)
        self.optimizer = self.create_optimizer(conf, self.model)
        assert  self.optimizer is not None, f'Unknown optimizer {conf.optim}'
        if checkpoint:
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        num_samples = len(self.train_loader.dataset)
        restart_epoch = conf.restart_epoch[conf.arch]
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=restart_epoch*(num_samples//conf.batch_size),
            T_mult=1, eta_min=conf.min_lr,
        )

        self.loss_funcs = [
            smp.losses.SoftBCEWithLogitsLoss(),
            monai.losses.DiceLoss(
                sigmoid=True, smooth_nr=0.01, smooth_dr=0.01,
                include_background=True, batch=True, squared_pred=True),
        ]
        self.history = None

    def create_dataloaders(self, num_workers, subset, random_split):
        conf = self.conf
        meta_file = os.path.join(self.input_dir, '{{ cookiecutter.train_metadata }}')
        assert os.path.exists(meta_file), f'{meta_file} not found on Compute Server'
        meta_df = pd.read_csv(meta_file, dtype=str)
        class_names = util.get_class_names(meta_df)
        self.num_classes = len(class_names)

        df = util.process_files(
            conf, self.input_dir, '{{ cookiecutter.train_image_dir }}', meta_df, class_names)
        if random_split:
            # shuffle before splitting into training and validation subsets
            df = df.sample(frac=1, random_state=0).reset_index(drop=True)
        train_aug = make_train_augmenter(conf)
        val_aug = util.make_val_augmenter(conf)

        # split into train and validation sets
        split = df.shape[0]*80//100
        train_df = df.iloc[:split].reset_index(drop=True)
        val_df = df.iloc[split:].reset_index(drop=True)

        train_dataset = create_dataset(
            train_df, conf, self.input_dir, '{{ cookiecutter.train_image_dir }}',
            class_names, train_aug, subset=subset)
        val_dataset = create_dataset(
            val_df, conf, self.input_dir, '{{ cookiecutter.train_image_dir }}',
            class_names, val_aug, subset=subset)
        drop_last = (len(train_dataset) % conf.batch_size) == 1
        self.train_loader = DataLoader(
            train_dataset, batch_size=conf.batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=False,
            worker_init_fn=worker_init_fn, drop_last=drop_last)
        val_batch_size = 1
        self.val_loader = DataLoader(
            val_dataset, batch_size=val_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=False)

    def create_optimizer(self, conf, model):
        if conf.optim == 'sgd':
            return torch.optim.SGD(
                model.parameters(), lr=conf.lr, momentum=0.9,
                weight_decay=conf.weight_decay)
        if conf.optim == 'adam':
            return torch.optim.AdamW(
                model.parameters(), lr=conf.lr,
                weight_decay=conf.weight_decay)
        return None

    def fit(self, epochs):
        best_loss = None
        patience = self.max_patience
        self.sample_count = 0
        self.history = util.LossHistory()

        print(f'Running on {device}')
        print(f'{len(self.train_loader.dataset)} examples in training set')
        print(f'{len(self.val_loader.dataset)} examples in validation set')
        trial = os.environ.get('TRIAL')
        suffix = f"-trial{trial}" if trial is not None else ""
        log_dir = f"runs/{datetime.now().strftime('%b%d_%H-%M-%S')}{suffix}"
        writer = SummaryWriter(log_dir=log_dir)

        print('The best model will be saved as model.pth')
        print('Training in progress...')
        for epoch in range(epochs):
            # train for one epoch
            print(f'Epoch {epoch}:')
            train_loss = self.train_epoch(epoch)
            writer.add_scalar('Training loss', train_loss, epoch)
            print(f'training loss {train_loss:.5f}')
            if epoch % self.conf.val_interval == 0:
                val_loss, val_dice, val_hausdorff = self.validate()
                val_score = 0.4*val_dice + 0.6*val_hausdorff
                writer.add_scalar('Validation loss', val_loss, epoch)
                writer.add_scalar('Validation score', val_score, epoch)
                print(f'Validation loss {val_loss:.4} dice {val_dice:.4} hausdorff {val_hausdorff:.4} score {val_score:.4}')
                self.history.add_epoch_val_loss(epoch, self.sample_count, val_loss)
                state = {
                    'epoch': epoch, 'model': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                    'conf': self.conf.as_dict()
                }
                if best_loss is None or val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(state, 'model.pth')
                    patience = self.max_patience
                else:
                    torch.save(state, 'latest.pth')
                    patience -= 1
                    if patience == 0:
                        print(
                            f'Validation loss did not improve for '
                            f'{self.max_patience} epochs')
                        break
            writer.flush()

            self.history.save()
        writer.close()

    def criterion(self, outputs, labels):
        result = 0
        for func in self.loss_funcs:
            result += func(outputs, labels)
        return result/len(self.loss_funcs)

    def train_epoch(self, epoch):
        model = self.model
        optimizer = self.optimizer

        val_iter = iter(self.val_loader)
        val_interval = len(self.train_loader)//len(self.val_loader)
        #assert val_interval > 0
        train_loss_list = []
        model.train()
        roi_size = (self.conf.test_roi, self.conf.test_roi, self.conf.test_depth)
        sw_batch_size = 4
        for step, batch in enumerate(self.train_loader):
            # data in NCHWD format
            images, labels = batch['img'], batch['msk']
            # XXX: history collection disabled for now
            if False and (step + 1) % val_interval == 0:
                model.eval()
                # collect validation history for tuning
                try:
                    with torch.no_grad():
                        batch = next(val_iter)
                        val_images, val_labels = batch['img'], batch['msk']
                        val_images = val_images.to(device)
                        val_labels = val_labels.to(device)
{%- if cookiecutter.AMP == "True" %}
                        with autocast(device_type, enabled=self.use_amp):
                            val_outputs = sliding_window_inference(
                                val_images, roi_size, sw_batch_size, model, mode='gaussian')
{%- elif cookiecutter.AMP == "False" %}
                        val_outputs = sliding_window_inference(
                            val_images, roi_size, sw_batch_size, model, mode='gaussian')
{%- endif %}
                        val_loss = self.criterion(val_outputs, val_labels)
                        self.history.add_val_loss(epoch, self.sample_count, val_loss.item())
                except StopIteration:
                    pass
                # switch back to training mode
                model.train()

            images = images.to(device)
            labels = labels.to(device)
            # compute output
{%- if cookiecutter.AMP == "True" %}
            # use AMP
            with autocast(device_type, enabled=self.use_amp):
                outputs = model(images)
                loss = self.criterion(outputs, labels)
{%- elif cookiecutter.AMP == "False" %}
            outputs = model(images)
            loss = self.criterion(outputs, labels)
{%- endif %}

            train_loss_list.append(loss.item())
            self.sample_count += images.shape[0]
            self.history.add_train_loss(epoch, self.sample_count, loss.item())
            if (step + 1) % self.print_interval == 0:
                print(f'Batch {step + 1}: training loss {loss.item():.5f}')
            # compute gradient and do SGD step
{%- if cookiecutter.AMP == "True" %}
            if self.use_amp:
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
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
            self.scheduler.step()

        mean_train_loss = np.array(train_loss_list).mean()
        return mean_train_loss

    def validate(self):
        sigmoid = nn.Sigmoid()
        losses = []
        dice = []
        hausdorff = []
        sw_batch_size = 4
        overlap = self.conf.sliding_win_overlap
        if '3D' in self.conf.arch:
            roi_size = (self.conf.test_roi, self.conf.test_roi, self.conf.test_depth)
        else:
            roi_size = (self.conf.test_roi, self.conf.test_roi)
        flip_dims = self.conf.tta_flip_dims
        self.model.eval()
        with torch.no_grad():
            for batch in self.val_loader:
                images, labels = batch['img'], batch['msk']
                images = images.to(device)
                labels = labels.to(device)
                outputs = None
                for tta in flip_dims:
                    tta_images = torch.flip(images, tta)
{%- if cookiecutter.AMP == "True" %}
                    with autocast(device_type, enabled=self.use_amp):
                        tta_outputs = sliding_window_inference(
                            tta_images, roi_size, sw_batch_size, self.model,
                            mode='gaussian', overlap=overlap)
                        tta_outputs = torch.flip(tta_outputs, tta)
                        if outputs == None:
                            outputs = tta_outputs
                        else:
                            outputs += tta_outputs
                outputs /= len(flip_dims)
                with autocast(device_type, enabled=self.use_amp):
                    losses.append(self.criterion(outputs, labels).item())
{%- elif cookiecutter.AMP == "False" %}
                    tta_outputs = sliding_window_inference(
                        tta_images, roi_size, sw_batch_size, self.model,
                        mode='gaussian', overlap=overlap)
                    tta_outputs = torch.flip(tta_outputs, tta)
                    if outputs == None:
                        outputs = tta_outputs
                    else:
                        outputs += tta_outputs
                outputs /= len(flip_dims)
                losses.append(self.criterion(outputs, labels).item())
{%- endif %}
                preds = sigmoid(outputs).round().to(torch.float32)
                if self.conf.multi_slice_label:
                    # only consider the middle slice
                    num_slices = self.conf.num_slices
                    assert labels.shape[1] == num_slices*self.num_classes
                    start = (num_slices//2)*self.num_classes
                    labels = labels[:, start:start + self.num_classes]
                    preds = preds[:, start:start + self.num_classes]
                dice.append(util.dice_coeff(labels, preds).item())
                hausdorff.append(util.hausdorff_score(labels, preds))
        return np.mean(losses), np.mean(dice), np.mean(hausdorff)


def worker_init_fn(worker_id):
    random.seed(random.randint(0, 2**32) + worker_id)
    np.random.seed(random.randint(0, 2**32) + worker_id)


def main():
    args = parser.parse_args()
    if args.subset != 100:
        print(f'\nWARNING: {args.subset}% of the data will be used for training\n')
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    input_dir = args.input
    model_file = args.resume
    model_file = args.resume or args.validate
    if model_file:
        print(f'Loading model from {model_file}')
        checkpoint = torch.load(model_file)
        conf = Config(checkpoint['conf'])
        # XXX
        conf['tta_flip_dims'] = [[], [2], [3], [2, 3]]
    else:
        checkpoint = None
        conf = Config()

    print(conf)
    if '3D' not in conf.arch:
        args.print_interval *= 10
    trainer = Trainer(
        conf, input_dir, device, args.num_workers,
        checkpoint, args.print_interval, args.subset)

    if args.validate:
        loss, dice, hausdorff = trainer.validate()
        score = 0.4*dice + 0.6*hausdorff
        print(f'Validation loss {loss:.4} dice {dice:.4} hausdorff {hausdorff:.4} score {score:.4}')
    else:
        trainer.fit(args.epochs)


if __name__ == '__main__':
    main()
    print('Done')
