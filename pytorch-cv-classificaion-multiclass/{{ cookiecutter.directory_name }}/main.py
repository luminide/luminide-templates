import argparse
import os
import logging
import random
import cv2
import numpy as np
import pandas as pd
import multiprocessing as mp

import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
import torch.utils.data as data
from torch.cuda.amp import GradScaler, autocast

from augment import make_augmenters
from dataset import VisionDataset
from models import ModelWrapper
from config import hp_dict, Config
from report import plot_images


NUM_CLASSES = {{ cookiecutter.num_classes }}

logging.basicConfig(
    level=logging.INFO,
    format=('%(asctime)s - %(name)s - ' '%(levelname)s -  %(message)s'))
logger = logging.getLogger('main')
parser = argparse.ArgumentParser(description='Pattern recognition')
parser.add_argument(
    '-b', '--batch-size', default=128, type=int, help='mini-batch size')
parser.add_argument(
    '-j', '--workers', default=mp.cpu_count(), type=int, metavar='N',
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
    '-f', '--f32', dest='f32', action='store_true',
    help='use 32 bit precision while training')
parser.add_argument(
    '--input', default='../input', metavar='DIR', help='input directory')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info('Running on %s', device)


def train(loader, model, optimizer, scaler, epoch):
    loss_func = nn.BCEWithLogitsLoss()
    sigmoid = nn.Sigmoid()

    # skip bprop on some minibatches
    val_percent = 10
    skip_interval = 100 // val_percent
    loss_sums = [0, 0]
    batch_counts = [0, 0]
    model.train()
    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        # compute output
        optimizer.zero_grad()
        outputs = model(images)
        if scaler:
            # use AMP
            with autocast():
                loss = loss_func(outputs, labels)
        else:
            loss = loss_func(outputs, labels)

        if i % skip_interval == 0:
            # skip this minibatch while tuning
            loss_sums[1] += loss.item()
            batch_counts[1] += 1
            continue

        probs = sigmoid(outputs).data.cpu().numpy()
        preds = probs.round()
        loss_sums[0] += loss.item()
        batch_counts[0] += 1
        # compute gradient and do SGD step
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    train_loss = loss_sums[0]/batch_counts[0]
    val_loss = loss_sums[1]/batch_counts[1]
    return train_loss, val_loss

def worker_init_fn(worker_id):
    random.seed(random.randint(0, 2**32) + worker_id)

def make_report(model, loader, input_dir):
    images, labels = iter(loader).next()
    with torch.no_grad():
        outputs = model(images.to(device)).cpu()
    plot_images(images, labels, outputs, input_dir)

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
        logger.info(f'loading model from {model_file}')
        checkpoint = torch.load(model_file)
        conf = Config(checkpoint['hp_dict'])
    else:
        conf = Config(hp_dict)

    logger.info(conf.get())
    model = ModelWrapper(NUM_CLASSES, conf)
    model = model.to(device)
    if model_file:
        model.load_state_dict(checkpoint['model'])

    train_aug, test_aug = make_augmenters(conf)

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), conf.lr, conf.momentum, conf.nesterov)

    # data loading code
    train_dataset = VisionDataset(
        True, conf, input_dir, '{{ cookiecutter.train_image_dir }}',
        NUM_CLASSES, train_aug, args.quick)
    train_loader = data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        num_workers=args.workers, worker_init_fn=worker_init_fn)
    writer = SummaryWriter()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=conf.gamma)
    scaler = None if args.f32 else GradScaler()
    best_loss = None
    for epoch in range(args.epochs):
        # train for one epoch
        train_loss, val_loss = train(train_loader, model, optimizer, scaler, epoch)
        writer.add_scalar('training loss', train_loss, epoch)
        scheduler.step()
        writer.add_scalar('validation loss', val_loss, epoch)
        writer.flush()
        print(
            f'Epoch {epoch + 1}: training loss {train_loss:.4f} '
            f' validation loss {val_loss:.4f}')
        if best_loss is None or val_loss < best_loss:
            best_loss = val_loss
            state = {
                'epoch': epoch, 'model': model.state_dict(),
                'optimizer' : optimizer.state_dict(), 'arch' : conf.arch,
                'hp_dict': conf.get()
            }
            torch.save(state, 'model.pth')

    writer.close()
    make_report(model, train_loader, input_dir)

if __name__ == '__main__':
    main(None)
    logger.info('Done')
