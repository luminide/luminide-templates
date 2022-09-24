import os
import argparse
import random
import cv2
import multiprocessing as mp
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
import torch.utils.data as data
{%- if cookiecutter.AMP == "True" %}
from torch import autocast
from torch.cuda.amp import GradScaler
{%- endif %}

from augment import make_train_augmenter
from dataset import VisionDataset
from models import ModelWrapper, SelfSupervisedModel
from config import Config
from util import LossHistory, get_class_names, make_test_augmenter


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
    '-s', '--subset', default=100, type=int, metavar='N',
    help='use a percentage of the data for training and validation')
parser.add_argument(
    '--input', default='../input', metavar='DIR',
    help='input directory')

device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_type)


def to_hwc(img):
    return img.transpose(1, 2, 0)

def denorm(img):
    img -= img.min()
    maxval = img.max()
    if maxval != 0:
        img /= maxval
    return np.uint8((img*255).round())

class Trainer:
    def __init__(
            self, conf, input_dir, device, num_workers,
            checkpoint, print_interval=100, subset=100):
        self.conf = conf
        self.input_dir = input_dir
        self.device = device
        self.max_patience = 100
        self.print_interval = print_interval
{%- if cookiecutter.AMP == "True" %}
        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler()
{%- endif %}

        self.create_dataloaders(num_workers, subset)

        if conf.mode == 'ssl':
            self.model = SelfSupervisedModel(conf, self.num_classes)
        else:
            self.model = ModelWrapper(conf, self.num_classes)
        self.model = self.model.to(device)
        self.optimizer = self.create_optimizer(conf, self.model)
        assert  self.optimizer is not None, f'Unknown optimizer {conf.optim}'
        if checkpoint:
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=conf.gamma)
        self.history = None

    def create_dataloaders(self, num_workers, subset):
        conf = self.conf
        meta_file = os.path.join(self.input_dir, '{{ cookiecutter.train_metadata }}')
        assert os.path.exists(meta_file), f'{meta_file} not found on Compute Server'
        df = pd.read_csv(meta_file, dtype=str)
        class_names = get_class_names(df)
        self.num_classes = len(class_names)

        # shuffle
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)
        train_aug = make_train_augmenter(conf)
        test_aug = make_test_augmenter(conf)

        # split into train and validation sets
        split = df.shape[0]*90//100
        train_df = df.iloc[:split].reset_index(drop=True)
        val_df = df.iloc[split:].reset_index(drop=True)
        train_dataset = VisionDataset(
            train_df, conf, self.input_dir, '{{ cookiecutter.train_image_dir }}',
            class_names, train_aug, subset)
        val_dataset = VisionDataset(
            val_df, conf, self.input_dir, '{{ cookiecutter.train_image_dir }}',
            class_names, test_aug, subset)
        drop_last = (len(train_dataset) % conf.batch_size) == 1
        self.train_loader = data.DataLoader(
            train_dataset, batch_size=conf.batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=False,
            worker_init_fn=worker_init_fn, drop_last=drop_last)
        self.val_loader = data.DataLoader(
            val_dataset, batch_size=conf.batch_size, shuffle=False,
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
        self.history = LossHistory()

        print(f'Running on {device}')
        print(f'{len(self.train_loader.dataset)} examples in training set')
        print(f'{len(self.val_loader.dataset)} examples in validation set')
        trial = os.environ.get('TRIAL')
        suffix = f"-trial{trial}" if trial is not None else ""
        log_dir = f"runs/{datetime.now().strftime('%b%d_%H-%M-%S')}{suffix}"
        writer = SummaryWriter(log_dir=log_dir)

        print('Training in progress...')
        for epoch in range(epochs):
            # train for one epoch
            print(f'Epoch {epoch}:')
            if self.conf.mode == 'ssl':
                train_loss = self.ssl_train_epoch(epoch)
                val_loss, val_score = self.ssl_validate()
            else:
                train_loss = self.train_epoch(epoch)
                val_loss, val_score = self.validate()
            self.scheduler.step()
            writer.add_scalar('Training loss', train_loss, epoch)
            writer.add_scalar('Validation loss', val_loss, epoch)
            writer.add_scalar('Validation F1 score', val_score, epoch)
            writer.flush()
            print(f'training loss {train_loss:.5f}')
            self.history.add_epoch_val_loss(epoch, self.sample_count, val_loss)
            if best_loss is None or val_loss < best_loss:
                print(f'* Validation F1 score {val_score:.4f} loss {val_loss:.4f}\n')
                best_loss = val_loss
                state = {
                    'epoch': epoch, 'model': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                    'conf': self.conf.as_dict()
                }
                torch.save(state, 'model.pth')
                patience = self.max_patience
            else:
                print(f'Validation F1 score {val_score:.4f} loss {val_loss:.4f}\n')
                patience -= 1
                if patience == 0:
                    print(
                        f'Validation loss did not improve for '
                        f'{self.max_patience} epochs')
                    break

            self.history.save()
        writer.close()

    def get_ssl_labels(self, outputs):
        labels = torch.zeros_like(outputs)
        if False:
            # softmax so that the probabilities add up to 1.
            sm = torch.nn.Softmax(dim=1)
            probs = sm(outputs).cpu().detach().numpy()

            # roll a die weighted with the probabilities predicted for each alphabet
            # (the die has num_classes faces)
            #item_probs = np.exp(item_probs)/np.sum(np.exp(item_probs))
            for idx, item_probs in enumerate(probs):
                # TODO: use torch.randperm?
                # instead of hard labels, just exaggerate the predictions
                val = int(np.random.choice(range(self.conf.ssl_num_classes), p=item_probs))
                labels[idx, val] = 1
        num_classes = self.conf.ssl_num_classes
        assert labels.shape[0] % num_classes == 0

        for i in range(labels.shape[0] // num_classes):
            labels[range(i * num_classes, num_classes * (i + 1)), range(num_classes)] = 1
        return labels

    def save_examples(self, epoch, images):
        conf = self.conf
        model = self.model
        images = images.cpu().detach().numpy()
        masks = model.masks.cpu().detach().numpy()
        masked_input = model.masked_input.cpu().detach().numpy()
        num_classes = self.conf.ssl_num_classes
        for i, _ in enumerate(images):
            img = denorm(to_hwc(images[i]))
            cv2.imwrite(f'input{i}_e{epoch}.png', img)
            for c in range(num_classes):
                msk = denorm(masks[i, c, 0])
                cv2.imwrite(f'mask{i}_c{c}_e{epoch}.png', msk)
                msk_input = denorm(to_hwc(masked_input[i*num_classes + c]))
                cv2.imwrite(f'masked_input{i}_c{c}_e{epoch}.png', msk_input)

    def ssl_train_epoch(self, epoch):
        loss_func = nn.BCEWithLogitsLoss()
        model = self.model
        optimizer = self.optimizer

        #if epoch != 0 and epoch % 9 == 0:
        if False:
            print('unfreezing the classifier')
            model.unfreeze_classifier()
            model.freeze_encoder()
        #else:
        if False:
            print('freezing the classifier')
            model.freeze_classifier()
            model.unfreeze_encoder()

        train_loss_list = []
        plain_loss_list = []
        bin_loss_list = []
        sum_diff_list = []
        prod_diff_list = []
        model.train()
        for step, (images, bin_labels) in enumerate(self.train_loader):
            images = images.to(device)
            bin_labels = bin_labels.to(device)
            with autocast(device_type, enabled=self.use_amp):
                outputs = model(images)
                labels = self.get_ssl_labels(outputs)
                loss = loss_func(outputs, labels)
                bin_loss = loss_func(self.model.bin_outputs, bin_labels)
                # the masks are in NMCHW format, where M is the number of masks and C = 1
                # constrain the sum of all mask pixels to be 1
                mask_sums = torch.sum(self.model.masks, axis=1)
                diff = mask_sums - 1
                sum_diff = (diff*diff).mean()
                M = self.model.masks.shape[1]
                prod_diff = 0
                # encourage the product of any 2 masks to be zero
                for i in range(M):
                    for j in range(i + 1, M):
                        assert i != j
                        mask_prods = self.model.masks[:, i] * self.model.masks[:, j]
                        prod_diff += mask_prods.mean()

            plain_loss_list.append(loss.item())
            bin_loss_list.append(bin_loss.item())
            sum_diff_list.append(sum_diff.item())
            prod_diff_list.append(prod_diff.item())

            #loss += sum_diff
            loss += prod_diff
            loss += bin_loss

            train_loss_list.append(loss.item())
            if (step + 1) % self.print_interval == 0:
                print(f'Batch {step + 1}: training loss {loss.item():.5f}')
            # compute gradient and do SGD step
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad()
            if step == len(self.train_loader) - 1:
                self.save_examples(epoch, images)

        mean_loss =  np.array(plain_loss_list).mean()
        mean_bin_loss =  np.array(bin_loss_list).mean()
        mean_sum_diff =  np.array(sum_diff_list).mean()
        mean_prod_diff =  np.array(prod_diff_list).mean()
        mean_train_loss = np.array(train_loss_list).mean()
        print(f'plain loss {mean_loss:.4f} bin loss {mean_bin_loss:.4f} sum_diff {mean_sum_diff:.4f} prod_diff {mean_prod_diff:.4f} all {mean_train_loss:.4f}')
        return mean_train_loss

    def train_epoch(self, epoch):
        loss_func = nn.BCEWithLogitsLoss()
        model = self.model
        optimizer = self.optimizer

        val_iter = iter(self.val_loader)
        val_interval = len(self.train_loader)//len(self.val_loader)
        assert val_interval > 0
        train_loss_list = []
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
                        with autocast(device_type, enabled=self.use_amp):
                            val_outputs = model(val_images)
{%- elif cookiecutter.AMP == "False" %}
                        val_outputs = model(val_images)
{%- endif %}
                        val_loss = loss_func(val_outputs, val_labels)
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
                loss = loss_func(outputs, labels)
{%- elif cookiecutter.AMP == "False" %}
            outputs = model(images)
            loss = loss_func(outputs, labels)
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

    def ssl_validate(self):
        loss_func = nn.BCEWithLogitsLoss()
        sigmoid = nn.Sigmoid()
        losses = []
        num_classes = self.num_classes
        all_labels = np.zeros(
            (len(self.val_loader.dataset), num_classes), dtype=np.float32)
        preds = np.zeros_like(all_labels)
        start_idx = 0
        self.model.eval()
        with torch.no_grad():
            for images, bin_labels in self.val_loader:
                images = images.to(device)
                bin_labels = bin_labels.to(device)
                with autocast(device_type, enabled=self.use_amp):
                    outputs = self.model(images)
                end_idx = start_idx + self.model.bin_outputs.shape[0]
                all_labels[start_idx:end_idx] = bin_labels.cpu().numpy()
                preds[start_idx:end_idx] = sigmoid(self.model.bin_outputs).round().cpu().numpy()
                if np.isfinite(preds).all() == False:
                    import ipdb;ipdb.set_trace()
                start_idx = end_idx
                losses.append(loss_func(self.model.bin_outputs, bin_labels).item())

        if np.isfinite(preds).all():
            score = f1_score(all_labels, preds, average='micro')
        else:
            score = np.nan
        return np.mean(losses), score

    def validate(self):
        loss_func = nn.BCEWithLogitsLoss()
        sigmoid = nn.Sigmoid()
        losses = []
        all_labels = np.zeros(
            (len(self.val_loader.dataset), self.num_classes), dtype=np.float32)
        preds = np.zeros_like(all_labels)
        start_idx = 0
        self.model.eval()
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(device)
                labels = labels.to(device)
{%- if cookiecutter.AMP == "True" %}
                with autocast(device_type, enabled=self.use_amp):
                    outputs = self.model(images)
{%- elif cookiecutter.AMP == "False" %}
                outputs = self.model(images)
{%- endif %}
                end_idx = start_idx + outputs.shape[0]
                all_labels[start_idx:end_idx] = labels.cpu().numpy()
                preds[start_idx:end_idx] = sigmoid(outputs).round().cpu().numpy()
                start_idx = end_idx
                losses.append(loss_func(outputs, labels).item())

        if np.isfinite(preds).all():
            score = f1_score(all_labels, preds, average='micro')
        else:
            score = np.nan
        return np.mean(losses), score


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
    if model_file:
        print(f'Loading model from {model_file}')
        checkpoint = torch.load(model_file)
        conf = Config(checkpoint['conf'])
    else:
        checkpoint = None
        conf = Config()

    print(conf)
    trainer = Trainer(
        conf, input_dir, device, args.num_workers,
        checkpoint, args.print_interval, args.subset)
    trainer.fit(args.epochs)


if __name__ == '__main__':
    main()
    print('Done')
