import os
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

def plot_images(images, labels, outputs, input_dir):
    num_images = min(12, images.shape[0])
    images = images[:num_images]
    labels = np.argmax(labels[:num_images], axis=1)

    outputs = outputs[:num_images]
    num_cols = min(4, num_images)
    num_rows = (num_images + num_cols - 1) // num_cols
    preds = outputs.argmax(axis=1)
    probs = [100*F.softmax(el, dim=0)[i] for i, el in zip(preds, outputs)]

    # plot the images, along with predicted and true labels
    fig = plt.figure(figsize=(10, num_images))
    for idx, (img, pred, prob, label) in enumerate(
            zip(images, preds, probs, labels)):
        ax = fig.add_subplot(num_rows, num_cols, idx+1, xticks=[], yticks=[])
        img -= img.min()
        img *= 255/img.max()
        npimg = np.uint8(img.numpy().round())
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        color = 'green' if pred == label else 'red'
        ax.set_title(
            f'pred: {pred}, {prob:.1f}%\n(label: {label})', color=color)
    plt.savefig('prediction-samples.png', dpi=150)
    return fig
