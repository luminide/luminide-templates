import torch.nn as nn
import segmentation_models_pytorch as smp
import timm


class ModelWrapper(nn.Module):

    def __init__(self, conf, num_classes):
        super().__init__()
        weights = 'imagenet' if conf.pretrained else None
        self.model = smp.FPN(
            encoder_name=conf.arch, encoder_weights=weights, in_channels=5,
            classes=num_classes, activation=None)

    def forward(self, x):
        x = self.model(x)
        return  x
