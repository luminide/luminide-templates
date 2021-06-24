import torch.nn as nn
import timm


class ModelWrapper(nn.Module):

    def __init__(
        self, num_classes, conf
    ):
        super().__init__()
        self.model = timm.create_model(
            conf.arch, conf.pretrained, num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        return  x
