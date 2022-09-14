import torch.nn as nn
import timm


# adapted from https://www.kaggle.com/code/analokamus/a-sample-of-multi-instance-learning-model/notebook
class Flatten(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        input_shape = x.shape
        output_shape = [input_shape[i] for i in range(self.dim)] + [-1]
        return x.view(*output_shape)

class ModelWrapper(nn.Module):

    def __init__(self, conf, num_classes):
        super().__init__()
        self.num_instances = 16
        self.encoder = timm.create_model(
            conf.arch, conf.pretrained,
            num_classes=num_classes, drop_rate=conf.dropout_rate)
        feature_dim = self.encoder.get_classifier().in_features
        self.head = nn.Sequential(
            nn.AdaptiveMaxPool2d(1), Flatten(),
            nn.Linear(feature_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )


    def forward(self, x):
        # x: bs x N x C x W x W
        bs, _, ch, w, h = x.shape
        x = x.view(bs*self.num_instances, ch, w, h) # x: N bs x C x W x W
        x = self.encoder.forward_features(x) # x: N bs x C' x W' x W'

        # Concat and pool
        bs2, ch2, w2, h2 = x.shape
        x = x.view(-1, self.num_instances, ch2, w2, h2).permute(0, 2, 1, 3, 4)\
            .contiguous().view(bs, ch2, self.num_instances*w2, h2) # x: bs x C' x N W'' x W''
        x = self.head(x)

        return  x
