import torch
import torch.nn as nn
import timm
from timm.models.resnet import (
    BasicBlock, Bottleneck, make_blocks, drop_blocks, downsample_conv, downsample_avg)


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
        self.num_instances = conf.num_patches
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

class EncoderHead(nn.Module):

    def __init__(self, conf, num_classes, in_channels):
        super().__init__()
        layers = []
        for out_channels in [128, 32, 8]:
            stage_modules, _ = make_blocks(
                BasicBlock, [out_channels], [2], in_channels)
            in_channels = out_channels
            layers.append(stage_modules[0][1])
        self.layer1 = layers[0]
        self.layer2 = layers[1]
        self.layer3 = layers[2]

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class SelfSupervisedModel(nn.Module):

    def __init__(self, conf, num_classes):
        super().__init__()
        self.conf = conf
        self.create_encoder()

        feature_dim = self.encoder.get_classifier().in_features
        self.classifier = timm.create_model(
            conf.arch, conf.pretrained,
            num_classes=conf.ssl_num_classes,
            in_chans=3,
            drop_rate=conf.dropout_rate)
        self.upsample = nn.Upsample(size=(conf.image_size, conf.image_size))
        self.save = False

    def create_encoder(self):
        enc_arch = 'resnet18'
        ssl_num_classes = self.conf.ssl_num_classes
        self.encoder = timm.create_model(
            enc_arch, True,
            num_classes=ssl_num_classes)

        # change the strides on layers 2, 3 and 4 from 2 to 1
        layers = []
        for layer in [self.encoder.layer2, self.encoder.layer3, self.encoder.layer4]:
            stage_modules, _ = make_blocks(
                BasicBlock, [layer[0].conv2.out_channels], [2],
                layer[0].conv1.in_channels)
            layers.append(stage_modules[0][1])
        self.encoder.layer2 = layers[0]
        self.encoder.layer3 = layers[1]
        self.encoder.layer4 = layers[2]

        self.encoder.head = EncoderHead(
            self.conf, ssl_num_classes,
            self.encoder.layer4[0].conv2.out_channels)
        self.bin_classifier = nn.Linear(ssl_num_classes*ssl_num_classes, 2)

    def make_head(self):
        stage_modules, _ = make_blocks(
            BasicBlock, [self.conf.ssl_num_classes], [2],
            self.encoder.layer4[0].conv2.out_channels)
        self.encoder.head = stage_modules[0][1]
        pass

    def freeze_classifier(self):
        self.classifier.requires_grad_(False)

    def unfreeze_classifier(self):
        self.classifier.requires_grad_(True)

    def freeze_encoder(self):
        self.encoder.requires_grad_(False)

    def unfreeze_encoder(self):
        self.encoder.requires_grad_(True)

    def encode(self, x):
        x = self.encoder.forward_features(x)
        x = self.encoder.head(x)
        return x

    def forward(self, inputs):
        masks = self.encode(inputs)
        masks = self.upsample(masks)
        inputs = inputs.unsqueeze(1)
        masks =  masks.unsqueeze(2)
        # scale masks to (0, 1)
        N, M, C, H, W = masks.shape
        mask_view = masks.view((N*M, C*H*W))
        mask_view /= mask_view.max(axis=0).values + 1e-6
        prod = masks*inputs
        # at this point, the shape of the data is NMCHW, where M is the number of SSL classes
        masked_input = prod.view(N*M, 3, H, W)
        # TODO concatenate masked_input with masks before feeding it to the classifier
        outputs = self.classifier(masked_input)

        self.bin_outputs = self.bin_classifier(outputs.view(N, M*M))
        # TODO: remove this
        self.masks = masks
        self.masked_input = masked_input
        return outputs
