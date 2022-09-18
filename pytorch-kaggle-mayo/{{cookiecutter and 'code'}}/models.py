import torch.nn as nn
import timm
from timm.models.resnet import Bottleneck, make_blocks, downsample_conv


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

class SelfSupervisedModel(nn.Module):

    def __init__(self, conf, num_classes):
        super().__init__()
        enc_arch = 'resnet18'
        self.encoder = timm.create_model(
            enc_arch, True,
            num_classes=conf.ssl_num_classes)
        #self.encoder_head = make_blocks(
        #    block_fn=Bottleneck,
        #    channels=[conf.ssl_num_classes], block_repeats=[1], inplanes=64)[0][0]

        in_channels =  self.encoder.layer4[0].conv2.out_channels
        self.encoder_head = self.make_layer(
            Bottleneck, in_channels, conf.ssl_num_classes, stride=1)
        #self.encoder.add_module(self.encoder_head)
        feature_dim = self.encoder.get_classifier().in_features
        self.classifier = timm.create_model(
            conf.arch, conf.pretrained,
            num_classes=conf.ssl_num_classes,
            in_chans=3,
            drop_rate=conf.dropout_rate)
        self.upsample = nn.Upsample(size=(conf.image_size, conf.image_size))
        self.save = False

    def make_head(self, inplanes, planes):
        pass

    def make_layer(self, block, inplanes, planes, stride):
        assert planes % Bottleneck.expansion == 0
        downsample = downsample_conv(inplanes, planes, 1)
        return block(inplanes, planes // Bottleneck.expansion, stride, downsample)

    def freeze_classifier(self):
        self.classifier.requires_grad_(False)

    def unfreeze_classifier(self):
        self.classifier.requires_grad_(True)

    def freeze_encoder(self):
        self.encoder.requires_grad_(False)

    def unfreeze_encoder(self):
        self.encoder.requires_grad_(True)

    def encode(self, x):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.act1(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)
        x = self.encoder_head(x)
        return x

    def forward(self, inputs):
        masks = self.encode(inputs)
        masks = self.upsample(masks)
        inputs = inputs.unsqueeze(1)
        masks =  masks.unsqueeze(2)
        #masks *= masks
        prod = masks * inputs
        # at this point, the shape of the data is NMCHW, where M is the number of SSL classes
        masked_input = prod.view(
            prod.shape[0] * prod.shape[1], prod.shape[2], prod.shape[3], prod.shape[4])
        # TODO concatenate masked_input with masks before feeding it to the classifier
        outputs = self.classifier(masked_input)

        # TODO: remove this
        self.masks = masks
        self.masked_input = masked_input
        return outputs
