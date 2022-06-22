import torch.nn as nn
import monai
import segmentation_models_pytorch as smp


class ModelWrapper(nn.Module):

    def __init__(self, conf, num_classes):
        super().__init__()
        if conf.arch == 'FPN':
            arch = smp.FPN
        elif conf.arch == 'Unet':
            arch = smp.Unet
        elif conf.arch == 'Unet++':
            arch = smp.UnetPlusPlus
        elif conf.arch == 'DeepLabV3':
            arch = smp.DeepLabV3
        elif conf.arch == 'Unet3D':
            arch = monai.networks.nets.UNet
        else:
            assert 0, f'Unknown architecture {conf.arch}'

        if conf.arch == 'Unet3D':
            self.model = arch(
                spatial_dims=3, in_channels=1, out_channels=3,
                channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2),
                norm='INSTANCE',
                num_res_units=conf.res_units,
                dropout=conf.dropout
                )
        else:
            weights = 'imagenet' if conf.pretrained else None
            if 'num_slices' not in conf._params:
                conf['num_slices'] = 5
            if conf.multi_slice_label:
                num_output_channels = conf.num_slices*num_classes
            else:
                num_output_channels = num_classes
            self.model = arch(
                encoder_name=conf.backbone, encoder_weights=weights, in_channels=conf.num_slices,
                classes=num_output_channels, activation=None)

    def forward(self, x):
        x = self.model(x)
        return  x
