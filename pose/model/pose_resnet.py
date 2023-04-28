'''
Code borrowed and adapted from
https://github.com/bearpaw/pytorch-pose/blob/master/pose/models/pose_resnet.py
'''
import torch.nn as nn
import torchvision.models as M
from easydict import EasyDict as edict

resnets = {
    'resnet18' : [M.resnet18,  512],
    'resnet34' : [M.resnet34,  512],
    'resnet50' : [M.resnet50,  2048],
    'resnet101': [M.resnet101, 2048],
    'resnet152': [M.resnet152, 2048],
}

def get_default_network_config():
    config = edict()
    config.pretrained = True
    config.final_conv_kernel = 1
    config.transposed_conv_layers = 3
    config.transposed_conv_channels = 256
    config.transposed_conv_kernel_size = 4
    config.depth_dim = 1
    config.final_bias = True
    return config

def pose_resnet(model_name, num_joints):
    cfg = get_default_network_config()
    model_func, embedding_dim = resnets[model_name]
    resnet   = model_func(pretrained=cfg.pretrained)
    modules  = list(resnet.children())[:-2]
    backbone = nn.Sequential(*modules)
    head = TransposedConvHead(in_channels = embedding_dim,
                    num_layers    = cfg.transposed_conv_layers,
                    num_channels  = cfg.transposed_conv_channels,
                    kernel_size   = cfg.transposed_conv_kernel_size,
                    final_conv_kernel_size = cfg.final_conv_kernel,
                    num_joints    = num_joints,
                    depth_dim     = cfg.depth_dim,
                    final_bias    = cfg.final_bias
            )
    pose_net = SimpleBaselinesNet(backbone, head)
    return pose_net

class SimpleBaselinesNet(nn.Module):
    def __init__(self, backbone, head):
        super(SimpleBaselinesNet, self).__init__()
        self.backbone = backbone
        self.head     = head

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

class TransposedConvHead(nn.Module):
    def __init__(self, in_channels, num_layers, num_channels, kernel_size, final_conv_kernel_size, num_joints, depth_dim, final_bias=True):
        super(TransposedConvHead, self).__init__()
        final_conv_channels = num_joints * depth_dim

        assert kernel_size == 2 or kernel_size == 3 or kernel_size == 4, 'Only support kenerl 2, 3 and 4'
        padding, output_padding = 1, 0
        if kernel_size == 3:
            output_padding = 1
        elif kernel_size == 2:
            padding = 0

        assert final_conv_kernel_size == 1 or final_conv_kernel_size == 3, 'Only support kenerl 1 and 3'
        if final_conv_kernel_size == 1:
            pad = 0
        elif final_conv_kernel_size == 3:
            pad = 1

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            _in_channels = in_channels if i == 0 else num_channels
            self.layers.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels=_in_channels, out_channels=num_channels, kernel_size=kernel_size,
                                            stride=2, padding=padding, output_padding=output_padding, bias=False),
                nn.BatchNorm2d(num_channels),
                nn.ReLU(inplace=True)
                )
            )

        # 1x1 convolution to get the output heatmaps
        self.layers.append(
            nn.Conv2d(in_channels=num_channels, out_channels=final_conv_channels,
                        kernel_size=final_conv_kernel_size, padding=pad, bias=True)
            )
        if not final_bias:
            self.layers.append( nn.Sequential( nn.BatchNorm2d(final_conv_channels), nn.ReLU(inplace=True) ) )

        # initialize parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                if final_bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x