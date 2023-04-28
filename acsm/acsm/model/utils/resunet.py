'''
Code borrowed from
https://github.com/nileshkulkarni/acsm/blob/master/acsm/nnutils/resunet.py
'''

from __future__ import absolute_import, division, print_function
import torchvision
import torch.nn as nn

from acsm.model.utils import net_blocks as nb

class ResnetEncoder(nn.Module):
    def __init__(self, n_blocks):
        super(ResnetEncoder, self).__init__()
        self.n_blocks = n_blocks
        self.resnet   = torchvision.models.resnet18(pretrained=True)

    def forward(self, x):
        n_blocks = self.n_blocks
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        if n_blocks >= 1:
            x = self.resnet.layer1(x)
        if n_blocks >= 2:
            x = self.resnet.layer2(x)
        if n_blocks >= 3:
            x = self.resnet.layer3(x)
        if n_blocks >= 4:
            x = self.resnet.layer4(x)
        return x

class ResNetConcatGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, n_blocks=4, ngf=64):
        super(ResNetConcatGenerator, self).__init__()
        self.encoder  = ResnetEncoder(n_blocks=n_blocks)
        self.n_blocks = n_blocks
        decoder = []
        if n_blocks == 3:
            nlayers  = 4
            inner_nc = 256
        elif n_blocks == 4:
            nlayers  = 5
            inner_nc = 512

        for lx in range(nlayers):
            outnc = max(inner_nc // 2, 16)
            up = nb.upconv2d(inner_nc, outnc)
            decoder.append(up)
            inner_nc = outnc

        up = nn.Conv2d(inner_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=True)
        decoder.append(up)
        self.decoder = nn.Sequential(*decoder)
        nb.net_init(self.decoder)
        return

    def reinit_weights(self):
        self.encoder = ResnetEncoder(n_blocks=self.n_blocks)
        nb.net_init(self.decoder)

    def forward(self, x):
        img_enc = self.encoder(x)
        img_dec = self.decoder(img_enc)
        return img_dec
