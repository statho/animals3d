'''
Code borrowed from
https://github.com/nileshkulkarni/acsm/blob/master/acsm/nnutils/unet.py
'''

from __future__ import absolute_import, division, print_function

import functools
import torch
import torch.nn as nn
from acsm.model.utils import net_blocks as nb

#------ UNet style generator ------#
# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet.
# For example, if |num_downs| == 7, image of size 128x128 will become of size 1x1
# concats additional features at the bottleneck

class UnetConcatGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.modules.normalization.GroupNorm, remove_skips=-1):
        super(UnetConcatGenerator, self).__init__()

        if num_downs >= 5:
            ngf_max = ngf*8
        else:
            ngf_max = ngf*pow(2, num_downs - 2)

        # construct unet structure
        all_blocks = []
        self.inner_most_block = unet_block = UnetSkipConnectionConcatBlock(
            ngf_max, ngf_max, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        all_blocks.append(unet_block)

        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionConcatBlock(
                ngf_max, ngf_max, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
            all_blocks.append(unet_block)
        if remove_skips > 0:
            for i in range(min(remove_skips, num_downs - 2)):
                unet_block = UnetSkipConnectionConcatBlock(ngf_max // pow(2, i+1), ngf_max // pow(
                    2, i), input_nc=None, submodule=unet_block, norm_layer=norm_layer)
                all_blocks.append(unet_block)
            for i in range(min(remove_skips, num_downs - 2), min(3, num_downs - 2)):
                unet_block = UnetSkipConnectionConcatBlock(ngf_max // pow(2, i+1), ngf_max // pow(
                    2, i), input_nc=None, submodule=unet_block, norm_layer=norm_layer, no_skip=True)
                all_blocks.append(unet_block)
        else:
            for i in range(min(3, num_downs - 2)):
                unet_block = UnetSkipConnectionConcatBlock(ngf_max // pow(2, i+1), ngf_max // pow(
                    2, i), input_nc=None, submodule=unet_block, norm_layer=norm_layer)
                all_blocks.append(unet_block)

        unet_block = UnetSkipConnectionConcatBlock(
            output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)
        all_blocks.append(unet_block)

        self.model = unet_block
        self.all_blocks = all_blocks
        nb.net_init(self.model)

    def get_inner_most(self):
        return self.inner_most_block

    def get_all_block(self):
        return self.all_blocks

    def forward(self, x):
        return self.model(x)



# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionConcatBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False,
                                                        norm_layer=nn.modules.normalization.GroupNorm, no_skip=False):
        super(UnetSkipConnectionConcatBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.no_skip   = no_skip
        # if submodule is None:
        #     pdb.set_trace()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.01, False)
        uprelu = nn.ReLU(False)
        if outermost:
            self.down = [downconv]
            self.up   = [nb.upconv2d(inner_nc * 2, inner_nc), nn.Conv2d(
                inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=True)]
        elif innermost:
            self.down = [nb.conv2d(False, input_nc, inner_nc, kernel_size=4, stride=2)]
            self.up   = [nb.upconv2d(inner_nc, outer_nc)]
        elif no_skip:
            self.down = [nb.conv2d(False, input_nc, inner_nc, kernel_size=4, stride=2)]
            self.up   = [nb.upconv2d(inner_nc*2, outer_nc*2)]
        else:
            self.down = [nb.conv2d(False, input_nc, inner_nc, kernel_size=4, stride=2)]
            self.up   = [nb.upconv2d(inner_nc * 2, outer_nc)]

        self.up        = nn.Sequential(*self.up)
        self.down      = nn.Sequential(*self.down)
        self.submodule = submodule

    def forward(self, x):
        x_inp = x
        self.x_enc = self.down(x_inp)
        if self.submodule is not None:
            out = self.submodule(self.x_enc)
        else:
            out = self.x_enc
        # if self.outermost:
        #     pdb.set_trace()
        self.x_dec = self.up(out)
        if self.outermost:
            return self.x_dec
        elif self.no_skip:
            return self.x_dec
        else:
            return torch.cat([x_inp, self.x_dec], 1)
