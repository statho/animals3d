'''
Code borrowed and adapted from
https://github.com/bearpaw/pytorch-pose/blob/master/pose/utils/imutils.py
'''

import torch
import numpy as np
from PIL import Image


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array".format(type(tensor)))
    return tensor

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray

class GenerateHeatmap():
    def __init__(self, output_res, num_parts, conf_list):
        self.output_res = output_res
        self.num_parts  = num_parts
        self.conf_list  = conf_list
        sigma           = self.output_res / 64
        self.sigma      = sigma
        size   = 6*sigma + 3
        x      = np.arange(0, size, 1, float)
        y      = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, kpts):
        hms     = np.zeros(shape=(self.num_parts, self.output_res, self.output_res), dtype=np.float32)
        viz_arr = np.zeros(shape=(self.num_parts, 1), dtype=np.int8)
        sigma   = self.sigma
        for idx, kpt in enumerate(kpts):
            # use threshold to keep only the labels we want
            visibility   = 1 if kpt[2]>=self.conf_list[idx] else 0
            viz_arr[idx] = visibility
            if visibility > 0:
                x, y = int(kpt[0]), int(kpt[1])
                if x<0 or y<0 or x>=self.output_res or y>=self.output_res:
                    viz_arr[idx] = 0
                    continue
                ul    = int(x-3*sigma - 1), int(y-3*sigma - 1)
                br    = int(x+3*sigma + 2), int(y+3*sigma + 2)
                c,d   = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                a,b   = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]
                cc,dd = max(0, ul[0]), min(br[0], self.output_res)
                aa,bb = max(0, ul[1]), min(br[1], self.output_res)
                hms[idx, aa:bb,cc:dd] = np.maximum(hms[idx, aa:bb,cc:dd], self.g[a:b,c:d])
        return hms, viz_arr


def gaussian(x, a, b, c, d=0):
    return a * np.exp(-(x - b)**2 / (2 * c**2)) + d

def color_heatmap(x):
    color            = np.zeros((x.shape[0], x.shape[1], 3))
    color[:,:,0]     = gaussian(x, 0.5, 0.6, 0.2) + gaussian(x, 1, 0.8, 0.3)
    color[:,:,1]     = gaussian(x, 1, 0.5, 0.3)
    color[:,:,2]     = gaussian(x, 1, 0.2, 0.3)
    color[color > 1] = 1
    return (color * 255).astype(np.uint8)

def sample_with_heatmap(inp, out, num_rows=2, parts_to_show=None):
    '''
    Args:
    - inp: torch.Tensor of size [3, H, W]
    - out: torch.Tensor of size [NJ, out_H, out_W]
    '''
    if parts_to_show is None:
        parts_to_show = np.arange(out.shape[0])
    img = np.transpose(to_numpy(inp * 255), (1, 2, 0))
    out = to_numpy(out)
    # generate a single image to display input/output pair
    size      = img.shape[0] // num_rows
    num_cols  = int(np.ceil(float(len(parts_to_show)) / num_rows))
    full_img  = np.zeros((img.shape[0], size * (num_cols + num_rows), 3), np.uint8)
    full_img[:img.shape[0], :img.shape[1]] = img
    img       = Image.fromarray(np.uint8(img)).convert('RGB')
    inp_small = np.array(img.resize(size=(size, size), resample=Image.BILINEAR))
    # set up heatmap display for each part
    for i, part_idx in enumerate(parts_to_show):
        outp        = Image.fromarray(out[part_idx])
        out_resized = np.array(outp.resize(size=(size,size)))
        out_resized = out_resized.astype(float)
        out_img     = inp_small.copy() * 0.3
        color_hm    = color_heatmap(out_resized)
        out_img    += color_hm * 0.7
        col_offset  = (i % num_cols + num_rows) * size
        row_offset  = (i // num_cols) * size
        full_img[row_offset:row_offset + size, col_offset:col_offset + size] = out_img
    return full_img

def batch_with_heatmap(inputs, outputs, mean=torch.Tensor([0.485, 0.456, 0.406]),
                        std=torch.Tensor([0.229, 0.224, 0.225]), num_rows=2, parts_to_show=None):
    '''
    Args:
    - inputs : torch.Tensor of size [B, 3, H, W]
    - outputs : torch.Tenosr of size [B, NJ, out_H, out_W]
    - mean : per channel mean subtracted from the images
    - std : per channel std that images were divided with
    - num_rows : number of rows to use for plotting
    - parts_to_show : keypoints to plot
    '''
    batch_img = []
    for i in range(min(inputs.size(0), 4)):
        inp  = inputs[i] * std.view(3, 1, 1).expand_as(inputs[i]) + mean.view(3, 1, 1).expand_as(inputs[i])    # unnormalize
        outp = outputs[i]
        batch_img.append( sample_with_heatmap( inp.clamp(min=0, max=1), outp, num_rows=num_rows, parts_to_show=parts_to_show ) )
    return np.concatenate(batch_img)