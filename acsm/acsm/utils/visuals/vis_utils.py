'''
Code from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
'''

from __future__ import print_function
import os
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import collections
import torch
import torch.nn as nn

def saveTensorAsImage(im_tensor, im_path):
    '''
    3 X H X W tensor
    '''
    image = tensor2im2(im_tensor)
    save_image(image, im_path)
    return image

def torch2numpy(tensor):
    return tensor.data.cpu().numpy()

def tensor2im(image_tensor, imtype=np.uint8):
    ''' Converts a Tensor into a Numpy array '''
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    return image_numpy.astype(imtype)

def tensor2im2(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    return image_numpy.astype(imtype)

def tensor2kps(kp_tensor):
    '''
    Input is either B x N x 3 or N x 2 -- 3rd dim is visibility
    '''
    kps_numpy = kp_tensor[0].cpu().float().numpy()
    if kps_numpy.shape[1] == 3:
        kps_numpy = kps_numpy[:, :2]
    return kps_numpy

def tensor2verts(vert_tensor):
    if vert_tensor.dim() == 2: # N x 3
        vert_numpy = vert_tensor.cpu().float().numpy()
    else: # B x N x 3
        vert_numpy = vert_tensor[0].cpu().float().numpy()
    return vert_numpy

def tensor2im_batch(image_tensor, num_batch, imtype=np.uint8):
    images = []
    for i in range(num_batch):
        image_numpy = image_tensor[i].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
        images.append(image_numpy.astype(imtype))

    images = np.hstack(images)
    return images

class ImageSmoothing(nn.Module):

    def __init__(self, kernel_size):
        super(ImageSmoothing, self, ).__init__()
        kernel = torch.zeros((1,1,kernel_size, kernel_size)) + 1
        self.kernel = kernel.cuda()
        self.conv = torch.nn.functional.conv2d
        return
    def forward(self, image):
        output = self.conv(image, weight=self.kernel, stride=1, padding=2)
        return output

def image_montage(images, nrow):
    img_size    = images[0].shape
    ncol        = np.ceil(len(images)*1.0/nrow)
    nrow , ncol = int(ncol), int(nrow)
    n_channels  = images[0].shape[-1]
    big_image = np.zeros((img_size[0]* nrow, img_size[1]*ncol, n_channels)).astype(images[0].dtype)
    index = 0
    for rx in range(nrow):
        for cx in range(ncol):
            big_image[img_size[0]*rx: img_size[0]*(rx+1), img_size[1]*cx: img_size[1]*(cx+1),:] = images[index]
            index += 1
            if index == len(images):
                return big_image
    return big_image

def undo_resnet_preprocess(image_tensor):
    image_tensor = image_tensor.clone()
    image_tensor.narrow(1,0,1).mul_(.229).add_(.485)
    image_tensor.narrow(1,1,1).mul_(.224).add_(.456)
    image_tensor.narrow(1,2,1).mul_(.225).add_(.406)
    return image_tensor

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" % (method.ljust(spacing), processFunc(str(getattr(object, method).__doc__)))
                                                                                            for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)