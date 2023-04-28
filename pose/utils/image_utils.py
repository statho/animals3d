'''
Code borrowed and adatped from
https://github.com/bearpaw/pytorch-pose/blob/master/pose/utils/transforms.py
'''

import cv2
import numpy as np
from PIL import Image


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def opencv_loader(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def crop(img, center, scale, res):
    '''
    Crop the bounding box and reshape image to res x res
    Args:
    - img    : ndarray containg the image
    - center : tuple (x,y) containing the central pixels of the bounding box
    - scale  : max(height, width) / 200
    - res    : tuple (W, H) of output resolution
    Return:
    - ndarray of size HxWx3
    '''
    # upper left point
    ul = np.array(transform([0, 0], center, scale, res, invert=1))
    # bottom right point
    br = np.array(transform(res, center, scale, res, invert=1))
    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)
    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]
    return cv2.resize(new_img, res)

def get_transform(center, scale, res, rot=0):
    '''
    Generate 3x3 transformation matrix
    Code adapted from: https://github.com/princeton-vl/pytorch_stacked_hourglass/
    Args:
    - center : tuple containing the center pixel (x,y)
    - scale  : max(H, W) / 200
    - res    : tuple containing output resolution (x_out, y_out)
    Return:
    - t      : transformation matric (3x3 ndarray)
    '''
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if rot != 0:
        # to match direction of rotation from cropping
        rot     = -rot
        rot_rad = rot*np.pi / 180
        sn, cs  = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat       = np.zeros((3,3))
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2, 2] = 1
        # need to rotate around center
        t_mat      = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv        = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot( t_inv, np.dot( rot_mat, np.dot(t_mat, t) ) )
    return t

def transform(pt, center, scale, res, rot=0, invert=0):
    '''
    Transform pixel location to different reference
    Code adapted from: https://github.com/princeton-vl/pytorch_stacked_hourglass/
    Args:
    - pt      : ndarray containing pixel (x_in, y_in) to transform
    - center  : ndarray containing the center pixel (x,y)
    - scale   : max(H, W) / 200
    - res     : tuple containing output resolution
    - invert  : when invert==1, compute the inverse transformation matrix
    Return:
    - new_pt  : ndarray with transformed pixel [x_new, y_new]
    '''
    t = get_transform(center=center, scale=scale, res=res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array( [pt[0], pt[1], 1.0] ).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int)

def inv_mat(mat):
    return np.linalg.pinv(np.array(mat).tolist() + [[0,0,1]])[:2]