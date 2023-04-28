"""
Mesh stuff from https://github.com/akanazawa/cmr/
"""
from __future__ import absolute_import, division, print_function
import itertools
import numpy as np

from acsm.utils import geom_utils

def get_spherical_coords(X):
    # X is N x 3
    uv = geom_utils.convert_3d_to_uv_coordinates(X)
    # Normalize between -1 to 1. Since grid sampling requires it to be so.
    uv = 2*uv -1
    return uv

def compute_uvsampler(verts, faces, tex_size=2):
    """
    For this mesh, pre-computes the UV coordinates for F x T x T points.
    Returns F x T x T x 2
    """
    alpha = np.arange(tex_size, dtype=np.float) / (tex_size-1)
    beta  = np.arange(tex_size, dtype=np.float) / (tex_size-1)
    # Barycentric coordinate values
    coords = np.stack([p for p in itertools.product(*[alpha, beta])])
    vs = verts[faces]
    # Compute alpha, beta (this is the same order as NMR)
    v2   = vs[:, 2]
    v0v2 = vs[:, 0] - vs[:, 2]
    v1v2 = vs[:, 1] - vs[:, 2]
    # F x 3 x T**2
    samples = np.dstack([v0v2, v1v2]).dot(coords.T) + v2.reshape(-1, 3, 1)
    # F x T*2 x 3 points on the sphere
    samples = np.transpose(samples, (0, 2, 1))
    # Now convert these to uv.
    uv = get_spherical_coords(samples.reshape(-1, 3))
    # uv = uv.reshape(-1, len(coords), 2)
    uv = uv.reshape(-1, tex_size, tex_size, 2)
    return uv