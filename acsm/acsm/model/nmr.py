"""
Code borrowed from
https://github.com/akanazawa/cmr/
"""
from __future__ import absolute_import, division, print_function
import torch
import warnings
warnings.filterwarnings("ignore")

import neural_renderer
from acsm.utils import geom_utils


class NeuralRenderer(torch.nn.Module):
    """
    This is the core pytorch function to call.
    Every torch NMR has a chainer NMR.
    Only fwd/bwd once per iteration.
    """
    def __init__(self, img_size=256, device=0):
        super(NeuralRenderer, self).__init__()
        self.renderer = neural_renderer.Renderer()
        # Set a default camera to be at (0, 0, -2.732)
        self.renderer.eye = [0, 0, -2.732]
        self.proj_fn      = geom_utils.orthographic_proj_withz
        self.offset_z     = 5.

        # Adjust the core renderer
        self.renderer.perspective = False
        self.renderer.image_size  = img_size

        # Make it a bit brighter for vis
        self.renderer.light_intensity_ambient = 0.8

    def ambient_light_only(self):
        # Make light only ambient.
        self.renderer.light_intensity_directional = 0.
        self.renderer.light_intensity_ambient = 1.

    def set_bgcolor(self, color):
        self.renderer.background_color = color

    def project_points(self, verts, cams):
        proj = self.proj_fn(verts, cams)
        return proj[:, :, :2]

    def forward(self, vertices, faces, cams, textures=None, depth_only=False):
        verts = self.proj_fn(vertices, cams, offset_z=self.offset_z)
        vs    = verts.clone()
        # flipping the y-axis here to make it align with the image coordinate system!
        vs[:, :, 1] *= -1
        fs    = faces.clone()
        if depth_only:
            self.mask_only = False
            depth = self.renderer.render_depth(vs, fs)
            return depth
        elif textures is None:
            self.mask_only = True
            masks = self.renderer.render_silhouettes(vs, fs)
            return masks
        else:
            self.mask_only = False
            ts = textures.clone()
            imgs = self.renderer.render(vs, fs, ts)[0]  #only keep rgb, no alpha and depth
            return imgs

    def set_light_status(self, use_lights):
        renderer = self.renderer.renderer
        renderer.use_lights = use_lights