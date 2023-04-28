'''
Visualize the predicted shapes.

Example usage:
CUDA_VISIBLE_DEVICES=0 python scripts/visualize.py --scale_mesh --dataset pascal --category horse --vis_num 20

Running the above will generate visualizations for 20 random images with horses from Pascal dataset.
'''

from __future__ import absolute_import, division, print_function

import os
import random
import imageio
import numpy as np
from absl import app, flags
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader

import constants
from acsm.model import model_utils
from acsm.utils.test_utils import Tester
from acsm.model.network import ACSM
from acsm.data.objects import ImageDatasetEval
from acsm.utils.visuals import vis_utils, vis_utils_cmr


flags.DEFINE_integer('seed', 0, 'seed for randomness')
flags.DEFINE_integer('vis_num', 10, 'number of images to save visualizations for')
flags.DEFINE_string('dataset', 'pascal', 'dataset to visualize')
flags.DEFINE_boolean('scale_mesh', False, 'Scale mesh to unit sphere and translate its mean position to (0,0,0)')
opts = flags.FLAGS


class Visualizer(Tester):

    def preload_model_data(self):
        opts = self.opts
        model_dir, self.mean_shape, _ = model_utils.load_template_shapes(opts, device=self.device)
        dpm, parts_data, self.kp_vertex_ids = model_utils.init_dpm(model_dir, self.mean_shape, parts_file=f'acsm/cachedir/part_files/{opts.category}.txt')
        self.part_active_state, self.part_axis_init, self.part_perm = model_utils.load_active_parts(model_dir, self.save_dir, dpm, parts_data, suffix='')

    def init_render(self):
        opts = self.opts
        verts_np = self.mean_shape['sphere_verts'].cpu().numpy()
        faces_np = self.mean_shape['faces'].cpu().numpy()

        self.uv_sampler = None
        if opts.train_with_masks and opts.mask_anno:
            from acsm.utils import mesh_utils
            uv_sampler = mesh_utils.compute_uvsampler(verts_np, faces_np, tex_size=opts.tex_size)
            uv_sampler = torch.from_numpy(uv_sampler).float().to(self.device)
            self.uv_sampler = uv_sampler.view(-1, len(faces_np), opts.tex_size * opts.tex_size, 2)

        self.vis_rend = vis_utils_cmr.VisRenderer(opts.img_size, faces_np)
        self.vis_rend.set_bgcolor((1., 1., 1.))
        self.vis_rend.set_light_dir([0, 1, -1], 0.38)
        self.vis_rend.renderer.image_size = 256
        self.vis_rend.set_light_status(True)

    def init_dataset(self):
        opts = self.opts

        # initialize the dataset
        dset = ImageDatasetEval(opts)
        self.dataloader = DataLoader(dset, batch_size = opts.batch_size, num_workers = opts.n_data_workers, pin_memory = True)
        self.resnet_transform = torchvision.transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] )

        # initialize mesh stuff
        self.preload_model_data()
        if opts.scale_mesh:
            scale = 2. / torch.max(torch.nn.functional.pdist( self.mean_shape['verts'] )).item()
            self.mean_shape['verts'] = self.mean_shape['verts'] * scale
            self.mean_shape['verts'] = self.mean_shape['verts'] - self.mean_shape['verts'].mean(0)

        # initialize renderer
        self.init_render()


    def define_model(self):
        opts           = self.opts
        self.img_size  = opts.img_size
        self.offset_z  = 5.0
        self.gpu_count = 1
        init_stuff     = {
                    'alpha':         self.mean_shape['alpha'],  # (|V|, num_parts) assignment of verts to a part
                    'active_parts':  self.part_active_state,    # boolean list for active parts
                    'part_axis':     self.part_axis_init,       # list of axis to init each part (e.g. [1, 0, 0])
                    'part_perm':     self.part_perm,            # part permutation list -- need this for when image is flipped
                    'kp_perm':       torch.from_numpy(constants.QUAD_JOINT_PERM),
                    'mean_shape':    self.mean_shape,
                    'cam_location':  self.cam_location,
                    'offset_z':      self.offset_z,
                    'kp_vertex_ids': self.kp_vertex_ids,
                    'uv_sampler':    self.uv_sampler
        }
        self.model = ACSM(opts, init_stuff)
        self.is_dataparallel_model = self.dataparallel_model(self.opts.iter_num)
        if self.is_dataparallel_model:
            self.model = torch.nn.DataParallel(self.model)
        self.load_network(self.model, self.opts.iter_num)
        self.model.to(self.device)
        self.model.eval()

    def set_input(self, batch):
        self.codes_gt = {}
        self.img_name = batch['img_name']
        # get images
        input_imgs = batch['img'].type(self.Tensor)
        for b in range(input_imgs.size(0)):
            input_imgs[b] = self.resnet_transform(input_imgs[b])
        self.imgs  = input_imgs.to(self.device)

    def predict(self):
        with torch.no_grad():
            predictions = self.model.predict(self.imgs, deform=True) if not self.is_dataparallel_model else self.model.module.predict(self.imgs, deform=True)
        self.codes_pred = predictions
        batch_size = len(self.imgs)
        camera, verts, kp_project_selected = [], [], []
        for b in range(batch_size):
            max_ind = torch.argmax(self.codes_pred['cam_probs'][b], dim=0).item()
            camera.append(self.codes_pred['cam'][b][max_ind])
            verts.append(self.codes_pred['verts'][b][max_ind])
            kp_project_selected.append( self.codes_pred['kp_project'][b][max_ind] )
        self.codes_pred['camera_selected'] = torch.stack(camera)
        self.codes_pred['verts_selected'] = torch.stack(verts)
        self.codes_pred['kp_project_selected'] = torch.stack(kp_project_selected)


    def test(self):
        opts   = self.opts
        save_dir = f'../demo/{opts.dataset}_{opts.name}_{opts.iter_num}' if opts.iter_num > 0 else f'../demo/{opts.dataset}_{opts.category}'
        print('=> Saving images in {}'.format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for ii, batch in enumerate(self.dataloader):
            if ii >= opts.vis_num:
                break
            self.set_input(batch)
            self.predict()
            img_name = self.img_name[0].split('.')[0]

            # get predicted shape and viewpoint
            verts = self.codes_pred['verts_selected'][0]
            camera = self.codes_pred['camera_selected'][0]

            # save input image
            img = vis_utils.tensor2im( vis_utils.undo_resnet_preprocess(self.imgs.data[0, None, :, :, :]) )
            plt.imsave( f'{save_dir}/{img_name}.jpg', img )

            # shape from predicted camera view
            shape = self.vis_rend(verts=verts, cams=camera)
            plt.imsave(f'{save_dir}/{img_name}_shape.jpg', shape)

            # rotate shape and render it from different viewpoints
            frames = []
            for angle in range(0, 360, 4):
                rot_shape = self.vis_rend.diff_vp(verts=verts, angle=angle, axis=[0, 1, 0], cam=camera)
                frames.append(rot_shape)
            imageio.mimsave(f'{save_dir}/{img_name}.gif', frames, fps=30)


def main(_):
    assert (opts.category in ['horse', 'cow', 'sheep'] and opts.dataset in ['pascal', 'animal_pose']) or \
                opts.category in ['giraffe', 'bear'] and opts.dataset=='coco', 'Error in category/dataset arguments'

    seed = opts.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    opts.split = 'val'
    opts.batch_size = 1
    opts.padding_frac = 0

    tester = Visualizer(opts)
    tester.init_dataset()
    tester.define_model()
    tester.test()


if __name__ == '__main__':
    app.run(main)