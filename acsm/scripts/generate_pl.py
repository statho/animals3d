'''
Generate keypoint pseudo-labels using ACSM.

Example usage:
CUDA_VISIBLE_DEVICES=0 python scripts/generate_pl.py --scale_mesh --category sheep --name <name> --iter_num <iter_num>

Running the above will genrate and save keypoint PLs on the downloaded web images for sheeps.
'''

from __future__ import absolute_import, division, print_function

import os
import json
import copy
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from absl import app, flags
from collections import defaultdict
from torch.utils.data import DataLoader

import constants
from acsm.model import model_utils
from acsm.model.network import ACSM
from acsm.utils.test_utils import Tester
from acsm.datasets.objects import UnlabeledDataset
from acsm.utils.visuals import vis_utils_cmr


flags.DEFINE_integer('seed',           0, 'seed for randomness')
flags.DEFINE_boolean('scale_mesh', False, 'Scale mesh to unit sphere and translate its mean position to (0,0,0)')
opts = flags.FLAGS


class GeneratePL(Tester):

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
        self.vis_rend.renderer.image_size = 1024
        self.vis_rend.set_light_status(True)

    def init_dataset(self):
        opts = self.opts

        # initialize the dataset
        dset = UnlabeledDataset(opts)
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
        input_imgs = batch['img'].type(self.Tensor)
        for b in range(input_imgs.size(0)):
            input_imgs[b] = self.resnet_transform(input_imgs[b])
        self.imgs = input_imgs.to(self.device)
        self.img_id = batch['img_id']  # list with ids
        self.trans = batch['trans'].numpy().astype('float32')
        self.scale = batch['scale'].numpy().astype('float32')

    def predict(self):
        with torch.no_grad():
            predictions = self.model.predict(self.imgs, deform=True) if not self.is_dataparallel_model else self.model.module.predict(self.imgs, deform=True)
        self.codes_pred = predictions
        batch_size = len(self.imgs)
        kp_project_selected = []
        for b in range(batch_size):
            max_ind = torch.argmax(self.codes_pred['cam_probs'][b], dim=0).item()
            kp_project_selected.append( self.codes_pred['kp_project'][b][max_ind] )
        self.codes_pred['kp_project_selected'] = torch.stack(kp_project_selected)

    def test(self):
        img_id_list, kp_list = [], []
        for batch in tqdm(self.dataloader):
            self.set_input(batch)
            self.predict()

            # reprojected keypoints
            kp_pred = self.codes_pred['kp_project_selected'] # (bs, 16, 2)
            kp_pred = (kp_pred * 0.5 + 0.5) * self.img_size
            kp_pred_np = kp_pred.cpu().numpy()
            kp_pred_org = (kp_pred_np / self.scale[:,None,None]) + self.trans[:,None,:]

            img_id_list += self.img_id
            kp_list     += kp_pred_org.tolist()

        # add keypoint pseudo-labels to the annotations
        joint_dict = defaultdict(list)
        for ind, img_id in enumerate(img_id_list):
            for keypoint_pred in kp_list[ind]:
                joint_dict[img_id].append([ keypoint_pred[0], keypoint_pred[1] ])

        bbox_file = f'../data/yfcc100m/labels_0/{self.opts.category}_bbox.json'
        with open(bbox_file) as f:
            annos = json.load(f)
        for anno in annos:
            anno['joints'] = copy.deepcopy(joint_dict[anno['img_id']])

        name = self.opts.name.split('/')[-1] if len(self.opts.name)>0 else self.opts.category
        save_path = f'../data/yfcc100m/labels'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_name = f'{save_path}/{name}_pl_3d.json'
        with open(save_name, 'w') as f:
            json.dump(annos, f)



def main(_):
    seed = opts.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    tester = GeneratePL(opts)
    tester.init_dataset()
    tester.define_model()
    tester.test()


if __name__ == '__main__':
    app.run(main)