'''
Training code -- Code borrowed and adapted from
https://github.com/nileshkulkarni/acsm/blob/master/acsm/experiments/pascal/csp.py

Example usage:
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --name horse_150 --category horse \
--kp_anno --scale_mesh --flip_train True --plot_scalars --display_visuals \
--use_pascal --use_coco

Running the above will train ACSM with the default settings using 150 labeled images for horses.

To include keypoint PLs from web images, also include the following arguments in the previous command.
--use_web_images --web_images_num <number of images with PL> --filter <selection criterion name>
'''

from __future__ import absolute_import, division, print_function
import numpy as np
import os.path as osp
from absl import app, flags
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
cm = plt.get_cmap('jet')
import torch
import torchvision
from torch.utils.data import DataLoader

import constants
from acsm.model import model_utils
from acsm.utils import mesh_utils
from acsm.model.network import ACSM
from acsm.data.objects import ImageDataset
from acsm.utils.visuals import vis_utils, vis_utils_cmr, visdom_render
from acsm.utils.train_utils import Trainer


flags.DEFINE_integer('seed',               0, 'seed for randomness')
flags.DEFINE_boolean('scale_mesh',     False, 'Scale mesh to unit sphere')
flags.DEFINE_boolean('use_pascal',     False, 'Use Pascal for training')
flags.DEFINE_boolean('use_coco',       False, 'Use part of COCO I labeled with KP for training')
flags.DEFINE_boolean('use_web_images', False, 'Use Flickr for training')
opts = flags.FLAGS


def data_sampler(dataset, shuffle):
    if shuffle:
        return torch.utils.data.RandomSampler(dataset)
    else:
        return torch.utils.data.SequentialSampler(dataset)


class ShapeTrainer(Trainer):

    def preload_model_data(self):
        opts = self.opts
        model_dir, self.mean_shape, self.mean_shape_np = model_utils.load_template_shapes(opts, device=self.device)
        dpm, parts_data, self.kp_vertex_ids = model_utils.init_dpm(model_dir, self.mean_shape, parts_file=f'acsm/cachedir/part_files/{opts.category}.txt')
        self.part_active_state, self.part_axis_init, self.part_perm = model_utils.load_active_parts(model_dir, self.save_dir, dpm, parts_data, suffix='')

    def init_render(self):
        opts = self.opts
        nkps = len(self.kp_vertex_ids)
        self.keypoint_cmap = [cm(i * 255 // nkps) for i in range(nkps)]
        faces_np = self.mean_shape['faces'].data.cpu().numpy()
        verts_np = self.mean_shape['sphere_verts'].data.cpu().numpy()

        model_obj_dir = osp.join(self.save_dir, 'model')
        vis_utils.mkdir(model_obj_dir)
        self.model_obj_path = osp.join(model_obj_dir, 'mean_{}.obj'.format(opts.category))
        self.verts_obj = self.mean_shape['verts']

        uv_sampler = mesh_utils.compute_uvsampler(verts_np, faces_np, tex_size=opts.tex_size)
        uv_sampler = torch.from_numpy(uv_sampler).float().to(self.device)
        self.uv_sampler = uv_sampler.view(-1, len(faces_np), opts.tex_size * opts.tex_size, 2)

        self.vis_rend = vis_utils_cmr.VisRenderer(opts.img_size, faces_np)
        self.renderer   = visdom_render.RendererWrapper(self.vis_rend, self.verts_obj, self.uv_sampler, self.offset_z,
                                                self.mean_shape_np, self.model_obj_path, self.keypoint_cmap, self.opts
        )

        vis_rend = vis_utils_cmr.VisRenderer(opts.img_size, faces_np)
        renderer_no_light = visdom_render.RendererWrapper(vis_rend, self.verts_obj, self.uv_sampler, self.offset_z,
                                                self.mean_shape_np, self.model_obj_path, self.keypoint_cmap, self.opts
        )
        renderer_no_light.vis_rend.set_light_status(False)
        renderer_no_light.vis_rend.set_bgcolor((255, 255, 255))
        self.renderer_no_light = renderer_no_light


    def init_dataset(self):
        opts = self.opts

        # initialize the dataset
        dset = ImageDataset(opts)
        self.dataloader = DataLoader(
                                        dset,
                                        sampler = data_sampler(dset, shuffle=True),
                                        batch_size = opts.batch_size,
                                        num_workers = opts.n_data_workers,
                                        drop_last = True,
                                        pin_memory = True
        )
        self.resnet_transform = torchvision.transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] )

        # initialize mesh stuff
        self.preload_model_data()
        if opts.scale_mesh:
            scale = 2. / torch.max(torch.nn.functional.pdist( self.mean_shape['verts'] )).item()
            self.mean_shape['verts'] = self.mean_shape['verts'] * scale
            self.mean_shape['verts'] = self.mean_shape['verts'] - self.mean_shape['verts'].mean(0)
            self.mean_shape_np['verts'] = self.mean_shape_np['verts'] * scale
            self.mean_shape_np['verts'] = self.mean_shape_np['verts'] - self.mean_shape_np['verts'].mean(0)

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
                    'part_axis':     self.part_axis_init,       # list of axis to init each part (e.g [1, 0, 0])
                    'part_perm':     self.part_perm,            # part permutation list -- need this for when image is flipped
                    'kp_perm':       torch.from_numpy(constants.QUAD_JOINT_PERM),
                    'mean_shape':    self.mean_shape,
                    'cam_location':  self.cam_location,
                    'offset_z':      self.offset_z,
                    'kp_vertex_ids': self.kp_vertex_ids,
                    'uv_sampler':    self.uv_sampler
        }
        self.model = ACSM(opts, init_stuff)
        self.model.to(self.device)

    def define_criterion(self):
        self.smoothed_factor_losses = self.sc_dict

    def set_input(self, batch):
        opts = self.opts
        self.codes_gt = {}
        self.inds = [k.item() for k in batch['inds']]
        self.codes_gt['inds'] = torch.LongTensor(self.inds).to(self.device)

        # get images
        input_imgs = batch['img'].type(self.Tensor)
        for b in range(input_imgs.size(0)):
            input_imgs[b] = self.resnet_transform(input_imgs[b])
        self.input_img_tensor = input_imgs.to(self.device)
        img_size = self.input_img_tensor.shape[-1]

        # get joints
        if 'kp' in batch:
            self.codes_gt['kp'] = batch['kp'].type(self.Tensor).to(self.device)

        # get masks
        if self.opts.train_with_masks and self.opts.mask_anno and 'mask' in batch:
            mask = batch['mask'].type(self.Tensor)
            mask = (mask > 0.5).float()
            self.mask = mask.to(self.device)
            self.mask_df = batch['mask_df'].type(self.Tensor).to(self.device) if 'mask_df' in batch else None
            if 'contour' in batch:
                self.codes_gt['contour'] = (batch['contour']).float().to(self.device)
                self.codes_gt['contour'] = (self.codes_gt['contour'] / img_size - 0.5) * 2
            if opts.flip_train:
                flipped_contour = (batch['flip_contour']).float().to(self.device)
                self.codes_gt['flip_contour'] = (flipped_contour / img_size - 0.5) * 2


    def forward(self):
        opts = self.opts
        feed_dict = {}
        codes_gt = self.codes_gt

        feed_dict['iter'] = self.real_iter
        feed_dict['inds'] = codes_gt['inds']
        feed_dict['img'] = self.input_img_tensor
        feed_dict['kp'] = codes_gt['kp'] if opts.kp_anno else None
        # mask annotations
        feed_dict['mask'] = self.mask.unsqueeze(1)           if opts.train_with_masks and opts.mask_anno else None
        feed_dict['mask_df'] = self.mask_df.unsqueeze(1)     if opts.train_with_masks and opts.mask_anno else None
        feed_dict['contour'] = codes_gt['contour']           if opts.train_with_masks and opts.mask_anno else None
        feed_dict['flip_contour'] = codes_gt['flip_contour'] if opts.train_with_masks and opts.mask_anno and opts.flip_train else None

        deform = self.real_iter > opts.warmup_deform_iter
        if deform and not self.use_articulations:
            self.use_articulations = True
            print('***\n=> Start predicting articulations!\n***')


        predictions, inputs = self.model(
            img          = feed_dict['img'],
            real_iter    = self.real_iter,
            deform       = deform,
            inds         = feed_dict['inds'],
            kp           = feed_dict['kp'],
            # mask annotations
            mask         = feed_dict['mask'],
            mask_df      = feed_dict['mask_df'],
            contour      = feed_dict['contour'],
            flip_contour = feed_dict['flip_contour'],
        )

        weight_dict = {}
        weight_dict['kp'] = opts.kp_loss_wt
        weight_dict['trans_reg'] = opts.trans_reg_loss_wt
        if opts.train_with_masks and opts.mask_anno:
            weight_dict['seg'] = opts.seg_mask_loss_wt
            weight_dict['con'] = opts.con_mask_loss_wt
            weight_dict['cov'] = opts.cov_mask_loss_wt
            # start using cycle, vis losses after warmup_pose_iter batches (as in original ACSM)
            if self.real_iter < opts.warmup_pose_iter:
                weight_dict['cyc'] = 0.0 * opts.reproject_loss_wt
                weight_dict['vis'] = 0.0 * opts.depth_loss_wt
                weight_dict['ent'] = 0.0 * opts.ent_loss_wt
            else:
                weight_dict['cyc'] = opts.reproject_loss_wt
                weight_dict['vis'] = opts.depth_loss_wt
                weight_dict['ent'] = opts.ent_loss_wt

        losses = self.model.compute_geometrics_losses(predictions, inputs, weight_dict)
        total_loss = 0
        for key, loss in losses.items():
            total_loss += loss
        self.total_loss = total_loss

        codes_pred = predictions
        for key in inputs.keys():
            codes_gt[key] = inputs[key]

        # convert joints to input resolution
        if self.codes_gt['kp'] is not None:
            codes_gt['kp_original'] = torch.cat(( (codes_gt['kp'][:,:,:2]*0.5 + 0.5)*self.img_size, codes_gt['kp'][:,:,2,None]), 2)

        self.loss_factors = losses
        for key, loss in losses.items():
            self.register_scalars({key: loss.item()})

        self.codes_gt = codes_gt
        self.codes_pred = codes_pred


    ### Visuals and Logging ###

    def get_current_visuals(self):
        visuals = self.visuals_to_save(count=1)[0]
        visuals.pop('ind')
        return visuals

    def visuals_to_save(self, count=None):
        opts = self.opts
        inds = self.codes_gt['inds'].data.cpu().numpy()

        batch_visuals = []
        img = self.codes_gt['img']
        mask = self.codes_gt['mask'] if 'mask' in self.codes_gt else None
        camera = self.codes_pred['cam']

        if count is None:
            count = min(opts.save_visual_count, len(self.codes_gt['img']))
        visual_ids = list(range(count))

        visual_ids = visual_ids[0:count]
        for b in visual_ids:
            visuals = {}
            visuals['ind'] = "{:04}".format(inds[b])

            if opts.train_with_masks and opts.mask_anno and opts.render_mask:
                mask_renders = self.codes_pred['mask_render'][b, :, None, ...].repeat(1, 3, 1, 1).data.cpu()
                mask_renders = (mask_renders.numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)
                visuals['mask_render'] = vis_utils.image_montage(mask_renders, nrow=min(3, opts.num_hypo_cams // 3 + 1) )

            if opts.train_with_masks and opts.mask_anno and opts.render_depth:
                all_depth_hypo = (self.codes_pred['mask_render'][b] * self.codes_pred['depth'][b])[:, None, :, :].repeat(1, 3, 1, 1).data.cpu() / 50.0
                all_depth_hypo = (all_depth_hypo.numpy() *255).astype(np.uint8).transpose(0, 2, 3, 1)
                visuals['all_depth'] = vis_utils.image_montage(all_depth_hypo, nrow=min(3, opts.num_hypo_cams // 3 + 1) )

            vis_cam_hypotheses = self.renderer.render_all_hypotheses(
                camera[b],
                probs=self.codes_pred['cam_probs'][b],
                verts=self.codes_pred['verts'][b]
            )
            visuals.update(vis_cam_hypotheses)

            # get input images
            visuals['z_img'] = vis_utils.tensor2im(vis_utils.undo_resnet_preprocess(img.data[b, None, :, :, :]))
            camera_ind = torch.argmax(self.codes_pred['cam_probs'][b].squeeze()).item()

            # get joints
            if 'kp_original' in self.codes_gt:
                kp_viz = self.codes_gt['kp_original'][b,:,2]
                # GT joints
                visuals['img_kp'] = vis_utils_cmr.draw_keypoint_on_image(
                    visuals['z_img'], self.codes_gt['kp_original'][b,:,:2], kp_viz, self.keypoint_cmap
                )
                # predicted joints
                visuals['img_kp_rp'] = vis_utils_cmr.draw_keypoint_on_image(
                    visuals['z_img'], (self.codes_pred['kp_project'][b][camera_ind]*0.5+0.5)*opts.img_size, kp_viz, self.keypoint_cmap
                )

            # get masks and contours
            if opts.train_with_masks and opts.mask_anno and 'seg_mask' in self.codes_pred:
                visuals['pred_mask'] = vis_utils.tensor2im(self.codes_pred['seg_mask'].data.repeat(1, 3, 1, 1)[b, None, :, :, :])
            if opts.train_with_masks and opts.mask_anno and mask is not None:
                visuals['z_mask'] = vis_utils.tensor2im(mask.data.repeat(1, 3, 1, 1)[b, None, :, :, :])
                visuals['texture_copy'] = vis_utils_cmr.copy_texture_from_img(mask[b], img[b], self.codes_pred['project_points'][b][camera_ind])
                contour_indx = (opts.img_size * (self.codes_gt['contour'][b] * 0.5 + 0.5)).data.cpu().numpy().astype(np.int)
                visuals['contour'] = self.renderer.visualize_contour_img(contour_indx, opts.img_size)

            # rendered shape
            rendered_shape = self.vis_rend(verts=self.codes_pred['verts'][b][camera_ind], cams=camera[b][camera_ind])
            img_temp = visuals['z_img'][:]
            rendered_mask = (rendered_shape==0).astype(float)
            img_temp = img_temp * rendered_mask
            img_with_shape = img_temp + rendered_shape
            visuals['rendered_shape'] = img_with_shape

            batch_visuals.append(visuals)
        return batch_visuals

    def get_current_points(self):
        pts_dict = {}
        return pts_dict

    def get_current_scalars(self):
        for param_group in self.optimizer.param_groups:
            learning_rate = param_group['lr']
            break
        loss_dict = {
            'total_loss'    : self.smoothed_total_loss,
            'iter_frac'     : self.real_iter * 1.0 / self.total_steps,
            'learning_rate' : learning_rate
        }
        for k in self.sc_dict.keys():
            loss_dict['loss_' + k] = self.sc_dict[k]
        return loss_dict



def main(_):
    seed = opts.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    opts.split = 'train'
    trainer = ShapeTrainer(opts)
    trainer.init_training()
    trainer.train()


if __name__ == '__main__':
    app.run(main)