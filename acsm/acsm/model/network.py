"""
Code borrowed and adapted from
https://github.com/nileshkulkarni/acsm/blob/master/acsm/nnutils/icn_net.py
"""

from __future__ import absolute_import, division, print_function
import itertools
import numpy as np
from absl import flags
import torch
import torch.nn as nn

from acsm.utils import geom_utils
from acsm.model import camera as cb
from acsm.model.utils import net_blocks as nb
from acsm.model.nmr import NeuralRenderer


flags.DEFINE_boolean('kp_anno',         False, 'use keypoint annotations (during training or evaluation)')
flags.DEFINE_boolean('mask_anno',       False, 'use mask annotations (during training or evaluation)')
flags.DEFINE_boolean('sfm_anno',        False, 'use camera from sfm (pseudo-gt) -- used for evaluation')
flags.DEFINE_boolean('train_with_masks',False, 'train with mask annotations (not used in experiments in the paper)')

flags.DEFINE_integer('nz_feat',             200, 'encoded dimension of image features size')
flags.DEFINE_integer('warmup_deform_iter', 3000, 'do not predict articulation in the first batches')
flags.DEFINE_float('kp_loss_wt',            3.0, 'reprojection loss for keypoints')
flags.DEFINE_float('trans_reg_loss_wt',    10.0, 'transform regularization loss weight')

flags.DEFINE_boolean('render_mask',       False, 'visualize rendered mask')
flags.DEFINE_boolean('render_depth',      False, 'visualize depth rendering')
flags.DEFINE_boolean('render_uv',         False, 'render uv to add loss for uv prediction')

# flags for training with mask annotations
flags.DEFINE_boolean('resnet_style_decoder', False,  'Use resnet style decoder for uvs')
flags.DEFINE_integer('resnet_blocks',            4, 'Encodes using resnet to layer')
flags.DEFINE_integer('remove_skips',            -1, 'Removes skip starting after which layer of UNet')
flags.DEFINE_integer('warmup_pose_iter',        -1, '1) Do not use cycle+vis losses, \
                                                     2) (if using camere multiplex) warm-up camera multiplex for the first batches')
flags.DEFINE_integer('warmup_semi_supv',         0, 'When only a subset of the training data have keypoints use only the subset with keypoints for some iterations')
flags.DEFINE_float('reproject_loss_wt',        1.0, 'Cycle loss.')
flags.DEFINE_float('cov_mask_loss_wt',        10.0, 'Mask loss wt.')
flags.DEFINE_float('con_mask_loss_wt',         0.1, 'Mask loss wt.')
flags.DEFINE_float('seg_mask_loss_wt',         1.0, 'Predicted Seg Mask loss wt.')
flags.DEFINE_float('depth_loss_wt',            1.0, 'Depth loss wt.')
flags.DEFINE_float('ent_loss_wt',             0.05,'Entropy loss wt. (entropy term over the probabilities of camera hypothesis')
flags.DEFINE_float('rot_reg_loss_wt',          0.1, 'Rotation Reg loss wt. (maximize a pairwise-distance between predicted rotations, for all the predicted hypothesis of an instance)')



class Encoder(nn.Module):

    def __init__(self, input_shape, n_blocks=4, nz_feat=100, batch_norm=True):
        super(Encoder, self).__init__()
        self.resnet_conv = nb.ResNetConv(n_blocks=n_blocks)
        self.enc_conv1   = nb.conv2d( batch_norm, 512, 256, stride=2, kernel_size=4 )
        nc_input         = 256 * (input_shape[0] // 64) * (input_shape[1] // 64)
        self.enc_fc      = nb.fc_stack(nc_input, nz_feat, 2)
        nb.net_init(self.enc_conv1)

    def forward(self, img):
        resnet_feat      = self.resnet_conv(img)
        self.resnet_feat = resnet_feat
        out_enc_conv1    = self.enc_conv1(resnet_feat)
        out_enc_conv1    = out_enc_conv1.view(img.size(0), -1)
        feat = self.enc_fc(out_enc_conv1)
        return feat


class PartMembership(nn.Module):
    '''
    init_val n_verts x nparts
    '''
    def __init__(self, init_val):
        super(PartMembership, self).__init__()
        self.init_val   = init_val
        self.membership = nn.Parameter(self.init_val * 0)
        self.membership.data.normal_(0, 0.0001)

    def forward(self):
        return self.init_val


class ACSM(nn.Module):
    def __init__(self, opts, init_stuff):
        super(ACSM, self).__init__()

        self.opts        = opts
        self.nc_encoder  = 256
        self.uv_pred_dim = 3
        self.real_iter   = 0
        part_init        = { 'active_parts': init_stuff['active_parts'], 'part_axis': init_stuff['part_axis'] }
        self.kp_perm     = init_stuff['kp_perm']
        self.part_perm   = init_stuff['part_perm']

        self.img_encoder = Encoder( (opts.img_size, opts.img_size), nz_feat=opts.nz_feat )


        # network to train with mask annotations -- not used in paper experiments
        if opts.train_with_masks:
            if opts.resnet_style_decoder:
                from acsm.model.utils import resunet
                self.unet_gen = resunet.ResNetConcatGenerator(
                    input_nc  = 3,
                    output_nc = self.uv_pred_dim + 1,
                    n_blocks  = opts.resnet_blocks
                )
            else:
                from acsm.model.utils import unet
                self.unet_gen = unet.UnetConcatGenerator(
                    input_nc     = 3,
                    output_nc    = self.uv_pred_dim + 1,
                    num_downs    = 5,
                    remove_skips = opts.remove_skips
                )
                self.unet_innermost = self.unet_gen.get_inner_most()
            img_size  = (int(opts.img_size * 1.0), int(opts.img_size * 1.0))
            self.grid = nb.get_img_grid(img_size).repeat(1, 1, 1, 1)
            from acsm.model.utils import uv_to_3d
            self.uv2points = uv_to_3d.UVTo3D(self.mean_shape)


        # single camera or camera-multiplex
        if opts.num_hypo_cams == 1:
            self.cam_predictor = cb.SingleCamPredictor(
                                                        nz_feat        = opts.nz_feat,
                                                        scale_lr_decay = opts.scale_lr_decay,
                                                        scale_bias     = opts.scale_bias,
                                                        no_trans       = opts.no_trans,
                                                        part_init      = part_init,            # {'active parts': , 'part_axis': }
            )
        else:
            self.cam_predictor = cb.MultiCamPredictor(
                                                        nz_feat        = opts.nz_feat,
                                                        num_cams       = opts.num_hypo_cams,
                                                        scale_lr_decay = opts.scale_lr_decay,
                                                        scale_bias     = opts.scale_bias,
                                                        no_trans       = opts.no_trans,
                                                        part_init      = part_init,
                                                        euler_range    = [opts.az_euler_range, opts.el_euler_range, opts.cyc_euler_range] # [30, 20, 20]
            )

        self.mean_shape = init_stuff['mean_shape']
        self.init_nmrs()
        self.cam_location = init_stuff['cam_location']
        self.offset_z = init_stuff['offset_z']
        self.kp_vertex_ids = init_stuff['kp_vertex_ids']
        self.part_membership = PartMembership(init_stuff['alpha'])


    def init_nmrs(self):
        opts      = self.opts
        devices   = list(range(torch.cuda.device_count())) if torch.cuda.device_count() > 1 else [0]
        renderers = []
        for device in devices:
            multi_renderer_mask  = nn.ModuleList( [ NeuralRenderer(opts.img_size) for _ in range(opts.num_hypo_cams) ] )
            multi_renderer_depth = nn.ModuleList( [ NeuralRenderer(opts.img_size) for _ in range(opts.num_hypo_cams) ] )
            renderers.append( {'mask': multi_renderer_mask, 'depth': multi_renderer_depth} )
        self.renderers = renderers

    def flip_inputs(self, inputs):
        '''
        Append flipped version of image, kp (and optionally mask) to the input
        '''
        device = inputs['img'].get_device()
        flip_img = geom_utils.flip_image(inputs['img'])
        inputs['img'] = torch.cat([inputs['img'], flip_img])
        inputs['inds'] = torch.cat([inputs['inds'], inputs['inds'] + 10000])
        if self.opts.train_with_masks and self.opts.mask_anno:
            flip_mask = geom_utils.flip_image(inputs['mask'])
            inputs['mask'] = torch.cat([inputs['mask'], flip_mask])
            flip_mask_df = geom_utils.flip_image(inputs['mask_df'])
            inputs['mask_df'] = torch.cat([inputs['mask_df'], flip_mask_df])
            inputs['contour'] = torch.cat([inputs['contour'], inputs['flip_contour']])
        if self.opts.kp_anno:
            kp_perm             = self.kp_perm.to(device)
            flipped_kp          = inputs['kp'].clone()
            flipped_kp[:, :, 0] = -1 * flipped_kp[:, :, 0]
            flipped_kp          = flipped_kp[:, kp_perm, :]
            inputs['kp']        = torch.cat([inputs['kp'], flipped_kp])
        return inputs

    def flip_predictions(self, codes_pred, true_size):
        device = codes_pred['cam_probs'].get_device()

        # if self.opts.multiple_cam:
        keys_to_copy = ['cam_probs']
        for key in keys_to_copy:
            codes_pred[key] = torch.cat([codes_pred[key][:true_size], codes_pred[key][:true_size]])

        part_perm = self.part_perm.to(device)
        # if self.opts.multiple_cam:
        keys_to_copy = ['part_transforms']
        for key in keys_to_copy:
            mirror_transforms_swaps = codes_pred[key][:true_size][:, :, part_perm, :]
            codes_pred[key]         = torch.cat([codes_pred[key][:true_size], mirror_transforms_swaps])

        # mirror rotation
        camera = codes_pred['cam'][:true_size]
        # if self.opts.multiple_cam:
        new_cam           = cb.reflect_cam_pose(camera[:true_size])
        codes_pred['cam'] = torch.cat([camera[:true_size], new_cam])
        # else:
            # new_cam           = cb.reflect_cam_pose(camera[:true_size, None, :]).squeeze(1)
            # codes_pred['cam'] = torch.cat([camera[:true_size], new_cam])
        return codes_pred


    def forward(self,
                img,
                real_iter    = 0,
                deform       = False,
                inds         = None,
                mask         = None,
                mask_df      = None,
                contour      = None,
                flip_contour = None,
                kp           = None,
    ):
        opts = self.opts
        inputs = {}
        predictions = {}
        device_id = img.get_device()
        inputs['img'] = img
        self.real_iter = real_iter
        inputs['iter'] = real_iter
        inputs['inds'] = inds
        inputs['kp'] = kp
        # mask annotations
        inputs['mask']     = mask
        inputs['mask_df']  = mask_df
        inputs['contour']  = contour
        inputs['flip_contour'] = flip_contour
        if opts.flip_train:
            inputs = self.flip_inputs(inputs)
        self.inputs = inputs
        img = inputs['img']
        predictions = self.predict(img, deform=deform)
        if opts.train_with_masks:
            inputs['xy_map'] = torch.cat( [self.grid[0:1, :, :, None, 0], self.grid[0:1, :, :, None, 1]], dim=-1 ).unsqueeze(1)
            inputs['xy_map'] = inputs['xy_map'].to(device_id)
        return predictions, inputs

    def predict(self, img, deform):
        opts = self.opts
        predictions = {}
        device_id = img.get_device()

        # Unet => I -> embedding -> uv-map
        if opts.train_with_masks:
            unet_output = self.unet_gen(img)
            uv_map      = unet_output[:, 0:self.uv_pred_dim, :, :]
            mask        = torch.sigmoid(unet_output[:, self.uv_pred_dim:, :, :])
            uv_map      = torch.tanh(uv_map) * (1 - 1E-6)
            uv_map      = torch.nn.functional.normalize(uv_map, dim=1, eps=1E-6)
            uv_map_3d   = uv_map.permute(0, 2, 3, 1).contiguous()
            # convert predicted 3D vector to uv-coordinates
            uv_map      = geom_utils.convert_3d_to_uv_coordinates( uv_map.permute(0, 2, 3, 1) ).permute(0, 3, 1, 2)
            uv_map      = uv_map.permute(0, 2, 3, 1).contiguous()

            predictions['seg_mask'] = mask
            predictions['uv_map']   = uv_map

        # RenNet => I -> embedding -> camera and articulation
        img_feat = self.img_encoder(img)
        camera, part_transforms = self.cam_predictor(img_feat)
        if opts.num_hypo_cams == 1:
            camera          = camera.unsqueeze(1)
            part_transforms = part_transforms.unsqueeze(1)
            cam_probs       = camera[:, :, 0:1] * 0 + 1
        else:
            camera, cam_probs = camera[:, :, :7], camera[:, :, 7:]

        membership                     = self.part_membership.forward().unsqueeze(0).repeat( img_feat.size(0), 1, 1 )
        self.membership                = membership.to(device_id)
        predictions['membership']      = self.membership
        predictions['cam']             = camera
        predictions['cam_probs']       = cam_probs
        predictions['part_transforms'] = part_transforms
        predictions                    = self.post_process_predictions(predictions, deform=deform)
        return predictions

    def post_process_predictions(self, predictions, deform):
        opts       = self.opts
        real_iter  = self.real_iter
        b_size     = predictions['cam'].size(0)
        device     = predictions['cam'].get_device()
        geom_preds = []
        verts      = (self.mean_shape['verts'] * 1).to(device=device)

        if opts.flip_train:
            true_size   = b_size // 2
            predictions = self.flip_predictions( predictions, true_size=true_size )

        for cx in range(opts.num_hypo_cams):
            camera = predictions['cam'][:, cx]
            part_transforms = predictions['part_transforms'][:, cx] if deform else None
            renderers = self.renderers[0]
            multi_renderer_mask, multi_renderer_depth = renderers['mask'], renderers['depth']
            geom_pred = self._compute_geometric_predictions(
                verts           = verts,
                uv_map          = predictions['uv_map'] if 'uv_map' in predictions else None,
                camera          = camera,
                part_transforms = part_transforms,
                mask_renderer   = multi_renderer_mask[cx],
                depth_renderer  = multi_renderer_depth[cx]
            )
            geom_preds.append(geom_pred)

        self.membership = predictions['membership']
        for key in geom_preds[0].keys():
            predictions[key] = torch.stack( [geom_preds[cx][key] for cx in range(opts.num_hypo_cams)], dim=1 )

        # camere multiplex warm-up phase -- not used in our experiments
        if opts.train_with_masks and opts.warmup_pose_iter>0 and real_iter<opts.warmup_pose_iter:
            predictions['cam_probs'] = (1.0 / opts.num_hypo_cams) * (torch.zeros(predictions['cam_probs'].shape).float() + 1).to(device)
        return predictions

    def _compute_geometric_predictions(self, verts, uv_map, camera, part_transforms, mask_renderer, depth_renderer):
        predictions = {}
        bsize = camera.size(0)
        img_size = (self.opts.img_size, self.opts.img_size)
        device = camera.get_device()

        if part_transforms is not None:
            parts_rc = self.mean_shape['parts_rc']
            parts_rc = (torch.stack(parts_rc) * 1).to(device=device)
            parts_rc = parts_rc.unsqueeze(0).repeat(bsize, 1, 1)
            verts    = geom_utils.apply_part_transforms(verts, self.mean_shape['parts'], parts_rc, part_transforms, self.membership)
        else:
            verts    = verts[None, ].repeat(bsize, 1, 1)

        # project vertices with the predicted camera
        verts_proj = geom_utils.project_3d_to_image(verts, camera, self.offset_z)[..., 0:2]
        predictions['verts'] = verts
        predictions['verts_proj'] = verts_proj

        # render mask
        faces = (self.mean_shape['faces'] * 1).to(device)
        faces = faces[None, ...].repeat(bsize, 1, 1)
        mask_pred = mask_renderer.forward(verts, faces, camera)
        predictions['mask_render'] = mask_pred


        kp_verts = verts[:, self.kp_vertex_ids, :]
        kp_project = geom_utils.project_3d_to_image(kp_verts, camera, self.offset_z)
        kp_project = kp_project[..., 0:2].view(bsize, len(self.kp_vertex_ids), -1)
        predictions['kp_project'] = kp_project

        if self.opts.train_with_masks:
            points3d = geom_utils.project_uv_to_3d(self.uv2points, verts, uv_map)
            project_points_cam_pred = geom_utils.project_3d_to_image( points3d, camera, self.offset_z )
            project_points          = project_points_cam_pred[..., 0:2].view(bsize, img_size[0], img_size[1], 2)
            project_points_cam_z    = (project_points_cam_pred[..., 2] - self.cam_location[2]).view(bsize, img_size[0], img_size[1])

            predictions['points3d']                = points3d
            predictions['project_points_cam_pred'] = project_points_cam_pred
            predictions['project_points']          = project_points
            predictions['project_points_cam_z']    = project_points_cam_z

            # render depth
            depth_pred = depth_renderer.forward(verts, faces, camera, depth_only=True)
            predictions['depth'] = depth_pred

        return predictions


    ## Loss functions ##

    def kp_loss(self, kp_pred, kp_gt, conf):
        '''
        Keypoint Reprojection Loss
        Args:
            - kp_pred : tensor of shape (bs, num_joints, 2) with the predicted 2D keypoint locations (after reprojection)
            - kp_gt   : tensor of shape (bs, num_joints, 2) with the GT 2D keypoint locations
            - conf    : tensor of shape (bs, num_joints) with 2D keypoint confidence/visibility
        '''
        loss = conf * ((kp_pred - kp_gt)**2).sum(-1)
        loss = loss.mean(-1) / (conf.mean(-1) + 1e-4)
        return loss


    def cycle_consistency_loss(self, project_points, grid_points, mask):
        ''' Cycle Consistency Loss '''
        bsize, img_h, img_w = mask.shape
        non_mask_points     = mask.view(bsize, -1).mean(1)
        mask                = mask.unsqueeze(-1)
        loss                = (mask * project_points - mask * grid_points)
        loss                = loss.pow(2).sum(-1).view(bsize, -1).mean(-1)
        loss                = loss / (non_mask_points + 1E-10)
        return loss

    def depth_loss_fn(self, depth_render, depth_pred, mask):
        ''' Visibility Loss '''
        loss  = torch.nn.functional.relu(depth_pred - depth_render).pow(2) * mask
        shape = loss.shape
        loss  = loss.view(shape[0], -1)
        loss  = loss.mean(-1)
        return loss

    def rotation_reg(self, cameras):
        ''' Rotation Regularization for cameras in camera multiplex '''
        opts = self.opts
        device = cameras.get_device()
        NC2_perm = list(itertools.permutations(range(opts.num_hypo_cams), 2))
        NC2_perm = torch.LongTensor(list(zip(*NC2_perm))).to(device)
        if len(NC2_perm) > 0:
            quats        = cameras[:, :, 3:7]
            quats_x      = torch.gather(quats, dim=1, index=NC2_perm[0].view(1, -1, 1).repeat(len(quats), 1, 4) )
            quats_y      = torch.gather(quats, dim=1, index=NC2_perm[1].view(1, -1, 1).repeat(len(quats), 1, 4) )

            inter_quats  = geom_utils.hamilton_product( quats_x, geom_utils.quat_conj(quats_y) )
            quatAng      = geom_utils.quat2ang(inter_quats).view( len(inter_quats), opts.num_hypo_cams - 1, -1 )
            quatAng      = -1 * torch.nn.functional.max_pool1d(
                                -1 * quatAng.permute(0, 2, 1), opts.num_hypo_cams - 1, stride=1).squeeze()
            rotation_reg = (np.pi - quatAng).mean()
        else:
            rotation_reg = torch.zeros(1).mean().to(device)
        return rotation_reg


    def _compute_geometrics_losses(
                                    self,
                                    kps_gt,
                                    mask,
                                    mask_dt,
                                    contours,
                                    weight_dict,
                                    kps_projected,
                                    # for mask-based losses
                                    reprojected_points,
                                    rendered_depth,
                                    reprojected_points_z,
                                    reprojected_verts,
                                    rendered_mask,
    ):
        bsize  = kps_projected.size(0)
        device = kps_projected.get_device()

        # keypoint reprojection loss
        kp_loss = None
        if kps_gt is not None:
            kp_loss = self.kp_loss(kps_projected, kps_gt[:,:,:2], kps_gt[:,:,2].float())

        cycle_consistency_loss, depth_loss, mask_cov_err, mask_con_err = None, None, None, None
        if self.opts.train_with_masks and mask is not None:
            xy_grid_gt = torch.cat([self.grid[0:1, :, :, None, 0], self.grid[0:1, :, :, None, 1]], dim=-1).to(device)
            # cycle consitency loss
            cycle_consistency_loss = self.cycle_consistency_loss(reprojected_points, xy_grid_gt, mask.squeeze(1))

            # depth loss
            actual_depth_at_pixels = torch.nn.functional.grid_sample(rendered_depth.unsqueeze(1), reprojected_points.detach(), align_corners=True)
            depth_loss = self.depth_loss_fn(actual_depth_at_pixels, reprojected_points_z.unsqueeze(1), mask)

            # mask under-coverage error
            min_verts = []
            for bx in range(bsize):
                with torch.no_grad():
                    mask_cov_err = (reprojected_verts[bx, :, None, :] - contours[bx, :, :])**2
                    mask_cov_err = mask_cov_err.sum(-1)
                    _, min_indices = torch.topk(-1 * mask_cov_err, k=4, dim=0)
                min_verts.append(reprojected_verts[bx][min_indices])
            min_verts = torch.stack(min_verts, dim=0)
            mask_cov_err = (min_verts - contours[:, None, ...])**2
            mask_cov_err = mask_cov_err.sum(-1).view(bsize, -1).mean(1)

            # mask over-coverage error
            mask_con_err = mask_dt * rendered_mask[:, None]
            mask_con_err = mask_con_err.view(bsize, -1).mean(1)


        losses = {}
        if kp_loss is not None:
            losses['kp']  = weight_dict['kp'] * kp_loss
        if cycle_consistency_loss is not None:
            losses['cyc'] = weight_dict['cyc'] * cycle_consistency_loss
        if depth_loss is not None:
            losses['vis'] = weight_dict['vis'] * depth_loss
        if mask_cov_err is not None:
            losses['cov'] = weight_dict['cov'] * mask_cov_err
        if mask_con_err is not None:
            losses['con'] = weight_dict['con'] * mask_con_err
        return losses


    def compute_geometrics_losses(self, predictions, inputs, weight_dict):
        opts   = self.opts
        losses = {}
        for hx in range(opts.num_hypo_cams):
            loss_hx = self._compute_geometrics_losses(
                weight_dict = weight_dict,
                kps_gt   = inputs['kp'],
                mask     = inputs['mask'],
                mask_dt  = inputs['mask_df'],
                contours = inputs['contour'],
                kps_projected        = predictions['kp_project'][:, hx],
                reprojected_points   = predictions['project_points'][:, hx] if 'project_points' in predictions else None,
                rendered_depth       = predictions['depth'][:, hx] if 'depth' in predictions else None,
                reprojected_points_z = predictions['project_points_cam_z'][:, hx] if 'reprojected_points_z' in predictions else None,
                reprojected_verts    = predictions['verts_proj'][:, hx],
                rendered_mask        = predictions['mask_render'][:, hx],
            )
            for key in loss_hx.keys():
                if key in losses:
                    losses[key].append(loss_hx[key])
                else:
                    losses[key] = [loss_hx[key]]
        for key in losses.keys():
            losses[key] = torch.stack(losses[key], dim=1)

        probs = predictions['cam_probs'].squeeze(2)

        for key in losses.keys():
            if key in ['con', 'cov']:
                losses[key] = losses[key].mean()
            else:
                losses[key] = (losses[key] * probs).sum(-1)
                losses[key] = losses[key].mean()


        # regularization for translation in part transformations
        regularize_trans = (predictions['part_transforms'][..., 1:4])**2
        regularize_trans = regularize_trans.sum(-1).sum(-1).mean()
        losses['trans_reg'] = weight_dict['trans_reg'] * regularize_trans

        if opts.train_with_masks:
            # mask loss
            if inputs['mask'] is not None:
                seg_mask_loss = torch.nn.functional.binary_cross_entropy(predictions['seg_mask'], inputs['mask'])
                losses['seg'] = weight_dict['seg'] * seg_mask_loss

            # regularization for cameras in the multiplex
            if opts.num_hypo_cams > 1:
                losses['rot_reg'] = opts.rot_reg_loss_wt * self.rotation_reg(predictions['cam'])
                losses['ent'] = weight_dict['ent'] * (torch.log(probs + 1E-9) * probs).sum(1).mean()

        return losses