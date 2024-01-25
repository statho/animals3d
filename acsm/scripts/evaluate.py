'''
Evalutation code -- Code adapted from
https://github.com/nileshkulkarni/acsm/blob/master/acsm/benchmark/pascal/kp_project.py

Example usage:
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --scale_mesh --kp_anno --sfm_anno --dataset pascal --category sheep

Running the above will evaluate the provided model for the sheep category on Pascal.
'''

from __future__ import absolute_import, division, print_function
import numpy as np
from absl import app, flags
import torch
import torchvision
from torch.utils.data import DataLoader

import constants
from acsm.utils import geom_utils_ucmr
from acsm.datasets.objects import ImageDatasetEval
from acsm.model import model_utils
from acsm.model.network import ACSM
from acsm.utils.test_utils import Tester


flags.DEFINE_integer('seed',                  0, 'seed for randomness')
flags.DEFINE_string('dataset',         'pascal', 'dataset to visualize')
flags.DEFINE_boolean('scale_mesh',        False, 'scale mesh to unit sphere')
flags.DEFINE_float('initial_quat_bias_deg', 90., 'rotation bias in degrees (90 degrees for head-view)')
opts = flags.FLAGS


class Evaluator(Tester):

    def preload_model_data(self):
        opts = self.opts
        model_dir, self.mean_shape, _ = model_utils.load_template_shapes(opts, device=self.device)
        dpm, parts_data, self.kp_vertex_ids = model_utils.init_dpm(model_dir, self.mean_shape, parts_file=f'acsm/cachedir/part_files/{opts.category}.txt')
        self.part_active_state, self.part_axis_init, self.part_perm = model_utils.load_active_parts(model_dir, self.save_dir, dpm, parts_data, suffix='')

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
        # get input images
        input_imgs = batch['img'].type(self.Tensor)
        for b in range(input_imgs.size(0)):
            input_imgs[b] = self.resnet_transform(input_imgs[b])
        self.imgs = input_imgs.to(self.device)
        # get joints
        if 'kp' in batch:
            self.codes_gt['kp'] = batch['kp'].type(self.Tensor).to(self.device)
        # get masks
        if self.opts.mask_anno and 'mask' in batch:
            mask = batch['mask'].type(self.Tensor)
            mask = (mask > 0.5).float()
            self.mask = mask.to(self.device)
        # get pseudo-gt cameras from sfm
        if self.opts.sfm_anno and 'sfm_pose' in batch:
            self.quat_gt = batch['sfm_pose'][2].type(self.Tensor)
            self.valid_cam = batch['valid_cam'].numpy()

    def predict(self):
        with torch.no_grad():
            predictions = self.model.predict(self.imgs, deform=True) if not self.is_dataparallel_model else self.model.module.predict(self.imgs, deform=True)
        self.codes_pred = predictions
        bsize = len(self.imgs)
        camera, verts, kp_project_selected, mask_selected = [], [], [], []
        for b in range(bsize):
            max_ind = torch.argmax(self.codes_pred['cam_probs'][b], dim=0).item()
            camera.append(self.codes_pred['cam'][b][max_ind])
            verts.append(self.codes_pred['verts'][b][max_ind])
            kp_project_selected.append( self.codes_pred['kp_project'][b][max_ind] )
            mask_selected.append( self.codes_pred['mask_render'][b][max_ind] )
        self.codes_pred['camera_selected'] = torch.stack(camera)
        self.codes_pred['verts_selected'] = torch.stack(verts)
        self.codes_pred['kp_project_selected'] = torch.stack(kp_project_selected)
        self.codes_pred['mask_selected'] = torch.stack(mask_selected)

    def evaluate(self):
        ## Keypoint reprojection error
        padding_frac = self.opts.padding_frac
        # The [-1,1] coordinate frame in which keypoints corresponds to:
        #    (1+2*padding_frac)*max_bbox_dim in image coords
        # pt_norm = 2* (pt_img - trans)/((1+2*pf)*max_bbox_dim)
        # err_pt = 2*err_img/((1+2*pf)*max_bbox_dim)
        # err_pck_norm = err_img/max_bbox_dim = err_pt*(1+2*pf)/2
        # so the keypoint error in the canonical frame should be multiplied by:
        err_scaling = (1 + 2 * padding_frac) / 2.0
        kps_gt = self.codes_gt['kp'].cpu().numpy()
        kps_vis  = kps_gt[:, :, 2]
        kps_gt   = kps_gt[:, :, 0:2]
        kps_pred = self.codes_pred['kp_project_selected'].type_as(self.codes_gt['kp']).cpu().numpy()
        kps_err  = kps_pred - kps_gt
        kps_err  = np.sqrt(np.sum(kps_err * kps_err, axis=2)) * err_scaling

        ## mIoU metric
        iou = None
        if self.opts.mask_anno and self.mask is not None:
            mask = self.mask
            bs = mask.size(0)
            mask_gt = mask.view(bs, -1).cpu().numpy()
            mask_pred = self.codes_pred['mask_selected']
            mask_pred = (mask_pred > 0.5).float().view(bs, -1).cpu().detach().numpy()
            intersection = mask_gt * mask_pred
            union = mask_gt + mask_pred - intersection
            iou = intersection.sum(1) / union.sum(1)

        ## camera rotation error
        quat_error, azel_gt, azel_pred = None, None, None
        if self.opts.sfm_anno and self.quat_gt is not None:
            quat_gt   = self.quat_gt.float().cpu()
            quat_pred = self.codes_pred['camera_selected'][:, 3:7].float().cpu()
            # relative differnce of quaternions because of rotation
            quat_rel         = geom_utils_ucmr.hamilton_product(quat_pred, geom_utils_ucmr.quat_inverse(quat_gt))
            # get axis and rotation angle of quaternion
            axis, quat_error = geom_utils_ucmr.quat2axisangle(quat_rel)
            quat_error       = torch.min(quat_error, 2*np.pi - quat_error)
            quat_error       = quat_error * 180 / np.pi
            quat_error       = self.valid_cam * quat_error.numpy()
            # find viewpoint
            azel_gt   = geom_utils_ucmr.camera_quat_to_position_az_el(quat_gt,   initial_quat_bias_deg=opts.initial_quat_bias_deg).numpy()
            azel_pred = geom_utils_ucmr.camera_quat_to_position_az_el(quat_pred, initial_quat_bias_deg=opts.initial_quat_bias_deg).numpy()

        return kps_err, kps_vis, iou, quat_error, azel_gt, azel_pred


    def compute_metrics(self, stats):
        # AUC
        n_vis_p          = np.sum( stats['kps_vis'] )
        n_correct_p_pt06 = np.sum( (stats['kps_err'] < 0.06) * stats['kps_vis'])
        n_correct_p_pt07 = np.sum( (stats['kps_err'] < 0.07) * stats['kps_vis'])
        n_correct_p_pt08 = np.sum( (stats['kps_err'] < 0.08) * stats['kps_vis'])
        n_correct_p_pt09 = np.sum( (stats['kps_err'] < 0.09) * stats['kps_vis'])
        n_correct_p_pt10 = np.sum( (stats['kps_err'] < 0.10) * stats['kps_vis'])
        pck06 = 100 * (n_correct_p_pt06 / n_vis_p)
        pck07 = 100 * (n_correct_p_pt07 / n_vis_p)
        pck08 = 100 * (n_correct_p_pt08 / n_vis_p)
        pck09 = 100 * (n_correct_p_pt09 / n_vis_p)
        pck10 = 100 * (n_correct_p_pt10 / n_vis_p)
        auc   = (pck06+pck07+pck08+pck09+pck10) / 5
        print('=> AUC: {:.1f}'.format(auc))

        # Camera error
        if self.opts.sfm_anno:
            mean_cam_err = stats['quat_error'].sum() / sum(stats['quat_error']>0)
            print('=> cam_err: {:.1f}'.format(mean_cam_err))

        # mIoU
        if self.opts.mask_anno:
            mean_iou = 100 * stats['ious'].mean()
            print('=> mIOU : {:.1f}'.format(mean_iou))


    def test(self):
        print(f'Evaluating on {self.opts.dataset} ...')
        bench_stats = { 'kps_err': [], 'kps_vis': [], 'ious' : [], 'quat_error': [], 'azel_gt': [], 'azel_pred': [] }
        for batch in self.dataloader:
            self.set_input(batch)
            self.predict()
            kps_err, kps_vis, iou, quat_error, azel_gt, azel_pred = self.evaluate()
            bench_stats['kps_err'].append(kps_err)
            bench_stats['kps_vis'].append(kps_vis)
            if self.opts.mask_anno:
                bench_stats['ious'].append(iou)
            if self.opts.sfm_anno and quat_error is not None:
                bench_stats['quat_error'].append(quat_error)
                bench_stats['azel_gt'].append(azel_gt)
                bench_stats['azel_pred'].append(azel_pred)

        bench_stats['kps_err'] = np.concatenate(bench_stats['kps_err'])
        bench_stats['kps_vis'] = np.concatenate(bench_stats['kps_vis'])
        if self.opts.mask_anno:
            bench_stats['ious'] = np.concatenate(bench_stats['ious'])
        if self.opts.sfm_anno:
            bench_stats['quat_error'] = np.concatenate(bench_stats['quat_error'])
        self.compute_metrics(bench_stats)


def main(_):
    assert (opts.category in ['horse', 'cow', 'sheep'] and opts.dataset in ['pascal', 'animal_pose']) or \
                opts.category in ['giraffe', 'bear'] and opts.dataset=='coco', 'Error in category/dataset arguments'

    seed = opts.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    opts.split = 'val'
    opts.batch_size = 1

    # Don't pad/jitter while evaluating
    opts.padding_frac = 0
    opts.jitter_frac  = 0

    # Run evaluation
    tester = Evaluator(opts)
    tester.init_dataset()
    tester.define_model()
    tester.test()


if __name__ == '__main__':
    app.run(main)