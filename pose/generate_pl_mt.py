'''
Script for generating keypoint PL with multi-transform inference.

Example usage:
CUDA_VISIBLE_DEVICES=0 python generate_pl_mt.py --category horse

Running the above will create keypoint PL for horses and will save the PLs in a json file.
'''
import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

import utils.image_utils as image_utils
from model.pose_resnet import pose_resnet
from datasets.dataset_pl import UnlabeledImagesMT


parser = argparse.ArgumentParser()
parser.add_argument('--name',       default='',         type=str, help='name of the experiment')
parser.add_argument('--category',   default='',         type=str, help='animal category')
parser.add_argument('--model',      default='resnet18', type=str, help='name of model')
parser.add_argument('--batch_size', default=256,        type=int, help='mini-batch size')
parser.add_argument('--num_workers',default=4,          type=int, help='number of workers')
parser.add_argument('--input_res',  default=256,        type=int, help='input image resolution in pixels')
parser.add_argument('--output_res', default=64,         type=int, help='output resolution of heatmamp')
parser.add_argument('--save_dir',   default='results',  type=str, help='path to saved results and weights')
args = parser.parse_args()

args.njoints = 16
args.device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.checkpoint_dir = f'{args.save_dir}/checkpoints'



def generate_pl_mt():
    if len(args.name) > 0:
        args.checkpoint_dir = f'{args.checkpoint_dir}/{args.name}'

    # create dataset
    dataset = UnlabeledImagesMT(args)
    data_loader = DataLoader(
                                dataset     = dataset,
                                batch_size  = args.batch_size,
                                num_workers = args.num_workers,
                                shuffle     = False,
                                pin_memory  = True,
    )

    # create model and load checkpoint
    model = pose_resnet(args.model, args.njoints)
    model = model.to(args.device)
    model_path = f'{args.checkpoint_dir}/best_model.pth' if len(args.name) > 0 else f'{args.checkpoint_dir}/{args.category}.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # tranformations for getting the predicted joints back to the initial image
    center = (args.input_res/2, args.input_res/2)
    scale = args.input_res / 200
    resolution = (args.input_res, args.input_res)
    output_matrix = image_utils.get_transform(center, scale, (args.output_res, args.output_res))[:2]
    inv_matrix = image_utils.inv_mat(output_matrix)

    idx_arr, max_arr, inv_aug_arr = None, None, None
    img_id_list, box_center_list, box_scale_list = [], [], []
    for batch in tqdm(data_loader):
        img_ids = batch['img_id']
        batch_centers = batch['center'].numpy().tolist()
        batch_scales = batch['scale'].numpy().tolist()
        batch_size, num_aug, num_channels, _, _ = batch['image'].size()
        inv_aug_matrices = batch['inv_aug_matrices'].numpy()

        # get predictions
        with torch.no_grad():
            heatmap_pred = model(batch['image'].view(batch_size * num_aug, num_channels, args.input_res, args.input_res).contiguous().to(args.device))
        heatmap_pred = heatmap_pred.view(batch_size, num_aug, args.njoints, args.output_res, args.output_res)

        # compute keypoint locations from heatmaps
        heatmap_pred = heatmap_pred.view(batch_size, num_aug, args.njoints, -1)
        maxvals, idxs = torch.max(heatmap_pred, -1)
        maxvals_arr = maxvals.cpu().numpy()
        idxs_arr = idxs.cpu().numpy()

        if idx_arr is None:
            img_id_list = img_ids
            box_center_list = batch_centers
            box_scale_list = batch_scales
            idx_arr = idxs_arr
            max_arr = maxvals_arr
            inv_aug_arr = inv_aug_matrices
        else:
            img_id_list.extend(img_ids)
            box_center_list.extend(batch_centers)
            box_scale_list.extend(batch_scales)
            idx_arr = np.concatenate((idx_arr, idxs_arr), axis=0)
            max_arr = np.concatenate((max_arr, maxvals_arr), axis=0)
            inv_aug_arr = np.concatenate((inv_aug_arr, inv_aug_matrices), axis=0)

    # create labels using all the predictions and save them
    save_list = []
    # scale detection confidence to [0, 1]
    max_conf  = np.max(max_arr)
    print('=> Saving keypoint pseudo-labels')
    for augm in range(num_aug):
        augm_list = []
        for i, img_id in enumerate(img_id_list):
            box_center = box_center_list[i]
            box_scale  = box_scale_list[i]
            for nj in range(args.njoints):
                # transform from heatmap space to input image space
                keypoints_pred = np.array([idx_arr[i, augm, nj] % args.output_res, idx_arr[i, augm, nj] // args.output_res, 1])
                keypoints_pred[:2] = np.dot(inv_matrix, keypoints_pred)[:2]
                # inverse transormations for scaling/rotation
                keypoints_pred[:2] = np.dot(inv_aug_arr[i, augm], keypoints_pred)[:2]
                # trasform from input image space back to uncropped image space
                keypoints_pred = image_utils.transform(keypoints_pred[:2], box_center, box_scale, resolution, invert=1)
                # FORMAT: img_id kp_id kp_x kp_y kp_confidence
                augm_list.append( f'{img_id} {nj} {keypoints_pred[0]} {keypoints_pred[1]} {max_arr[i, augm, nj] / max_conf}\n' )
        save_list.append(augm_list)
    print(f'=> Created keypoint pseudo-labels for { int(len(save_list[1])/args.njoints) } images')

    # save pseudo-labels
    dir_name = args.name if len(args.name) > 0 else args.category
    save_path = f'../data/yfcc100m/labels_0/{dir_name}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for augm in range(num_aug):
        save_name = f'{save_path}/augm_{augm}.txt'
        with open(save_name, 'w') as f:
            f.write(''.join(save_list[augm]))


if __name__ == '__main__':
	generate_pl_mt()