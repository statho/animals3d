'''
Script for generating keypoint PL.

Example usage:
CUDA_VISIBLE_DEVICES=0 python generate_pl.py --category horse

Running the above will create keypoint PL for horses and will save the PLs in a json file.
'''
import os
import json
import copy
import torch
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
import utils.image_utils as image_utils
from datasets.dataset_pl import UnlabeledImages
from model.pose_resnet import pose_resnet


parser = argparse.ArgumentParser()
parser.add_argument('--name',       default='',         type=str, help='name of the experiment')
parser.add_argument('--category',   default='',         type=str, help='animal category')
parser.add_argument('--model',      default='resnet18', type=str, help='name of model')
parser.add_argument('--batch_size', default=256,        type=int, help='mini-batch size')
parser.add_argument('--num_workers',default=4,          type=int, help='number of workers')
parser.add_argument('--input_res',  default=256,        type=int, help='input image resolution in pixels')
parser.add_argument('--output_res', default=64,         type=int, help='output resolution of heatmamp')
parser.add_argument('--save_dir',   default='results',  type=str, help='path to saved results and weights')
parser.add_argument('--is_aux',     default=False, action='store_true', help='use this when generating PL with the auxiliary network')
args = parser.parse_args()

args.njoints = 16
args.device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.checkpoint_dir = f'{args.save_dir}/checkpoints'


def generate_pl():
    if len(args.name) > 0:
        args.checkpoint_dir = f'{args.checkpoint_dir}/{args.name}'

    # create dataset
    dataset = UnlabeledImages(args)
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


    idx_arr, max_arr = None, None
    img_id_list, center_list, scale_list = [], [], []
    for batch in tqdm(data_loader):
        img_ids = batch['img_id']
        batch_centers = batch['center'].numpy().tolist()
        batch_scales = batch['scale'].numpy().tolist()

        # get predictions
        with torch.no_grad():
            heatmap_pred = model(batch['image'].to(args.device))

        # find keypoint location in heatmap
        batch_size, num_joints, heatmap_height, heatmap_width = heatmap_pred.size()
        heatmap_pred  = heatmap_pred.view(batch_size*num_joints, heatmap_height*heatmap_width)
        maxvals, idxs = torch.max(heatmap_pred, 1)
        maxvals       = maxvals.view(batch_size, num_joints)
        idxs          = idxs.view(batch_size, num_joints)
        maxvals_arr   = maxvals.cpu().numpy()
        idxs_arr      = idxs.cpu().numpy()

        # keep important things for computing the keypoint pseudo-labels in the uncropped images
        if idx_arr is None:
            img_id_list = img_ids
            center_list = batch_centers
            scale_list  = batch_scales
            idx_arr     = idxs_arr
            max_arr     = maxvals_arr
        else:
            img_id_list.extend(img_ids)
            center_list.extend(batch_centers)
            scale_list.extend(batch_scales)
            idx_arr = np.concatenate((idx_arr, idxs_arr), axis=0)
            max_arr = np.concatenate((max_arr, maxvals_arr), axis=0)

    # create labels using all the predictions and save them
    save_list = []
    # scale detection confidence to [0, 1]
    max_conf = np.max(max_arr)
    joint_dict = defaultdict(list)
    print('=> Saving keypoint pseudo-labels')
    for ii, img_id in enumerate(img_id_list):
        c = center_list[ii]
        s = scale_list[ii]
        for nj in range(args.njoints):
            keypoint_pred = np.array([idx_arr[ii, nj] % heatmap_width, idx_arr[ii, nj] // heatmap_width, 1])
            # transform from heatmap space to input image space
            keypoint_pred[:2] = np.dot(inv_matrix, keypoint_pred)[:2]
            # trasform from input image space (cropped image) back to uncropped image space
            keypoint_pred = image_utils.transform(keypoint_pred[:2], c, s, resolution, invert=1)
            joint_dict[img_id].append([ keypoint_pred[0].astype(float), keypoint_pred[1].astype(float),  (max_arr[ii, nj] / max_conf).astype(float) ])

    # add keypoint pseudo-labels to the annotations
    bbox_file = f'../data/yfcc100m/labels_0/{args.category}_bbox.json'
    with open(bbox_file) as f:
        annos = json.load(f)
    for anno in annos:
        anno['joints'] = copy.deepcopy(joint_dict[anno['img_id']])

    # save filelist
    filelist = [anno['img_id'] for anno in annos]
    filelist_path = f'../data/yfcc100m/filelists'
    filelist_name = f'{filelist_path}/{args.category}.txt'
    if not os.path.exists(filelist_path):
        os.makedirs(filelist_path)
    with open(filelist_name, 'w') as f:
        f.write('\n'.join(filelist))

    # save annotations
    save_path = f'../data/yfcc100m/labels'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fname = args.name if len(args.name) > 0 else args.category
    if args.is_aux:
        save_name = f'{save_path}/{fname}_pl_2d_aux.json'
    else:
        save_name = f'{save_path}/{fname}_pl_2d.json'
    with open(save_name, 'w') as f:
    	json.dump(annos, f)


if __name__ == '__main__':
	generate_pl()