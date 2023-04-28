'''
Script used for evalution.

Example usage:
CUDA_VISIBLE_DEVICES=0 python evaluate.py --dataset pascal --category horse

Running the above will evaluate the provided model for the horse cateogry on Pascal.
'''

import torch
import argparse
from torch.utils.data import DataLoader

import utils.image_utils as image_utils
from datasets.dataset_eval import ImageDatasetEval
from model.pose_resnet import pose_resnet
from utils.metrics import AverageMeter, compute_pck

parser = argparse.ArgumentParser()
parser.add_argument('--name',       default='',         type=str, help='name of the experiment')
parser.add_argument('--category',   default='',         type=str, help='animal category')
parser.add_argument('--dataset',    default='pascal',   type=str, help='dataset')
parser.add_argument('--model',      default='resnet18', type=str, help='name of model')
parser.add_argument('--batch_size', default=256,        type=int, help='mini-batch size')
parser.add_argument('--num_workers',default=4,          type=int, help='number of workers')
parser.add_argument('--thres',      default='0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1', type=str,  help='thresholds for computing pck')
parser.add_argument('--input_res',  default=256,        type=int, help='input image resolution in pixels')
parser.add_argument('--output_res', default=64,         type=int, help='output resolution of heatmamp')
parser.add_argument('--save_dir',   default='results',  type=str, help='path to saved results and weights')
args = parser.parse_args()

args.njoints = 16
args.device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.thres   = list( map(float, args.thres.split(',')) )
args.checkpoint_dir = f'{args.save_dir}/checkpoints'


def eval():
    assert (args.category in ['horse', 'cow', 'sheep'] and args.dataset in ['pascal', 'animal_pose']) or \
                args.category in ['giraffe', 'bear'] and args.dataset=='coco', 'Error in category/dataset arguments'

    if len(args.name) > 0:
        args.checkpoint_dir = f'{args.checkpoint_dir}/{args.name}'

    # create dataset
    dataset = ImageDatasetEval(args)
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
    scale  = args.input_res / 200
    resolution = (args.input_res, args.input_res)
    output_matrix = image_utils.get_transform(center, scale, (args.output_res, args.output_res))[:2]
    inv_matrix = image_utils.inv_mat(output_matrix)

    num_thres = len(args.thres)
    pck_list = [AverageMeter() for _ in range(num_thres)]
    for batch in data_loader:
        with torch.no_grad():
            heatmap_pred = model(batch['image'].to(args.device))
        pcks = compute_pck(
                            keypoints_gt = batch['keypoints_gt'],
                            heatmap_output = heatmap_pred.cpu(),
                            target_weights = batch['target_weights'].squeeze(-1),
                            inv_matrix = inv_matrix,
                            center = batch['center'],
                            scale = batch['scale'],
                            resolution = resolution,
                            threshold_list = args.thres,
        )
        total_visible_joints = torch.sum(batch['target_weights']).item()
        for ii in range(num_thres):
            pck_list[ii].update(pcks[ii], total_visible_joints)
    pck_avgs = [float('{:.1f}'.format(pck.avg)) for pck in pck_list]
    print(f'PCK@{args.thres}: {pck_avgs}')


if __name__ == '__main__':
	eval()