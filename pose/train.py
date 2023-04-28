'''
Script used to train 2D pose estimation network.

Exmample usage:
CUDA_VISIBLE_DEVICES=0 python train.py --use_pascal --use_coco --category horse --name horse_150

Running the above will train a 2D pose estimation nework for horses with the default setting.
'''

import os
import logging
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.utils
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tensorboardX import SummaryWriter

from utils.loss import JointsMSELoss
from utils.data_utils import color_heatmap
from model.pose_resnet import pose_resnet
from datasets.dataset_train import ImageDataset
from utils.metrics import AverageMeter, compute_pck_heatmap


parser = argparse.ArgumentParser()
parser.add_argument('--seed',        default=14,            type=int,    help='random seed')
parser.add_argument('--name',        default='',            type=str,    help='name of the experiment', required=True)
parser.add_argument('--category',    default='',            type=str,    help='animal category', required=True)
parser.add_argument('--model',       default='resnet18',    type=str,    help='name of model')
parser.add_argument('--use_pascal',  default=False, action='store_true', help='use images from Pascal for training')
parser.add_argument('--use_coco',    default=False, action='store_true', help='use images from Coco for training')
parser.add_argument('--iter',        default=50000,         type=int,    help='number of total iterations to run')
parser.add_argument('--val_freq',    default=500,           type=int,    help='validation every val_freq iterations')
parser.add_argument('--batch_size',  default=32,            type=int,    help='mini-batch size')
parser.add_argument('--num_workers', default=2,             type=int,    help='number of workers')
parser.add_argument('--lr',          default=1e-4,          type=float,  help='learning rate')
parser.add_argument('--thres',       default='0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1', type=str,  help='thresholds for computing pck')
parser.add_argument('--thres_use',   default=0.04,          type=float,  help='save model with the best pck at this threshold')
parser.add_argument('--input_res',   default=256,           type=int,    help='input image resolution in pixels')
parser.add_argument('--output_res',  default=64,            type=int,    help='output resolution of heatmamp')
parser.add_argument('--save_dir',    default='results',     type=str,    help='path to save results and weights')
args = parser.parse_args()

args.njoints = 16
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.thres = list( map(float, args.thres.split(',')) )
args.thres_idx = args.thres.index(args.thres_use)
args.checkpoint_dir = f'{args.save_dir}/checkpoints'
args.tb_dir = f'{args.save_dir}/tensorboard'
torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = True

def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def train():
    args.checkpoint_dir = f'{args.checkpoint_dir}/{args.name}'
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    args.tb_dir = f'{args.tb_dir}/{args.name}'
    writer_val = SummaryWriter(log_dir=os.path.join(args.tb_dir, 'val'))
    writer_train = SummaryWriter(log_dir=os.path.join(args.tb_dir, 'train'))
    logging.basicConfig(filename=f'{args.tb_dir}/log', level=logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger().addHandler(console)
    console.setLevel(logging.INFO)
    logging.info(args)

    # create dataset
    dataset_train = ImageDataset(args, mode='train')
    dataset_val   = ImageDataset(args, mode='val')
    dl_train = DataLoader(
                            dataset     = dataset_train,
                            batch_size  = args.batch_size,
                            num_workers = args.num_workers,
                            sampler     = data_sampler(dataset_train, shuffle=True),
    )
    dl_val = DataLoader(
                            dataset     = dataset_val,
                            batch_size  = args.batch_size,
                            num_workers = args.num_workers,
                            sampler     = data_sampler(dataset_val, shuffle=False),
    )
    logging.info(f'=> Training: {len(dataset_train)} samples -- Validation: {len(dataset_val)} samples')
    logging.info(f'=> Training for {args.iter} iterations with batch size {args.batch_size}')

    # create model
    model = pose_resnet(args.model, args.njoints)
    model = model.to(args.device)

    # create loss function and optimizer
    criterion = JointsMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    ### Main Loop ###
    best_pck = 0
    loader = sample_data(dl_train)
    pbar = tqdm(range(args.iter), dynamic_ncols=True, smoothing=0.01)

    for idx in pbar:
        model.train()

        if idx > args.iter:
            print("TRAINING IS DONE!")
            break

        batch = next(loader)
        images = batch['image'].to(args.device)
        heatmaps_gt = batch['heatmap_gt'].to(args.device)
        target_weights = batch['target_weights'].to(args.device)

        heatmaps_pred = model(images)
        loss = criterion(heatmaps_pred, heatmaps_gt, target_weights)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 10 == 0:
            pcks = compute_pck_heatmap(heatmap_output = heatmaps_pred, heatmap_gt = heatmaps_gt, target_weights = target_weights, threshold_list = args.thres)
            writer_train.add_scalar('loss', loss.item(), idx)
            for tt in range(len(args.thres)):
                writer_train.add_scalar('pck_{}'.format(args.thres[tt]), pcks[tt], idx)

        if (idx % args.val_freq == 0) and idx>0:
            val_loss, val_pck_list = validate(dl_val, model, criterion, idx, writer_val)

            is_best  = val_pck_list[args.thres_idx] > best_pck
            best_pck = max(best_pck, val_pck_list[args.thres_idx])
            if is_best:
                logging.info('=> Saving best model (pck@{}={:.1f}, iteration {})'.format(args.thres[args.thres_idx], best_pck, idx))
                torch.save(model.state_dict(), f'{args.checkpoint_dir}/best_model.pth')

            writer_val.add_scalar('loss', val_loss, idx)
            for tt in range(len(args.thres)):
                writer_val.add_scalar('pck_{}'.format(args.thres[tt]), val_pck_list[tt], idx)


def validate(loader, model, criterion, iteration, writer_val):
    num_thres  = len(args.thres)
    losses     = AverageMeter()
    pck_list   = [AverageMeter() for _ in range(num_thres)]
    model.eval()

    for idx, batch in enumerate(loader):
        images = batch['image'].to(args.device)
        heatmaps_gt = batch['heatmap_gt'].to(args.device)
        target_weights = batch['target_weights'].to(args.device)
        njoints_vis = torch.sum(target_weights)

        # add visuals to TB
        if idx==0:
            log_imgs = []
            for sample in range(1,3):
                log_img = images[sample]
                log_img = log_img * loader.dataset.stats['std'].view(3, 1, 1).expand_as(log_img).to(args.device) + loader.dataset.stats['mean'].view(3, 1, 1).expand_as(log_img).to(args.device)
                log_imgs.append( T.Resize(128)(log_img) )

        with torch.no_grad():
            heatmaps_pred  = model(images)
        loss = criterion(heatmaps_pred, heatmaps_gt, target_weights)
        pcks = compute_pck_heatmap(heatmap_output = heatmaps_pred, heatmap_gt = heatmaps_gt, target_weights = target_weights, threshold_list = args.thres)

        losses.update(loss.item(), njoints_vis)
        for ii in range(num_thres):
            pck_list[ii].update(pcks[ii], njoints_vis)

            # add visuals to TB
            if idx==0:
                for sample in range(1, 3):
                    log_img = log_imgs[sample-1].cpu()
                    heatmap_gt = heatmaps_gt[sample].cpu().numpy()
                    heatmap_pred = heatmaps_pred[sample].cpu().numpy()
                    gt_list, pred_list = [], []
                    for nj in range(args.njoints):
                        # ground truth
                        gt = Image.fromarray(heatmap_gt[nj])
                        gt = np.array(gt.resize(size=(128,128))).astype(float)
                        color_hm = color_heatmap(gt)
                        out_gt = color_hm * 0.65
                        out_gt = T.ToTensor()( np.uint8(out_gt) )
                        out_gt += 0.35*log_img
                        gt_list.append(out_gt)
                        # prediction
                        pred = Image.fromarray(heatmap_pred[nj])
                        pred = np.array(pred.resize(size=(128,128))).astype(float)
                        color_hm = color_heatmap(pred)
                        out_pred = color_hm * 0.65
                        out_pred = T.ToTensor()( np.uint8(out_pred) )
                        out_pred += 0.35*log_img
                        pred_list.append(out_pred)
                    img_grid = torchvision.utils.make_grid(gt_list+pred_list, nrow=args.njoints)
                    writer_val.add_image(f'Im{sample}___GT_Top___Pred_Bottom', img_grid, iteration)

    return losses.avg, [pck.avg for pck in pck_list]


if __name__ == '__main__':
    train()