'''
Code borrowed and adaped from
https://github.com/bearpaw/pytorch-pose/blob/master/pose/losses/jointsmseloss.py
'''

import torch
import torch.nn as nn

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight=True):
        super(JointsMSELoss, self).__init__()
        self.criterion         = nn.MSELoss(reduction='sum')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        loss = 0
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        njoints_vis = torch.sum(target_weight)
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt   = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
        return loss / njoints_vis