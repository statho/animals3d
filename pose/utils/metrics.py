import math
import torch
import numpy as np
import utils.image_utils as image_utils

class AverageMeter:
    '''
    Computes and stores the average and current value
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def is_within_range(pred_idx, gt_idx, threshold, heatmap_width):
    x_pred, y_pred = pred_idx.item()  % heatmap_width, pred_idx.item() // heatmap_width
    x_true, y_true = gt_idx.item()    % heatmap_width, gt_idx.item()   // heatmap_width
    return 1 if math.sqrt( (x_pred-x_true)**2 + (y_pred-y_true)**2 ) <= threshold else 0

def compute_pck_heatmap(heatmap_output, heatmap_gt, target_weights, threshold_list):
    '''
    Compute PCK @ k in the heatmap space
    Args:
    - heatmap_output : torch.Tensor of size (batch_size, num_joints, heatmap_height, heatmap_width)
    - heatmap_gt     : torch.Tensor of size [batch_size, num_joints, heatmap_height, heatmap_width)
    - target_weights : torch.Tensor of size [batch_size, num_joints, 1]
    - threshold_list : list with threshold values k to use for calculating PCK @ k
    Return:
    - pck : list with PCK @ k for different values of k
    '''
    batch_size, num_joints, heatmap_height, heatmap_width = heatmap_gt.size()
    heatmap_output = heatmap_output.reshape((batch_size, num_joints, -1)).split(1, 1)
    heatmap_gt = heatmap_gt.reshape((batch_size, num_joints, -1)).split(1, 1)
    total_visible_joints = torch.sum(target_weights).item()
    pck_list = [0]*len(threshold_list)
    for j in range(num_joints):
        pred = heatmap_output[j].squeeze(1)
        gt = heatmap_gt[j].squeeze(1)
        _, idxs_pred = torch.max(pred, 1)
        _, idxs_gt   = torch.max(gt,   1)
        targets = target_weights[:, j].squeeze(1)
        for b in range(batch_size):
            if targets[b]:
                for i in range(len(threshold_list)):
                    pck_list[i] += is_within_range(idxs_pred[b], idxs_gt[b], heatmap_height*threshold_list[i], heatmap_width)
    for k in range(len(pck_list)):
        pck_list[k] = 100 * pck_list[k] / total_visible_joints
    return pck_list


def compute_pck(keypoints_gt, heatmap_output, target_weights, inv_matrix, center, scale, resolution, threshold_list):
    '''
    Compute PCK @ k in the uncropped input image
    Args:
    - keypoints_gt   : torch.Tensor of size (batch_size, num_joints, 3)
    - heatmap_output : torch.Tensor of size (batch_size, num_joints, heatmap_height, heatmpa_width)
    - target_weights : torch.Tensor of size (batch_size, num_joints)
    - inv_matrix     : ndarray with mapping from heatmap to input image space
    - c_arr          : torch.Tensor center coords for images in the batch
    - s_arr          : torch.Tensor with scales for images in the batch
    - resolution     : input image resolution
    - threshold_list : list with threshold values k to use for calculating PCK @ k
    Return:
    - pck : list with PCK @ k for different values of k
    '''
    keypoints_gt = keypoints_gt.numpy()
    c_arr = center.numpy()
    s_arr = scale.numpy()
    total_visible_joints = torch.sum(target_weights).item()

    batch_size, num_joints, heatmap_height, heatmap_width = heatmap_output.size()
    heatmap_output = heatmap_output.view(batch_size*num_joints, heatmap_height*heatmap_width)

    _, idxs = torch.max(heatmap_output, 1)
    idxs = idxs.view(batch_size, num_joints).numpy()

    pck_list = [0]*len(threshold_list)
    for b in range(batch_size):
        c = c_arr[b]
        s = s_arr[b]
        unormalized_scale = s * 200
        target_heatmap_width = target_weights[b,:]
        for j in range(num_joints):
            if target_heatmap_width[j]:
                keypoint_pred = np.array([idxs[b,j] % heatmap_width, idxs[b,j] // heatmap_width, 1])
                # transform from heatmap space to input image space
                keypoint_pred[:2] = np.dot(inv_matrix, keypoint_pred)[:2]
                # trasform from input image space back to uncropped image space
                keypoint_pred = image_utils.transform(keypoint_pred[:2], c, s, resolution, invert=1)
                # trace()
                dist = math.sqrt( (keypoint_pred[0]-keypoints_gt[b,j,0])**2 + (keypoint_pred[1]-keypoints_gt[b,j,1])**2 )
                for k in range(len(threshold_list)):
                    if dist <= threshold_list[k]*unormalized_scale:
                        pck_list[k] += 1
    for k in range(len(pck_list)):
        pck_list[k] = 100 * pck_list[k] / total_visible_joints
    return pck_list