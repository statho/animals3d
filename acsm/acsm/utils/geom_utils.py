"""
Code borrowed from https://github.com/akanazawa/cmr/blob/master/nnutils/geom_utils.py
"""
from __future__ import absolute_import, division, print_function
import torch
import math
import numpy as np


def quat2azele(q, use_degrees=True):
    '''
    Arg:
        q: 4
    Returns:
        azimouth : 0 <= az < 2pi
        elevation: 0 <= el <= pi
    '''
    device   = q.get_device()

    # 1-reference in x-axis
    ref1     = torch.Tensor([1,0,0]).to(device)
    new_ref1 = quat_rotate(ref1[None,None,:], q[None,:]).squeeze()
    x1,y1,z1 = new_ref1
    az       = (y1/(x1+1e-6)).atan()
    if x1<0:
        # 2o,3o tetartimorio
        az += np.pi
    elif y1<0:
        # 4o tetartimorio
        az += 2*np.pi
    az = math.degrees(az) % 360

    # 2-reference in y-axis
    ref2     = torch.Tensor([0,1,0]).to(device)
    new_ref2 = quat_rotate(ref2[None,None,:], q[None,:]).squeeze()
    x2,y2,z2 = new_ref2
    ele      = (z2/(y2+1e-6)).atan()
    if y2<0 and z2>=0:
        # 2o
        ele += np.pi
    elif y2<0 and z2<0:
        # 3o
        ele = np.pi - ele
    elif y2>=0 and z2<0:
        # 4o
        ele *= -1
    ele = math.degrees(ele)

    return az, ele


def quat_conj(q):
    return torch.cat([q[:, :, [0]], -1 * q[:, :, 1:4]], dim=-1)

def hamilton_product(qa, qb):
    '''
    Multiply qa by qb.
    Args:
        qa: B X N X 4 quaternions
        qb: B X N X 4 quaternions
    Returns:
        q_mult: B X N X 4
    '''
    qa_0 = qa[:, :, 0]
    qa_1 = qa[:, :, 1]
    qa_2 = qa[:, :, 2]
    qa_3 = qa[:, :, 3]

    qb_0 = qb[:, :, 0]
    qb_1 = qb[:, :, 1]
    qb_2 = qb[:, :, 2]
    qb_3 = qb[:, :, 3]

    # See https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
    q_mult_0 = qa_0 * qb_0 - qa_1 * qb_1 - qa_2 * qb_2 - qa_3 * qb_3
    q_mult_1 = qa_0 * qb_1 + qa_1 * qb_0 + qa_2 * qb_3 - qa_3 * qb_2
    q_mult_2 = qa_0 * qb_2 - qa_1 * qb_3 + qa_2 * qb_0 + qa_3 * qb_1
    q_mult_3 = qa_0 * qb_3 + qa_1 * qb_2 - qa_2 * qb_1 + qa_3 * qb_0

    return torch.stack([q_mult_0, q_mult_1, q_mult_2, q_mult_3], dim=-1)

def quat_rotate(X, q):
    '''
    Rotate points by quaternions.
    Args:
        X: B X N X 3 points
        q: B X 4 quaternions
    Returns:
        X_rot: B X N X 3 (rotated points)
    '''
    # repeat q along 2nd dim
    ones_x = X[[0], :, :][:, :, [0]] * 0 + 1
    q      = torch.unsqueeze(q, 1) * ones_x
    q_conj = torch.cat([q[:, :, [0]], -1 * q[:, :, 1:4]], dim=-1)
    X      = torch.cat([X[:, :, [0]] * 0, X], dim=-1)               # convert point (x,y,z) -> (0,x,y,z) before rotation
    X_rot  = hamilton_product(q, hamilton_product(X, q_conj))       # result is (0, x', y', z'), so discard first dimension
    return X_rot[:, :, 1:4]


def axang2quat(angle, axis):
    '''
    https://en.wikipedia.org/wiki/Euler%E2%80%93Rodrigues_formula
    angle: ... x 1
    axis : ... x 3
    returns
        quat ... x 4
    '''
    cangle = torch.cos(angle/2)
    sangle = torch.sin(angle/2)
    qw = cangle
    qx = axis[...,None,0]*sangle
    qy = axis[...,None,1]*sangle
    qz = axis[...,None,2]*sangle
    quat = torch.cat([qw, qx, qy, qz], dim=-1)
    return quat

def quat2ang(q):
    ang  = 2*torch.acos(torch.clamp(q[:,:,0], min=-1 + 1E-6, max=1-1E-6))
    ang = ang.unsqueeze(-1)
    return ang

# ---------------

def convert_uv_to_3d_coordinates(uv):
    '''
    Takes a uv coordinate between [0,1] and returns a 3d point on the sphere.
    uv -- > [......, 2] shape
    '''
    phi   = 2*np.pi * (uv[..., 0] - 0.5)
    theta =   np.pi * (uv[..., 1] - 0.5)

    if type(uv) == torch.Tensor:
        x = torch.cos(theta)*torch.cos(phi)
        y = torch.cos(theta)*torch.sin(phi)
        z = torch.sin(theta)
        points3d = torch.stack([x,y,z], dim=-1)
    else:
        x = np.cos(theta)*np.cos(phi)
        y = np.cos(theta)*np.sin(phi)
        z = np.sin(theta)
        points3d = np.stack([x,y,z], axis=-1)
    return points3d

def convert_3d_to_uv_coordinates(points):
    '''
    Takes a 3D point and returns an uv between [0,1]
    '''
    eps = 1E-6
    if type(points) == torch.Tensor:
        rad   = torch.clamp(torch.norm(points, p=2, dim=-1), min=eps)
        phi   = torch.atan2(points[...,1], points[...,0])
        theta = torch.asin(torch.clamp(points[...,2]/rad, min=-1+eps, max=1-eps))
        u     = 0.5 + phi/(2*np.pi)
        v     = 0.5 + theta/np.pi
        return torch.stack([u,v],dim=-1)
    else:
        rad   = np.linalg.norm(points, axis=-1)
        phi   = np.arctan2(points[:,1], points[:,0])
        theta = np.arcsin(points[:,2]/rad)
        u     = 0.5 + phi/(2*np.pi)
        v     = 0.5 + theta/np.pi
        return np.stack([u,v],axis=-1)

def compute_distance_in_uv_space(uv1, uv2):
    '''
    uv1 --> N x 2
    uv2 --> M x 2
    distance = N x M
    '''
    uv1_3d = convert_uv_to_3d_coordinates(uv1)
    uv2_3d = convert_uv_to_3d_coordinates(uv2)
    pwd = ((uv1_3d[:,None,:] - uv2_3d[None, :,:])**2).sum(-1)
    return pwd

def sample_textures(texture_flow, images):
    '''
    Inputs:
    - texture_flow : B x F x T x T x 2 (In normalized coordinate [-1, 1])
    - images       : B x 3 x N x N
    Return
    - output       : B x F x T x T x 3
    '''
    # Reshape into B x F x T*T x 2
    T = texture_flow.size(-2)
    F = texture_flow.size(1)
    flow_grid = texture_flow.view(-1, F, T * T, 2)
    # B x 3 x F x T*T
    samples = torch.nn.functional.grid_sample(images, flow_grid, align_corners=True)
    # B x 3 x F x T x T
    samples = samples.view(-1, 3, F, T, T)
    # B x F x T x T x 3
    return samples.permute(0, 2, 3, 4, 1)

def project_uv_to_3d(uv2points, verts, uv_map):
    '''
    Input
    - verts  : B x nhypo x 642 x 3
    - uv_map : B x H x W x 2
    Output
    - points3d : B x nhypo x H x W x 3
    '''
    B, H, W        = uv_map.shape[0], uv_map.shape[1], uv_map.shape[2]
    uv_map_flatten = uv_map.view(B, -1, 2)
    points3d       = uv2points.forward(verts, uv_map_flatten)
    points3d       = points3d.view(B, H*W, 3)
    return points3d

def project_3d_to_image(points3d, cam, offset_z):
    projected_points = orthographic_proj_withz(points3d, cam, offset_z)
    return projected_points

def orthographic_proj(X, cam):
    '''
    - X   : B x N x 3
    - cam : B x 7: [sc, tx, ty, quaternions]
    '''
    quat  = cam[:, -4:]
    X_rot = quat_rotate(X, quat)
    scale = cam[:, 0].contiguous().view(-1, 1, 1)
    trans = cam[:, 1:3].contiguous().view(cam.size(0), 1, -1)
    return scale * X_rot[:, :, :2] + trans

def orthographic_proj_withz(X, cam, offset_z=0.):
    '''
    Orth preserving the z.
    - X: B x N x 3
    - cam: B x 7: [sc, tx, ty, quaternions]
    '''
    quat    = cam[:, -4:]
    X_rot   = quat_rotate(X, quat)
    scale   = cam[:, 0].contiguous().view(-1, 1, 1)
    trans   = cam[:, 1:3].contiguous().view(cam.size(0), 1, -1)
    proj    = scale * X_rot
    proj_xy = proj[:, :, :2] + trans
    proj_z  = proj[:, :, 2, None] + offset_z
    return torch.cat((proj_xy, proj_z), 2)

def orthographic_proj_usingmatrix(points3d, scale, R, T, batch=True):
    '''
    - point3d : B x N x 3
    - camera = [phi_11, phi_12, phi_13, phi_21, phi_23, phi_33, t_x, t_y]  B x 8
    '''
    if batch:
        projection = scale[:,None, None]*torch.bmm(points3d, R.permute(0, 2,1))[:, :, 0:2] +  T[:,None,:]
    else:
        projection = scale*torch.mm(points3d, R.permute(1,0))[:,0:2] + T[None,:]
    return projection

def affine_projection_withoutz(points3d, camera):
    '''
    - point3d : B x N x 3
    - camera = [phi_11, phi_12, phi_13, t_x, phi_21, phi_23, phi_33, t_y] B x 8
    '''
    camera     = camera.view(camera.size(0), 2, 4)
    points3d   = torch.cat([points3d, points3d[:, :, None, 1] * 0 + 1], dim=2)
    projection = torch.bmm(points3d, camera.permute(0, 2, 1))
    return projection

def apply_part_transforms(verts, parts, parts_rc, transforms, membership):
    '''
    Inputs:
    - verts      : 642 x 3,
    - parts      : list of Nodes
    - parts_rc   : B x npart x 3
    - transforms : B x  nparts x 7
    - membership : B x nV
    Return
    - new_verts : B x 642 x 3
    Currently we are only going to allow for rotation and translation.
    '''
    B, nparts = transforms.shape[0], transforms.shape[1]
    b_verts   = verts.unsqueeze(0).repeat(B,1,1) * 1
    new_verts = [verts.clone() for _ in range(nparts)]
    main_part = [part for part in parts if part['is_main']==True][0]
    recursively_apply_transfroms(part_root=main_part, parts=parts, parts_rc=parts_rc, transforms=transforms,
                                                                                new_verts=new_verts, parent_tfs=[])
    new_verts = torch.stack(new_verts,2)
    ## B x nV x nP x 3
    new_verts = new_verts * membership[...,None]
    new_verts = new_verts.sum(2)
    return new_verts

def recursively_apply_transfroms(part_root, parts, parts_rc, transforms, new_verts, parent_tfs):
    ind      = part_root['id']
    children = part_root['children']
    rc       = parts_rc[:,ind]
    parent_tfs_new = [k for k in parent_tfs]
    parent_tfs_new.append((transforms[:,ind], rc))
    for childId in children:
        recursively_apply_transfroms(parts[childId], parts, parts_rc, transforms, new_verts, parent_tfs_new)

    new_verts[ind] = transform_vertices(new_verts[ind], rc, quat=transforms[:,ind, 4:], trans=transforms[:,ind, 1:4])
    for (tfs,rc) in parent_tfs:
        new_verts[ind] = transform_vertices(new_verts[ind], rc, quat=tfs[:,4:], trans=tfs[:,1:4])
    return

def transform_vertices(vert, rc, quat, trans):
    '''
    vert  : B x N x 3
    rc    : B x 3
    quat  : B  x 4
    trans : B x 3
    '''
    new_vert = vert - rc.unsqueeze(1)
    new_vert = quat_rotate(new_vert, quat)
    new_vert = new_vert + rc.unsqueeze(1) + trans.unsqueeze(1)
    return new_vert

def flip_image(image):
    '''
        B x 3 x H x W
    '''
    flip_img = torch.flip(image, dims=[len(image.shape)-1])
    return flip_img

def cross_product(qa, qb):
    '''
    Cross product of va by vb.

    Args:
        qa: B X N X 3 vectors
        qb: B X N X 3 vectors
    Returns:
        q_mult: B X N X 3 vectors
    '''
    qa_0 = qa[:, :, 0]
    qa_1 = qa[:, :, 1]
    qa_2 = qa[:, :, 2]

    qb_0 = qb[:, :, 0]
    qb_1 = qb[:, :, 1]
    qb_2 = qb[:, :, 2]

    # See https://en.wikipedia.org/wiki/Cross_product
    q_mult_0 = qa_1 * qb_2 - qa_2 * qb_1
    q_mult_1 = qa_2 * qb_0 - qa_0 * qb_2
    q_mult_2 = qa_0 * qb_1 - qa_1 * qb_0

    return torch.stack([q_mult_0, q_mult_1, q_mult_2], dim=-1)