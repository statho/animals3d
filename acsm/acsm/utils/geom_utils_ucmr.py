"""
Code borrowed and adapted from
https://github.com/shubham-goel/ucmr/blob/master/src/nnutils/geom_utils.py
"""
from __future__ import absolute_import, division, print_function

import torch
import numpy as np

def get_viewpoint(azimouth):
    ''' Returns the viewpoint given azimouth angle '''
    return 'side' if (azimouth>=45 and azimouth<135) or (azimouth>=-135 and azimouth<-45) else 'front-back'

def quat_inverse(quat):
    """
    quat: B x 4: [quaternions]
    returns inverted quaternions
    """
    flip = torch.tensor([1,-1,-1,-1],dtype=quat.dtype,device=quat.device)
    quat_inv = quat * flip.view((1,)*(quat.dim()-1)+(4,))
    return quat_inv

def hamilton_product(qa, qb):
    """Multiply qa by qb.
    Args:
        qa: B X N X 4 quaternions
        qb: B X N X 4 quaternions
    Returns:
        q_mult: B X N X 4
    """
    qa_0 = qa[..., 0]
    qa_1 = qa[..., 1]
    qa_2 = qa[..., 2]
    qa_3 = qa[..., 3]

    qb_0 = qb[..., 0]
    qb_1 = qb[..., 1]
    qb_2 = qb[..., 2]
    qb_3 = qb[..., 3]

    # See https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
    q_mult_0 = qa_0*qb_0 - qa_1*qb_1 - qa_2*qb_2 - qa_3*qb_3
    q_mult_1 = qa_0*qb_1 + qa_1*qb_0 + qa_2*qb_3 - qa_3*qb_2
    q_mult_2 = qa_0*qb_2 - qa_1*qb_3 + qa_2*qb_0 + qa_3*qb_1
    q_mult_3 = qa_0*qb_3 + qa_1*qb_2 - qa_2*qb_1 + qa_3*qb_0

    return torch.stack([q_mult_0, q_mult_1, q_mult_2, q_mult_3], dim=-1)

def quat_rotate(X, quat):
    """Rotate points by quaternions.

    Args:
        X: B X N X 3 points
        quat: B X 4 quaternions

    Returns:
        X_rot: B X N X 3 (rotated points)
    """
    # repeat q along 2nd dim
    # import ipdb; ipdb.set_trace()
    # ones_x = X[[0], :, :][:, :, [0]]*0 + 1
    # q = torch.unsqueeze(q, 1)*ones_x
    quat      = quat[:,None,:].expand(-1,X.shape[1],-1)
    quat_conj = torch.cat([ quat[:, :, 0:1] , -1*quat[:, :, 1:4] ], dim=-1)
    X         = torch.cat([ X[:, :, 0:1]*0, X ], dim=-1)
    X_rot     = hamilton_product(quat, hamilton_product(X, quat_conj))
    return X_rot[:, :, 1:4]

# ------------------------------

def axisangle2quat(axis, angle):
    """
    axis : Bx3: [axis]
    angle: B  : [angle]
    returns
        quaternion: Bx4
    """
    axis  = torch.nn.functional.normalize(axis,dim=-1)
    angle = angle.unsqueeze(-1)/2
    quat  = torch.cat([angle.cos(), angle.sin()*axis], dim=-1)
    return quat

def quat2axisangle(quat):
    """
    quat: B x 4: [quaternions]
    returns quaternion axis, angle
    """
    cos   = quat[..., 0]
    sin   = quat[..., 1:].norm(dim=-1)
    axis  = quat[..., 1:]/sin[..., None]
    angle = 2*cos.clamp(-1+1e-6,1-1e-6).acos()
    return axis, angle

def get_base_quaternions(num_pose_az=8, num_pose_el=5, initial_quat_bias_deg=90., elevation_bias=0, azimuth_bias=0):
    _axis = torch.eye(3).float()

    # Quaternion base bias
    xxx_base = [1.,0.,0.]
    aaa_base = initial_quat_bias_deg
    axis_base = torch.tensor(xxx_base).float()
    angle_base = torch.tensor(aaa_base).float() / 180. * np.pi
    qq_base = axisangle2quat(axis_base, angle_base) # 4

    # Quaternion multipose bias
    azz = torch.as_tensor(np.linspace(0,2*np.pi,num=num_pose_az,endpoint=False)).float() + azimuth_bias * np.pi/180
    ell = torch.as_tensor(np.linspace(-np.pi/2,np.pi/2,num=(num_pose_el+1),endpoint=False)[1:]).float() + elevation_bias * np.pi/180
    quat_azz = axisangle2quat(_axis[1], azz) # num_pose_az,4
    quat_ell = axisangle2quat(_axis[0], ell) # num_pose_el,4
    quat_el_az = hamilton_product(quat_ell[None,:,:], quat_azz[:,None,:]) # num_pose_az,num_pose_el,4
    quat_el_az = quat_el_az.view(-1,4)                  # num_pose_az*num_pose_el,4
    _quat = hamilton_product(quat_el_az, qq_base[None,...]).float()

    return _quat


def camera_quat_to_position_az_el(quat, initial_quat_bias_deg=0.):
    """Quat: N,4"""
    assert(quat.dim()==2)
    assert(quat.shape[1]==4)
    quat    = quat_to_camera_position(quat, initial_quat_bias_deg=initial_quat_bias_deg)
    quat_uv = convert_3d_to_uv_coordinates(quat)
    return quat_uv


def quat_to_camera_position(quat, initial_quat_bias_deg):
    """Quat: N,4"""
    X = torch.zeros((quat.shape[0],1,3),dtype=torch.float32,device=quat.device)
    X[:,:,2] = -1
    new_quat = quat_inverse(quat)

    xxx_base   = [1.,0.,0.]
    aaa_base   = initial_quat_bias_deg
    axis_base  = torch.tensor(xxx_base)
    angle_base = torch.tensor(aaa_base) / 180. * np.pi
    qq_base    = axisangle2quat(axis_base, angle_base) # 4
    new_quat   = hamilton_product(qq_base[None,:], new_quat)

    new_quat = hamilton_product(axisangle2quat( torch.eye(3, dtype=torch.float32)[0], torch.tensor(np.pi/2))[None,:],  new_quat ) # rotate 90deg about X
    new_quat = hamilton_product(axisangle2quat( torch.eye(3, dtype=torch.float32)[2], torch.tensor(-np.pi/2))[None,:], new_quat ) # rotate -90deg about Z
    rotX     = quat_rotate(X, new_quat).squeeze(1)   # ...,3
    return rotX

def convert_3d_to_uv_coordinates(X):
    """
    X : N,3
    Returns UV: N,2 normalized to [-1, 1]
    U: Azimuth: Angle with +X [-pi,pi]
    V: Inclination: Angle with +Z [0,pi]
    """
    if type(X) == torch.Tensor:
        eps=1e-4
        rad = torch.norm(X, dim=-1).clamp(min=eps)
        theta = torch.acos( (X[..., 2] / rad).clamp(min=-1+eps,max=1-eps) )  # Inclination: Angle with +Z [0,pi]
        phi = torch.atan2(X[..., 1], X[..., 0])                              # Azimuth    : Angle with +X [-pi,pi]
        vv = (theta / np.pi) * 2 - 1
        uu = ((phi + np.pi) / (2*np.pi)) * 2 - 1
        uv = torch.stack([uu, vv],dim=-1)
    else:
        rad = np.linalg.norm(X, axis=-1)
        rad = np.clip(rad, 1e-12, None)
        theta = np.arccos(X[..., 2] / rad)                                   # Inclination: Angle with +Z [0,pi]
        phi = np.arctan2(X[..., 1], X[..., 0])                               # Azimuth    : Angle with +X [-pi,pi]
        vv = (theta / np.pi) * 2 - 1
        uu = ((phi + np.pi) / (2*np.pi)) * 2 - 1
        uv = np.stack([uu, vv],-1)
    return uv