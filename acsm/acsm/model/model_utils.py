'''
Code borrowed from
https://github.com/nileshkulkarni/acsm/blob/master/acsm/nnutils/model_utils.py
'''

from __future__ import absolute_import, division, print_function
import torch
import numpy as np
import pickle as pkl
import os.path as osp
import scipy.io as sio

import constants


def load_template_shapes(opts, device=None):
    model_dir     = f'acsm/cachedir/3D_models/{opts.category}'
    mpath         = osp.join(model_dir, 'mean_shape.mat')
    mean_shape    = load_mean_shape(mpath, device)
    mean_shape_np = sio.loadmat(mpath)
    return model_dir, mean_shape, mean_shape_np

def load_mean_shape(mean_shape_path, device=None):
    mean_shape = sio.loadmat(mean_shape_path)
    mean_shape['uv_verts']     = torch.from_numpy(mean_shape['uv_verts']).float().to(device)     # [642, 2] verts in uv domain
    mean_shape['sphere_verts'] = torch.from_numpy(mean_shape['sphere_verts']).float().to(device) # [642, 3] verts mapped to the sphere
    mean_shape['verts']        = torch.from_numpy(mean_shape['verts']).float().to(device)        # [642, 3] verts in template shape
    mean_shape['uv_map']       = torch.from_numpy(mean_shape['uv_map']).float().to(device)       # [1001, 1001, 2]
    mean_shape['faces']        = torch.from_numpy(mean_shape['faces']).long().to(device)         # [1280, 3] faces in template shape
    mean_shape['face_inds']    = torch.from_numpy(mean_shape['face_inds']).long().to(device)     # [1001, 1001]
    if 'sublookup_type' in mean_shape.keys():
        mean_shape['sublookup_type']     = torch.from_numpy(mean_shape['sublookup_type']).long().to(device)
        mean_shape['sublookup_offset']   = torch.from_numpy(mean_shape['sublookup_offset']).float().to(device)
        mean_shape['sublookup_length']   = torch.from_numpy(mean_shape['sublookup_length']).float().to(device)
        mean_shape['sublookup_faceinds'] = torch.from_numpy(mean_shape['sublookup_faceinds']).long().to(device)
    return mean_shape

def init_dpm(model_dir, mean_shape, parts_file, device=None):
    # get the vertex_id from the template mesh for each keypoint
    kp_vertex_ids = []
    kp2verts_file = osp.join(model_dir, 'kp2vertex.txt')
    with open(kp2verts_file) as f:
        kp2verts = { l.strip().split()[0]: int(l.strip().split()[1]) for l in f.readlines() }
        for kp_name in constants.QUAD_JOINT_NAMES:
            kp_vertex_ids.append(kp2verts[kp_name])
    kp_vertex_ids = torch.LongTensor(kp_vertex_ids).to(device)

    partPkl = osp.join(model_dir, 'parts.pkl')
    with open(partPkl, 'rb') as f:
        u = pkl._Unpickler(f)
        u.encoding = 'latin1'
        dpm = u.load()
    nparts                 = dpm['alpha'].shape[1]
    mean_shape['alpha']    = dpm['alpha'] = torch.FloatTensor(dpm['alpha'])

    # RC for the 0th part should be 0,0,0
    mean_shape['parts_rc'] = [ torch.FloatTensor(dpm['nodes'][1]['rc'] * 0).to(device)]
    mean_shape['parts_rc'].extend([ torch.FloatTensor(dpm['nodes'][i]['rc']).to(device) for i in range(1, nparts) ])
    mean_shape['parts']    = dpm['nodes']
    assert not (parts_file == ''), 'please specify active parts file'

    with open(parts_file, 'r') as f:
        parts_data = [l.strip().split() for l in f.readlines()]

    return dpm, parts_data, kp_vertex_ids

def load_active_parts(model_dir, save_dir, dpm, parts_data, suffix='train'):
    active_part_names = {k[0]: k[1] for k in parts_data}
    part_axis         = {k[0]: np.array([int(t) for t in k[2:]]) for k in parts_data}

    aa = sorted(active_part_names.keys())
    bb = sorted([k['name'] for k in dpm['nodes']])
    assert (aa == bb), 'part names do not match'

    # with open(osp.join(save_dir, 'active_parts_{}.txt'.format(suffix)), 'w') as f:
    #     for key in active_part_names.keys():
    #         f.write('{} {}\n'.format(key, active_part_names[key]))

    part_axis_init    = []
    part_active_state = []
    for _, key in enumerate([k['name'] for k in dpm['nodes']]):
        part_active_state.append(active_part_names[key] == 'True')
        part_axis_init.append(part_axis[key])

    with open(osp.join(model_dir, 'mirror_transforms.txt')) as f:
        mirror_pairs = [tuple(l.strip().split()) for l in f.readlines()]
        mirror_pairs = {v1: v2 for (v1, v2) in mirror_pairs}
        part_perm    = []
        name2index   = { key: ex for ex, key in enumerate([k['name'] for k in dpm['nodes']])}

        for _, key in enumerate([k['name'] for k in dpm['nodes']]):
            mirror_key = mirror_pairs[key]
            part_perm.append(name2index[mirror_key])
        part_perm = torch.LongTensor(part_perm)

    return part_active_state, part_axis_init, part_perm