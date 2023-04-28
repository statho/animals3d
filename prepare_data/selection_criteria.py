import os
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict


def kp_conf(pl_2d_dict):
	'''
	Keypoint Confidence criterion (KP-conf) from the paper (see section 3.2.1)
	Args:
	- pl_2d_dict: dictionary with PL from the 2D keypoint estimation network
	Returns:
	- sorted_filelist: list with annotations ids sorted based on KP-conf criterion
	'''
	kp_conf = {}
	for key in pl_2d_dict:
		joints = pl_2d_dict[key]['joints']
		kp_conf[key] = sum([joint[-1] for joint in joints])
	# descending order (of confidence)
	sorted_filelist = sorted(kp_conf, key=kp_conf.get, reverse=True)
	return sorted_filelist

def cf_mt(category):
	'''
	Multi-Transform Consistency critetion (CF-MT) from the paper (see section 3.2.1)
	Args:
	- category: str with category name
	Returns:
	- sorted_filelist: list with annotations ids sorted based on CF-MT criterion
	'''
	# get bounding boxes
	bbox_file = f'data/yfcc100m/labels_0/{category}_bbox.json'
	with open(bbox_file) as f:
		json_data = json.load(f)
	box_dict = { data['img_id']: data['img_bbox'] for data in json_data }

	# get PL from all transformations
	anno_dir = f'data/yfcc100m/labels_0/{category}'
	anno_files = os.listdir(anno_dir)
	transform_anno_dict = {}
	for filename in tqdm(anno_files):
		kp_file = f'{anno_dir}/{filename}'
		transform_id = filename.split('.')[0][-1]
		with open(kp_file) as f:
			kp_list = list(map( lambda x: x.strip().split(), f.readlines() ))
		kp_annos = defaultdict(list)
		for kp_anno in kp_list:
			kp_annos[kp_anno[0]].append( [ float(kp_anno[2]), float(kp_anno[3]), float(kp_anno[4]) ])
		transform_anno_dict[transform_id] = kp_annos

	# find average kp location from multi-transform inference
	inv_num = 1 / (len(anno_files)-1)
	img_ids = transform_anno_dict['0'].keys()
	kp_avg_dict = {}
	for img_id in img_ids:
		for transform_key in transform_anno_dict:
			if transform_key!='0':
				if img_id not in kp_avg_dict:
					kp_avg_dict[img_id] = inv_num * np.array(transform_anno_dict[transform_key][img_id])
				else:
					kp_avg_dict[img_id] += inv_num * np.array(transform_anno_dict[transform_key][img_id])

	# compute distance of untransformed kp and avg kp after multi-transform inference
	dist_dict = {}
	for img_id in img_ids:
		box = box_dict[img_id] # [x, y, w, h]
		kp_no_transf = np.array(transform_anno_dict['0'][img_id])
		kp_multi_transf = kp_avg_dict[img_id]
		# scale keypoints to a canonical size, so distances can be comparable
		width = box[2]
		height = box[3]
		scale_x = 256. / width
		scale_y = 256. / height
		kp_no_transf[:, 0]    *= scale_x
		kp_no_transf[:, 1]    *= scale_y
		kp_multi_transf[:, 0] *= scale_x
		kp_multi_transf[:, 1] *= scale_y
		dist = (kp_no_transf[:,:2] - kp_multi_transf[:,:2])**2
		dist = dist.sum(-1)**(1/2)
		dist = dist.mean()
		dist_dict[img_id] = dist
	# ascending order (of distance)
	sorted_filelist = sorted(dist_dict, key=dist_dict.get, reverse=False)
	return sorted_filelist


def cf_cm(pl_2d_dict, pl_2d_aux_dict):
	'''
	Cross-Model Consistency criterion (CF-CM) from the paper (see section 3.2.1 and Eq. (5))
	Args:
	- pl_2d_dict: dictionary with PL from the 2D keypoint estimation network
	- pl_2d_aux_dict: dictionary with PL from the auxiliary 2D keypoint estimation network
	Returns:
	- sorted_filelist: list with annotations ids sorted based on CF-CM criterion
	'''
	dist_dict = {}
	for key in pl_2d_dict:
		box   = pl_2d_dict[key]['img_bbox']
		kp_2d = np.array(pl_2d_dict[key]['joints'])
		kp_2d_aux = np.array(pl_2d_aux_dict[key]['joints'])
		### scale keypoints to a canonical size, so distances can be comparable
		width = box[2]
		height = box[3]
		scale_x = 256. / width
		scale_y = 256. / height
		kp_2d[:, 0] *= scale_x
		kp_2d[:, 1] *= scale_y
		kp_2d_aux[:, 0] *= scale_x
		kp_2d_aux[:, 1] *= scale_y
		dist = (kp_2d[:,:2] - kp_2d_aux[:,:2])**2
		dist = dist.sum(-1)**(1/2)
		dist = dist.mean()
		dist_dict[key] = dist
	# ascending order (of distance)
	sorted_filelist = sorted(dist_dict, key=dist_dict.get, reverse=False)
	return sorted_filelist


def cf_cm_sq(pl_2d_dict, pl_3d_dict):
	'''
	Cross-Model Cross-Modality Consistency criterion (CF-CM^2) from the paper (see section 3.2.1 and Eq. (6))
	Args:
	- pl_2d_dict: dictionary with PL from the 2D keypoint estimation network
	- pl_3d_dict: dictionary with PL from the 3D shape prediction network
	Returns:
	- sorted_filelist: list with annotations ids sorted based on CF-CM^2 criterion
	'''
	dist_dict = {}
	for key in pl_2d_dict:
		box = pl_2d_dict[key]['img_bbox'] # [x, y, w, h]
		kp_2d = np.array(pl_2d_dict[key]['joints'])
		kp_3d = np.array(pl_3d_dict[key]['joints'])
		# scale keypoints to a canonical size, so distances can be comparable
		width = box[2]
		height = box[3]
		scale_x = 256. / width
		scale_y = 256. / height
		kp_2d[:, 0] *= scale_x
		kp_2d[:, 1] *= scale_y
		kp_3d[:, 0] *= scale_x
		kp_3d[:, 1] *= scale_y
		dist = (kp_2d[:,:2] - kp_3d[:,:2])**2
		dist = dist.sum(-1)**(1/2)
		dist = dist.mean()
		dist_dict[key] = dist
	# ascending order (of distance)
	sorted_filelist = sorted(dist_dict, key=dist_dict.get, reverse=False)
	return sorted_filelist