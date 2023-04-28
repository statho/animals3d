'''
Script used for data selection.

Example usage:
python prepare_data/data_selection.py --category horse --filter kpconf --selection_num 1000 --visualize

Running the above will
(i) create a filelist (in data/yfcc100m/filelists) with 1000 annotation ids for the horse category according to KP-conf criterion;
(ii) save some indicative visualization in data/yfcc100m/visuals.
'''

import json
import argparse
import vis_utils
from selection_criteria import kp_conf, cf_mt, cf_cm, cf_cm_sq

parser = argparse.ArgumentParser()
parser.add_argument('--category',  		default='horse',  type=str,  help='category to use')
parser.add_argument('--filter',   		default='kpconf', type=str,  help='method to choose good images', choices=['kpconf', 'cf_mt', 'cf_cm', 'cf_cm_sq'])
parser.add_argument('--selection_num',  default=1000,     type=int,  help='number of web images to be selected for training ACSM')
parser.add_argument('--visualize', 		default=False, action='store_true', help='visualize some predictions')
parser.add_argument('--vis_num',  		default=50,       type=int,  help='number of images to use for visualization')
args = parser.parse_args()


def filter(category, filter, selection_num=1000, visualize=False, vis_num=50):
	vis_num = args.vis_num if visualize else 0
	vis_path = f'data/yfcc100m/visuals/{category}_{filter}_{selection_num}' if visualize else None
	out_filelist_name = f'data/yfcc100m/filelists/{category}_{filter}_{selection_num}.txt'
	print(f'=> Selecting {selection_num} samples with {filter}')
	print(f'=> Saving new filelist in {out_filelist_name}')

	# keypoint pseudo-labels from 2D pose estimator
	anno_fname = f'data/yfcc100m/labels/{category}_pl_2d.json'
	with open(anno_fname) as f:
		annos = json.load(f)
	pl_2d_dict = {anno['img_id']: anno for anno in annos}


	### Data Selection ###

	if filter=='kpconf':
		sorted_filelist = kp_conf(pl_2d_dict)
		with open(out_filelist_name, 'w') as f:
			f.write('\n'.join(sorted_filelist[:selection_num]))
		if visualize:
			vis_utils.save_visuals(pl_2d_dict, category, sorted_filelist, selection_num, vis_num, vis_path)

	elif filter=='cf_mt':
		sorted_filelist = cf_mt(category)
		with open(out_filelist_name, 'w') as f:
			f.write('\n'.join(sorted_filelist[:selection_num]))
		if visualize:
			vis_utils.save_visuals(pl_2d_dict, category, sorted_filelist, selection_num, vis_num, vis_path)

	elif filter=='cf_cm':
		# keypoint pseudo-labels from auxiliary 2D pose estimator
		anno_fname = f'data/yfcc100m/labels/{category}_pl_2d_aux.json'
		with open(anno_fname) as f:
			annos = json.load(f)
		pl_2d_aux_dict = {anno['img_id']: anno for anno in annos}
		# run criterion
		sorted_filelist = cf_cm(pl_2d_dict, pl_2d_aux_dict)
		with open(out_filelist_name, 'w') as f:
			f.write('\n'.join(sorted_filelist[:selection_num]))
		if visualize:
			vis_utils.save_visuals(pl_2d_dict, category, sorted_filelist, selection_num, vis_num, vis_path)

	elif filter=='cf_cm_sq':
		# keypoint pseudo-labels after reprojection from the 3D mesh
		anno_fname = f'data/yfcc100m/labels/{category}_pl_3d.json'
		with open(anno_fname) as f:
			annos = json.load(f)
		pl_3d_dict = {anno['img_id']: anno for anno in annos}
		# run criterion
		sorted_filelist = cf_cm_sq(pl_2d_dict, pl_3d_dict)
		with open(out_filelist_name, 'w') as f:
			f.write('\n'.join(sorted_filelist[:selection_num]))
		if visualize:
			vis_utils.save_visuals(pl_2d_dict, category, sorted_filelist, selection_num, vis_num, vis_path)


if __name__ == '__main__':
	filter(category = args.category, filter = args.filter, selection_num = args.selection_num, visualize = args.visualize)