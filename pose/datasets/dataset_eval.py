import os
import json
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T
import utils.image_utils as image_utils
from utils.data_utils import GenerateHeatmap


class ImageDatasetEval(Dataset):
	'''
	Dataset class for evaluation
	'''
	def __init__(self, args):
		dataset = args.dataset
		category = args.category
		self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		self.transform = T.Compose([ T.ToTensor(), self.normalize ])
		self.input_res = args.input_res
		self.output_res = args.output_res
		self.center = np.array( (self.input_res/2, self.input_res/2) )
		self.scale = self.input_res / 200
		conf_list = [0.3]*args.njoints
		self.generate_heatmap = GenerateHeatmap(self.output_res, args.njoints, conf_list)

		# load filelist
		fname = f'../acsm/acsm/cachedir/data/{dataset}/filelists/{category}_val.txt'
		with open(fname, 'r') as f:
			self.filelist = list(map(lambda x: x.rstrip(), f.readlines()))

		# load annoatations
		anno_file = f'../acsm/acsm/cachedir/data/{dataset}/annotations/{category}.json'
		with open(anno_file) as f:
			annos = json.load(f)
		for anno in annos:
			anno['img_path'] = os.path.join( f'../data/{dataset}/images', anno['img_path'] )
		self.anno_dict = {str(anno['img_id']): anno for anno in annos}

	def __len__(self):
		return len(self.filelist)

	def __getitem__(self, idx):
		img_id = self.filelist[idx]

		# get image
		img_path = self.anno_dict[img_id]['img_path']
		img = image_utils.opencv_loader(img_path)

		# get bbox
		bbox = self.anno_dict[img_id]['img_bbox']
		center = ( (bbox[0] + (bbox[0]+bbox[2]) ) / 2, (bbox[1] + (bbox[1]+bbox[3])) / 2  )
		scale = max(bbox[2], bbox[3]) / 200

		# get keypoints
		joints = np.array(self.anno_dict[img_id]['joints'])
		num_joints = len(joints)

		# crop image, transorm joints and generate heatmaps for the joints
		img = image_utils.crop(img, center, scale, (self.input_res, self.input_res))
		for j in range(num_joints):
			if joints[j, 2] > 0:
				joints[j, :2] = image_utils.transform(joints[j, :2], center, scale, (self.input_res, self.input_res))
		image_tensor = self.transform(np.uint8(img))

		output_matrix = image_utils.get_transform(self.center, self.scale, (self.output_res, self.output_res))[:2]
		for j in range(num_joints):
			if joints[j, 2] > 0:
				joints[j,:2] = np.dot( output_matrix, np.array([joints[j,0], joints[j,1], 1.0]) )
		_, usage_indicator = self.generate_heatmap(joints)

		batch = {}
		batch['image'] = image_tensor
		batch['keypoints_gt'] = np.array(self.anno_dict[img_id]['joints'])
		batch['target_weights'] = usage_indicator
		batch['center'] = np.array(center)
		batch['scale'] = scale
		return batch