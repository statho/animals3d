import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import utils.image_utils as image_utils
from utils.data_utils import GenerateHeatmap


class ImageDataset(Dataset):
	'''
	Dataset class for evaluation
	'''
	def __init__(self, args, mode='train'):
		self.mode = mode
		category = args.category
		self.stats = {'mean': torch.Tensor([0.485, 0.456, 0.406]), 'std': torch.Tensor([0.229, 0.224, 0.225])}
		self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		self.transform = T.Compose([ T.ToTensor(), self.normalize ])
		self.input_res = args.input_res
		self.output_res = args.output_res
		self.center = np.array( (self.input_res/2, self.input_res/2) )
		self.scale = self.input_res / 200
		conf_list = [0.3]*args.njoints
		self.generate_heatmap = GenerateHeatmap(self.output_res, args.njoints, conf_list)

		if self.mode == 'train':

			filelist = []
			annos = []

			if args.use_pascal:
				# load filelist
				fname = f'../acsm/acsm/cachedir/data/pascal/filelists/{category}_train.txt'
				with open(fname, 'r') as f:
					pascal_filelist = list(map(lambda x: x.rstrip(), f.readlines()))
				num_imgs = len(pascal_filelist)
				print('Pascal: {} training images'.format(num_imgs))

				# load annoatations
				anno_file = f'../acsm/acsm/cachedir/data/pascal/annotations/{category}_all.json'
				with open(anno_file) as f:
					anno = json.load(f)
				for ann in anno:
					ann['img_path'] = os.path.join( f'../data/pascal/images', ann['img_path'] )

				annos    += anno
				filelist += pascal_filelist

			if args.use_coco:
				# load filelist
				fname = f'../acsm/acsm/cachedir/data/coco/filelists/{category}_train.txt'
				with open(fname, 'r') as f:
					coco_filelist = list(map(lambda x: x.rstrip(), f.readlines()))
				num_imgs = len(coco_filelist)
				print('Coco: {} training images'.format(num_imgs))

				# load annoatations
				anno_file = f'../acsm/acsm/cachedir/data/coco/annotations/{category}.json'
				with open(anno_file) as f:
					anno = json.load(f)
				for ann in anno:
					ann['img_path'] = os.path.join( f'../data/coco/images', ann['img_path'] )

				annos    += anno
				filelist += coco_filelist

			self.filelist  = filelist
			self.anno_dict = {str(anno['img_id']): anno for anno in annos}

		else:

			dataset = 'coco' if category in ['giraffe', 'bear'] else 'pascal'

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

		# crop image and transform joints
		img = image_utils.crop(img, center, scale, (self.input_res, self.input_res))
		for j in range(num_joints):
			if joints[j, 2] > 0:
				joints[j, :2] = image_utils.transform(joints[j, :2], center, scale, (self.input_res, self.input_res))

		# image augmentations
		aug_rot = 0
		scale = self.scale
		if self.mode == 'train':
			aug_rot = (2*np.random.random() - 1) * 30.0
			aug_scale = (1.25-0.75)*np.random.random()+0.75
			scale *= aug_scale

		# get the transformation matrix for input and output
		input_matrix  = image_utils.get_transform(self.center, scale, (self.input_res, self.input_res), aug_rot)  [:2]
		output_matrix = image_utils.get_transform(self.center, scale, (self.output_res, self.output_res), aug_rot)[:2]

		if self.mode == 'train':
			img = cv2.warpAffine(img, input_matrix, (self.input_res, self.input_res))
		image_tensor = self.transform(np.uint8(img))

		for j in range(num_joints):
			if joints[j, 2] > 0:
				joints[j,:2] = np.dot( output_matrix, np.array([joints[j,0], joints[j,1], 1.0]) )
		heatmap, usage_indicator = self.generate_heatmap(joints)

		batch = {}
		batch['image'] = image_tensor
		batch['heatmap_gt'] = heatmap
		batch['target_weights'] = usage_indicator
		batch['joints_conf'] = joints[:, 2]
		return batch