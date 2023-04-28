import os
import cv2
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T
import utils.image_utils as image_utils


class UnlabeledImages(Dataset):
	'''
	Dataset used for creating keypoints pseudo-labels on web images
	'''
	def __init__(self, args):
		category = args.category
		self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		self.transform = T.Compose([ T.ToTensor(), self.normalize ])
		self.input_res = args.input_res
		self.output_res = args.output_res
		self.center = np.array( (self.input_res/2, self.input_res/2) )
		self.scale = self.input_res / 200

		# load annotations
		anno_file =  f'../data/yfcc100m/labels_0/{category}_bbox.json'
		with open(anno_file) as f:
			annos = json.load(f)
		for anno in annos:
			anno['img_path'] = os.path.join( f'../data/yfcc100m/images/{category}', anno['img_path'] )
		self.anno_dict = {str(anno['img_id']): anno for anno in annos}
		self.filelist = list(self.anno_dict.keys())

	def __len__(self):
		return len(self.filelist)

	def __getitem__(self, idx):
		img_id   = self.filelist[idx]

		# get image
		img_path = self.anno_dict[img_id]['img_path']
		img = image_utils.opencv_loader(img_path)

		# get bbox
		bbox = self.anno_dict[img_id]['img_bbox']
		center = ( (bbox[0] + (bbox[0]+bbox[2]) ) / 2, (bbox[1] + (bbox[1]+bbox[3])) / 2  )
		scale = max(bbox[2], bbox[3]) / 200

		# crop image
		img = image_utils.crop(img, center, scale, (self.input_res, self.input_res))
		image_tensor = self.transform(np.uint8(img))

		batch = {}
		batch['img_id'] = img_id
		batch['image'] = image_tensor
		batch['center'] = np.array(center)
		batch['scale'] = scale
		return batch


class UnlabeledImagesMT(Dataset):
	'''
	Dataset used for creating keypoint pseudo-labels on web images with multiple transformations
	'''
	def __init__(self, args, scales=[1.1, 1.2], rot=[-20, -10, 10, 20]):
		category = args.category
		self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		self.transform = T.Compose([ T.ToTensor(), self.normalize ])
		self.input_res = args.input_res
		self.output_res = args.output_res
		self.center = np.array( (self.input_res/2, self.input_res/2) )
		self.scale = self.input_res / 200

		self.scales = scales
		self.rot_angles = rot

		# load annotations
		anno_file =  f'../data/yfcc100m/labels_0/{category}_bbox.json'
		with open(anno_file) as f:
			annos = json.load(f)
		for anno in annos:
			anno['img_path'] = os.path.join( f'../data/yfcc100m/images/{category}', anno['img_path'] )
		self.anno_dict = {str(anno['img_id']): anno for anno in annos}
		self.filelist = list(self.anno_dict.keys())

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

		# crop image and save inverse transformation matrices
		img = image_utils.crop(img, center, scale, (self.input_res, self.input_res))
		image_tensors = self.transform(np.uint8(np.copy(img))) [None, ...]
		# transformation matrix with no scale/rotation augmentation
		inv_aug_matrices = image_utils.inv_mat( image_utils.get_transform(self.center, self.scale, (self.input_res, self.input_res))[:2] )[None, ...]

		for scale in self.scales:
			aug_matrix = image_utils.get_transform(self.center, self.scale * scale, (self.input_res, self.input_res))[:2]
			transf_img = self.transform(np.uint8( cv2.warpAffine( np.copy(img), np.copy(aug_matrix), (self.input_res, self.input_res) ) ))
			image_tensors = torch.cat( (image_tensors, transf_img[None, ...]) )
			inv_aug_matrices = np.concatenate( ( inv_aug_matrices, image_utils.inv_mat(aug_matrix) [None, ...] ) )

		for aug_rot in self.rot_angles:
			aug_matrix = image_utils.get_transform(self.center, self.scale, (self.input_res, self.input_res), aug_rot)[:2]
			transf_img = self.transform(np.uint8( cv2.warpAffine( np.copy(img), np.copy(aug_matrix), (self.input_res, self.input_res) ) ))
			image_tensors = torch.cat( (image_tensors, transf_img[None, ...]) )
			inv_aug_matrices = np.concatenate( (inv_aug_matrices, image_utils.inv_mat(aug_matrix) [None, ...]) )

		batch = {}
		batch['img_id'] = img_id
		batch['image'] = image_tensors
		batch['center'] = np.array(center)
		batch['scale'] = max(bbox[2], bbox[3]) / 200
		batch['inv_aug_matrices'] = inv_aug_matrices
		return batch