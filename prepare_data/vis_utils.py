import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
cm = plt.get_cmap('jet')

COLOR = (255, 0, 0)
JOINT_NAMES = [
            'L_eye', 'R_eye', 'L_ear', 'R_ear', 'Nose', 'Throat', 'Tail', 'Withers',
            'L_F_elbow', 'R_F_elbow', 'L_B_elbow', 'R_B_elbow', 'L_F_paw', 'R_F_paw', 'L_B_paw', 'R_B_paw'
]

def opencv_loader(path):
	return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def draw_bbox_kp(img, bbox, keypoints=None, diameter=2, color=COLOR, thickness=2):
	'''
	Visualizes a single bounding box and keypoints on the image
	'''
	if keypoints is not None:
		nkps = len(keypoints)
		kp_cmap = [cm(i * 255 // nkps) for i in range(nkps)]
		for i, (x, y, viz) in enumerate(keypoints):
			if kp_cmap is not None:
				color = kp_cmap[i]
			if float(viz)>0.1:
				cv2.circle(img, (int(x), int(y)), diameter, (color[0]*255, color[1]*255, color[2]*255) , -1)
				cv2.putText( img, '{}_{:.2f}'.format(JOINT_NAMES[i], viz), (int(x), int(y)),
							    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (color[0]*255, color[1]*255, color[2]*255), 1
				)
	x_min, y_min, x_max, y_max = bbox
	x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
	cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=COLOR, thickness=thickness)
	return img

def visualize(save_name, img, bbox, keypoints=None):
	img = draw_bbox_kp(img, bbox, keypoints)
	plt.imsave(save_name, img)

def save_visuals(pl_2d_dict, category, sorted_filelist, selection_num, vis_num, vis_path):
	best_samples = sorted_filelist[:vis_num]
	worst_best_samples = sorted_filelist[:selection_num][-20:]
	worst_samples = sorted_filelist[-20:]
	sample_list = [(best_samples, 'best'), (worst_best_samples, 'worst_best'), (worst_samples, 'worst')]
	for samples, visual_type in sample_list:
		save_path = f'{vis_path}/{visual_type}'
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		for img_id in samples:
			img_path     = pl_2d_dict[img_id]['img_path']
			img          = opencv_loader( f'data/yfcc100m/images/{category}/{img_path}' )
			x0, y0, w, h = pl_2d_dict[img_id]['img_bbox']
			bbox         = [x0, y0, x0+w, y0+h]
			keypoints    = np.array(pl_2d_dict[img_id]['joints'])
			save_name    = f'{save_path}/{img_id}.jpg'
			visualize(save_name, img, bbox, keypoints=keypoints)