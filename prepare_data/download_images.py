'''
Script for downloading web images given a file with metadata.

Example usage:
python prepare_data/download_images.py --category horse

Running the above will download web images with horses and create a filelist with the downloaded images.
'''

import os
import cv2
import json
import argparse
import urllib.request
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Download data from YFCC100M for a specific category')
parser.add_argument('--category', type=str, required=True, choices=['horse', 'cow', 'sheep', 'giraffe', 'bear'])
args = parser.parse_args()


def download_images(categ: str, img_path: str):
	if not os.path.exists(img_path):
		os.makedirs(img_path)
	vids = f'prepare_data/categ_ids/{categ}.json'
	with open(vids) as f:
		categ_list = [json.loads(element) for element in list(map(lambda x: x.rstrip(), f.readlines()))]
	categ_list = list( filter(lambda x: 'item_download_url' in x and x['item_download_url'].endswith('.jpg'), categ_list) )
	fail_num = 0
	for element in tqdm(categ_list):
		try:
			save_name = os.path.join( img_path, element['item_download_url'].split('/')[-1] )
			urllib.request.urlretrieve(element['item_download_url'], save_name)
		except:
			fail_num += 1
	print(f'=> Could not download {fail_num} images')

def dhash(image, hashSize=8):
	gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	resized = cv2.resize(gray, (hashSize + 1, hashSize))
	# compute the (relative) horizontal gradient
	diff    = resized[:, 1:] > resized[:, :-1]
	# convert the difference image to a hash and return it
	return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def create_filelist(categ: str, img_path: str):
	out_dir = f'data/yfcc100m/labels_0'
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	out_file = f'{out_dir}/{categ}.txt'
	img_list = [img_name for img_name in os.listdir(img_path)]
	hashes = {}
	filelist = []
	for img_name in tqdm(img_list):
		img  = cv2.imread('{}/{}'.format(img_path, img_name))
		hash = dhash(img)
		if hash not in hashes:
			hashes[hash] = img_name
			filelist.append('{}'.format(img_name))
	with open(out_file, 'w') as f:
		f.write('\n'.join(filelist))
	print(f'=> Created filelist with {len(filelist)} images')


if __name__ == '__main__':
	img_path = f'data/yfcc100m/images/{args.category}'
	download_images(categ=args.category, img_path=img_path)
	create_filelist(categ=args.category, img_path=img_path)