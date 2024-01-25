from __future__ import absolute_import, division, print_function
import json
import imageio
import numpy as np
import os.path as osp
from absl import flags
from torch.utils.data import Dataset

from acsm.utils import image_utils
from acsm.datasets.base import BaseDataset


flags.DEFINE_string('category', 'horse', 'object category')
flags.DEFINE_string('filter',   'all',   'data selection mechanism')  # all, kpconf, cf_mt, cf_cm, cf_cm_sq
flags.DEFINE_integer('web_images_num',  3000, 'number of web images to use for training')


class ImageDataset(BaseDataset):
    '''
    Dataset class for training
    '''
    def __init__(self, opts):
        super(ImageDataset, self).__init__(opts)
        category = opts.category
        self._out_joints = opts.kp_anno
        self._out_mask  = False
        self._out_pose  = False

        filelist = []
        annos = []

        if opts.use_pascal:
            # load filelist
            fname = f'acsm/cachedir/data/pascal/filelists/{category}_train.txt'
            with open(fname, 'r') as f:
                pascal_filelist = list(map(lambda x: x.rstrip(), f.readlines()))
            num_imgs = len(pascal_filelist)
            print('Pascal: {} training images'.format(num_imgs))

            # load annoatations
            anno_file = f'acsm/cachedir/data/pascal/annotations/{category}_all.json'
            with open(anno_file) as f:
                anno = json.load(f)
            for ann in anno:
                ann['img_path'] = osp.join( f'../data/pascal/images', ann['img_path'] )

            annos    += anno
            filelist += pascal_filelist

        if opts.use_coco:
            # load filelist
            fname = f'acsm/cachedir/data/coco/filelists/{category}_train.txt'
            with open(fname, 'r') as f:
                coco_filelist = list(map(lambda x: x.rstrip(), f.readlines()))
            num_imgs = len(coco_filelist)
            print('Coco: {} training images'.format(num_imgs))

            # load annoatations
            anno_file = f'acsm/cachedir/data/coco/annotations/{category}.json'
            with open(anno_file) as f:
                anno = json.load(f)
            for ann in anno:
                ann['img_path'] = osp.join( f'../data/coco/images', ann['img_path'] )

            annos    += anno
            filelist += coco_filelist

        if opts.use_web_images:
            # load filelist
            fname = f'../data/yfcc100m/filelists/{category}.txt' if opts.filter == 'all' \
                        else f'../data/yfcc100m/filelists/{category}_{opts.filter}_{opts.web_images_num}.txt'
            with open(fname, 'r') as f:
                web_filelist = list(map(lambda x: x.rstrip(), f.readlines()))
            num_imgs = len(web_filelist)
            print(f'Web images: {num_imgs} training images')

            # load annoations
            anno_file = f'../data/yfcc100m/labels/{category}_pl_2d.json'
            with open(anno_file) as f:
                anno = json.load(f)
            for ann in anno:
                ann['img_path'] = osp.join( f'../data/yfcc100m/images/{category}', ann['img_path'] )

            annos    += anno
            filelist += web_filelist

        # create a unified filelist and anno_dict for all datasets
        self.filelist  = filelist
        self.anno_dict = {str(anno['img_id']): anno for anno in annos}
        self.num_imgs  = len(self.filelist)
        print(f'=> Using {self.num_imgs} images it total')


class ImageDatasetEval(BaseDataset):
    '''
    Dataset class for evaluation
    '''
    def __init__(self, opts):
        super(ImageDatasetEval, self).__init__(opts)
        dataset  = opts.dataset
        category = opts.category
        self._out_joints = opts.kp_anno
        self._out_mask  = opts.mask_anno
        self._out_pose  = opts.sfm_anno

        # load filelist
        fname = f'acsm/cachedir/data/{dataset}/filelists/{category}_val.txt'
        with open(fname, 'r') as f:
            self.filelist = list(map(lambda x: x.rstrip(), f.readlines()))
        self.num_imgs = len(self.filelist)

        # load annoatations
        anno_file = f'acsm/cachedir/data/{dataset}/annotations/{category}.json'
        with open(anno_file) as f:
            annos = json.load(f)
        for anno in annos:
            anno['img_path'] = osp.join( f'../data/{dataset}/images', anno['img_path'] )
        self.anno_dict = {str(anno['img_id']): anno for anno in annos}


class UnlabeledDataset(Dataset):
    '''
    Dataset for generating keypoint pseudo-labels on web images
    '''
    def __init__(self, opts):
        self.opts = opts
        self.img_size = opts.img_size
        self.img_dir = f'../data/yfcc100m/images/{opts.category}'

        # load annotations
        anno_file =  f'../data/yfcc100m/labels_0/{opts.category}_bbox.json'
        with open(anno_file) as f:
            annos = json.load(f)
        for anno in annos:
            anno['img_path'] = osp.join( f'../data/yfcc100m/images/{opts.category}', anno['img_path'] )
        self.anno_dict = {str(anno['img_id']): anno for anno in annos}
        self.filelist = list(self.anno_dict.keys())

    def __len__(self):
        return len(self.filelist)

    def crop_image(self, img, bbox):
        return image_utils.crop(img, bbox, bgval=1)

    def scale_image(self, img):
        bwidth       = np.shape(img)[0]
        bheight      = np.shape(img)[1]
        scale        = self.img_size / float(max(bwidth, bheight))
        img_scale, _ = image_utils.resize_img(img, scale)
        return img_scale, scale

    def __getitem__(self, index):
        # get image
        img_id = self.filelist[index]
        img_path = self.anno_dict[img_id]['img_path']
        img = imageio.imread(img_path) / 255.0
        if len(img.shape) == 2:
            img = np.repeat(np.expand_dims(img, 2), 3, axis=2)

        # get bbox (x0, y0, w, h)
        bbox = self.anno_dict[img_id]['img_bbox']
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        if not self.opts.tight_crop:
            bbox = image_utils.square_bbox(bbox)

        img = self.crop_image(img, bbox)
        trans = np.array(bbox[:2])
        img, scale = self.scale_image(img)
        img = np.transpose(img, (2, 0, 1))
        elem = { 'img_id': img_id, 'img': img, 'trans': trans, 'scale': scale }
        return elem