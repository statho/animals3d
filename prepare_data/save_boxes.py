'''
Script used for detection bounding boxes on images downloaded from the web.

Example usage:
python prepare_data/save_boxes.py --category horse \
--opts MODEL.WEIGHTS detectron2://COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl

Running the above will detect and save bounding boxes for the horse category.
'''

import os
import json
import uuid
import torch
import argparse
import multiprocessing as mp
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode


class PredictorClass(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.parallel = parallel
        self.predictor = DefaultPredictor(cfg)

def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="prepare_data/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--category", default="horse")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.95,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    class_dict = {'horse': 17, 'sheep': 18, 'cow': 19, 'giraffe': 23, 'bear': 21}
    cat_id = class_dict[args.category]
    img_dir  = f'data/yfcc100m/images/{args.category}'
    filename = f'data/yfcc100m/labels_0/{args.category}.txt'
    save_file = f'data/yfcc100m/labels_0/{args.category}_bbox.json'
    with open(filename, 'r') as f:
        filelist = list(map( lambda x: x.strip(), f.readlines() ))

    # predictor
    cfg = setup_cfg(args)
    pred = PredictorClass(cfg)

    all_annos = []
    for img_name in tqdm(filelist):
        img_path = os.path.join(img_dir, img_name)
        img = read_image(img_path, format="BGR")

        predictions = pred.predictor(img)

        pred_instances = predictions['instances']
        boxes   = pred_instances.pred_boxes
        scores  = pred_instances.scores
        classes = pred_instances.pred_classes

        anno_list = []
        for b, cat in enumerate(classes):
            if cat == cat_id:
                img_id = str(uuid.uuid1())
                box    = list(map( int, list(boxes[b].tensor.squeeze(0).cpu().numpy()) ))
                box[2]-= box[0]
                box[3]-= box[1]
                score  = scores[b].item()
                anno_dict = {}
                anno_dict['img_id']    = img_id
                anno_dict['img_path']  = img_name
                anno_dict['img_bbox']  = box
                anno_dict['box_score'] = score
                anno_list.append(anno_dict)
        if anno_list:
            all_annos += anno_list

    print(f'=> Saving detections for {len(all_annos)} instances')
    with open(save_file, 'w') as f:
        json.dump(all_annos, f)