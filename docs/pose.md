# Pose Estimation
Download the weights for models trained with 150 labeled images from the [checkpoints.zip](https://drive.google.com/file/d/1ynhKPsiTfUmivNE9AnlCOQ8ZDOTvSKx4/view?usp=share_link) and extract it in `pose/results/checkpoints`. All commands that follow should be run from the `pose` directory.

## Evaluation
You can evaluaton the pose estimation network by running the following command:
```
python evaluate.py --dataset <dataset> --category <category>
```

## Training
You can train the pose estimation network with the default settings by running:
```
python train.py --use_pascal --use_coco --category <category> --name <name of experment>
```

## Generating PLs
You can generate keypoint PLs by running:
```
python generate_pl.py --category <cateogory> --name <name of experiment>
```

Additionally, you can generate keypoint PLs with multiple input transformations (needed for CF-MT criterion) with the following command:
```
python generate_pl_mt.py --category <cateogory> --name <name of experiment>
```