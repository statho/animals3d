# Data preparation
In a nutshell, we train our models with 150 images labeled with 2D keypoints from [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) and/or [COCO](https://cocodataset.org/#home). We also evaluate our models on [Animal Pose](https://sites.google.com/view/animal-pose/) dataset. You can download annotations for those datasets, the necessasry files (e.g., animal meshes etc), and model weights from [here](https://drive.google.com/file/d/14NTnURgs2RX2WNJIFeSt0fCfzl5zxdBj/view?usp=sharing). Place the downloaded cachedir.zip file in `acsm/acsm/` and extract it.

Download the images for Pascal VOC by running the following command:
```
. prepare_data/download_pascal.sh
```

Download the images for COCO by running the following command:
```
. prepare_data/download_coco.sh
```

In case you wish to evaluate your models or the provided ones on Animal Pose, you can download the dataset images from [images.zip](https://drive.google.com/drive/folders/1xxm6ZjfsDSmv6C9JvbgiGrmHktrUjV5x), and then unzip them to `data/animal_pose/images'.

The expected folder structure at the end of processing should look like:
```
data
|-- animal_pose
    |-- images
|-- coco
    |-- images
|-- pascal
    |-- images
```