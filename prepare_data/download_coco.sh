#!/bin/bash

if [ ! -d "data" ]; then
    mkdir "data"
fi

# download coco images
if [ ! -d "data/coco" ]; then
    mkdir "data/coco"
fi
cd data/coco
mkdir images
cd images
wget wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
rm train2017.zip