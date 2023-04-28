#!/bin/bash

if [ ! -d "data" ]; then
    mkdir "data"
fi

# download Pascal VOC images
if [ ! -d "data/pascal" ]; then
    mkdir "data/pascal"
fi
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar
mv VOCdevkit/VOC2012/JPEGImages/ data/pascal
mv data/pascal/JPEGImages/ data/pascal/images/
rm VOCtrainval_11-May-2012.tar
rm -rf VOCdevkit