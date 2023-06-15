#!/bin/bash

nvidia-smi
#hostname

source activate PL-CFE
cd ../cfe
pretrain_path='../scripts/imagenet/checkpoint_0199.pth.tar'
#pretrain_path='../pretrain/imagenet-embedding.pth.tar'
dataset='tieredimagenet'
for s in 'train' 'test' 'val'
do
python cfe_encoding.py -p $pretrain_path -d $dataset -s $s
done