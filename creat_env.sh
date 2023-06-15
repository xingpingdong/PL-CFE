#!/bin/bash

#conda create -y --name PL-CFE python=3.6
#source activate PL-CFE
#conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
#pip install torchmeta==1.2.0
#pip install -U scikit-learn

conda create -y --name PL-CFE python=3.9
source activate PL-CFE
#conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
#conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
#conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=11.3 -c pytorch -c conda-forge -y
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install torchmeta==1.8.0
pip install -U scikit-learn
pip install haven-ai==0.6.7