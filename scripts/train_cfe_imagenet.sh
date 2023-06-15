#!/bin/bash
#SBATCH --gres=gpu:a100:4
#SBATCH --tasks=4
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00

nvidia-smi
#hostname
export PATH="/home/xd1/miniconda3/bin:$PATH"
source /home/xd1/miniconda3/etc/profile.d/conda.sh
conda activate PL-CFE
#PL-CFE

save=imagenet
dir_imagenet=/home/xd1/datasets/imagenet
mkdir $save
cd $save
#cd ..
#v2
python -c 'import torch; print("GPU available? {}. Got {} GPUs.".format
(torch.cuda.is_available(),torch.cuda.device_count()))'

python ../../main_cfe_im84.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 256 \
  --mlp --cfe-t 0.2 --aug-plus --cos \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  ${dir_imagenet}

