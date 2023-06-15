#!/bin/bash
#SBATCH --gres=gpu:a100:2
#SBATCH --tasks=2
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00

nvidia-smi
export CUDA_VISIBLE_DEVICES=0,1

source activate PL-CFE

cd ..

python3 trainval.py -e pretrain_en_omni -sb ./logs/pretraining -d ../data

python3 trainval_un.py -e finetune_en_omni -sb ./logs/finetune-omni -d ../data

python3 test_score.py -sb ./logs/finetune-omni -d ../data