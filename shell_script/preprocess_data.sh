#!/bin/bash

#SBATCH -A mcintoshgroup_gpu
#SBATCH --reservation=mcintoshgroup_gpu1
#SBATCH -t 70:00:00
#SBATCH --mem=40G
#SBATCH -J preprocess_radchestCT
#SBATCH -p gpu
#SBATCH -c 10
#SBATCH -N 1
#SBATCH --gres=gpu:l40:1
#SBATCH --begin=now

source activate merlin

# python /cluster/home/t135419uhn/Merlin/preprocess_fp16_ctrate.py

python /cluster/home/t135419uhn/Merlin/preprocess_raw_radchestct.py
