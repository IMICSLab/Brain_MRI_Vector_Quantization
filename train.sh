#!/bin/bash 
#SBATCH -N 1 -c 2
#SBATCH --mem=128G
#SBATCH --time=72:00:00 ## this requests one day of walltime
#SBATCH --gres=gpu:1
#SBATCH -e
#SBATCH -o

module load python/3.9.2_torch_gpu
cd /home/szhou/pLGG

python3 ./train_ae_128.py
# python3 ./train_transformer.py