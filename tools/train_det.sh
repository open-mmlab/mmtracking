#!/bin/bash

#SBATCH --job-name=MAT_DET
#SBATCH --gres=gpu:1
#SBATCH --partition=mlvu
#SBATCH --time=2-00:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --output=./log_slurm/S-%x.%j.out

CONFIG=$1
WORK_DIR=$2

eval "$(conda shell.bash hook)"
conda activate mat

srun python train.py ${CONFIG} --work-dir=${WORK_DIR} --launcher="slurm"

# srun python -m torch.distributed.launch \
#     train.py \
#     $CONFIG \
#     --work-dir=${WORK_DIR} \
#     --launcher pytorch \
#     ${@:3}