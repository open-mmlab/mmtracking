#!/usr/bin/env bash
set -x
# ----configs----
config=$1
part=$2
gpu_nums=$3
gpu_per=$4
resume=${5:-0}
# --------------
folder=work_dirs
# -----------------
ROOT=.

if [[ $resume = *"resume"* ]]; then
    resume_from=./${folder}/${config}/latest.pth
else
    resume_from=""
fi

export PYTHONPATH=`pwd`:$PYTHONPATH

# train
srun -p ${part} --gres=gpu:${gpu_per} -n${gpu_nums} --ntasks-per-node=${gpu_per} \
--job-name=python --kill-on-bad-exit=1 \
python3 -u ${ROOT}/tools/train.py \
./configs/${config}.py \
--work-dir=./${folder}/${config} \
--resume-from=${resume_from} \
--launcher='slurm'

echo "Config: " ${config}
cp ./configs/${config}.py ./${folder}/${config}/.
