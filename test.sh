#!/usr/bin/env bash
set -x
# ----configs----
config=$1
part=$2
gpu_nums=$3
PY_ARGS=${@:4}
# --------------
folder=work_dirs
# -----------------
ROOT=.

# test
srun -p ${part} --gres=gpu:${gpu_nums} -n${gpu_nums} --ntasks-per-node=${gpu_nums} \
--job-name=python --kill-on-bad-exit=1 \
python3 -u ${ROOT}/tools/test.py \
./configs/${config}.py \
./${folder}/${config}/latest.pth \
--out ./${folder}/${config}/output.pkl \
--launcher="slurm" \
${PY_ARGS}
