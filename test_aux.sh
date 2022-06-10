srun --partition=mm_video \
    --job-name=mmtrack \
    --gres=gpu:8 \
    --ntasks=8 \
    --quotatype=auto \
    --ntasks-per-node=8 \
    --cpus-per-task=5 \
    --kill-on-bad-exit=1 \
    python tools/train.py ./configs/mot/qdtrack/qdtrack_faster-rcnn_r101_fpn_24e_lvis.py --launcher="slurm" --work-dir ~/work_dirs/mm_code_aux