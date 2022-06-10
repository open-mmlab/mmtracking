
srun --partition=mm_video \
    --job-name=mmtrack \
    --gres=gpu:2 \
    --ntasks=2 \
    --quotatype=spot \
    --ntasks-per-node=2 \
    --cpus-per-task=4 \
    --kill-on-bad-exit=1 \
python tools/train.py configs/det/faster-rcnn_r50_fpn_4e_mot17-half.py --launcher="slurm" 
