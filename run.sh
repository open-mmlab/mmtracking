
srun --partition=mm_video \
    --job-name=mmtrack \
    --gres=gpu:6 \
    --ntasks=6 \
    --quotatype=auto \
    --ntasks-per-node=6 \
    --cpus-per-task=5 \
    --kill-on-bad-exit=1 \
    python tools/test.py /mnt/cache/lipengxiang/mmtracking/work_dirs/sort_faster-rcnn_fpn_4e_mot17-private/sort_faster-rcnn_fpn_4e_mot17-private.py  --checkpoint ../new_vf_track.pth --launcher="slurm" --eval track
    # python tools/train.py configs/mot/qdtrack/qdtrack_faster-rcnn_r101_fpn_12e_tao.py --launcher="slurm" --work-dir ~/work_dirs/mm_code_tao
    # python tools/train.py ./configs/mot/qdtrack/qdtrack_faster-rcnn_r101_fpn_24e_lvis.py --launcher="slurm" --work-dir ~/work_dirs/mm_code_3
    # python tools/train.py configs/mot/qdtrack/qdtrack_faster-rcnn_r101_fpn_12e_tao.py --launcher="slurm" --work-dir ~/work_dirs/mm_code_tao
    # python tools/test.py ./configs/mot/qdtrack/qdtrack_faster-rcnn_r101_fpn_12e_tao.py --checkpoint ../new.pth --launcher="slurm" --eval bbox
# python tools/train.py ./configs/mot/qdtrack/qdtrack_faster-rcnn_r101_fpn_24e_lvis.py --launcher="slurm" --work-dir ~/work_dirs/mm_code
# python tools/test.py ./configs/mot/qdtrack/qdtrack_faster-rcnn_r101_fpn_24e_lvis.py --checkpoint /mnt/lustre/lipengxiang/work_dirs/mm_code/latest.pth --launcher="slurm" --eval bbox
# python tools/test.py configs/tao/qdtrack_frcnn_r101_fpn_12e_tao_ft.py ./qdtrack_tao_20210812_221438-b6bd07e2.pth
# srun --partition=mm_video \
#     --job-name=train_mask_track \
#     --gres=gpu:8 \
#     --ntasks=8 \
#     --ntasks-per-node=8 \
#     --cpus-per-task=4 \
#     --kill-on-bad-exit=1 \
#     python -u tools/train.py ${CONFIG} --launcher="slurm" ${PY_ARGS}
# ./tools/slurm_train.sh mm_video train configs/vis/masktrack_rcnn/masktrack_rcnn_r50_fpn_12e_youtubevis2019.py /mnt/lustre/lipengxiang/work_dirs 2
