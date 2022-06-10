srun --partition=mm_video \
    --job-name=mmtrack \
    --gres=gpu:1 \
    --ntasks=1 \
    --quotatype=auto \
    --ntasks-per-node=1 \
    --cpus-per-task=5 \
    --kill-on-bad-exit=1 \
    python tools/train.py ../iss.py