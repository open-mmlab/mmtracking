_base_ = [
    '../../_base_/models/qdtrack_faster-rcnn_r50_fpn.py',
    '../../_base_/datasets/bdd100k_track_joint_crop.py',
    '../../_base_/default_runtime.py'
]
model = dict(
    pretrains=None,
    frozen_modules=None,
    detector=dict(roi_head=dict(bbox_head=dict(num_classes=8))))
data = dict(samples_per_gpu=2, workers_per_gpu=2)
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=5000,
    warmup_ratio=1.0 / 1000,
    step=[16, 22])
# runtime settings
total_epochs = 24
evaluation = dict(metric=['bbox', 'track'], interval=24)
log_config = dict(interval=50)
search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs', 'MT', 'ML']
