_base_ = [
    '../../_base_/models/qdtrack_faster-rcnn_r50_fpn.py',
    '../../_base_/datasets/bdd100k_track.py', '../../_base_/default_runtime.py'
]
model = dict(
    type='TNT',
    pretrains=None,
    # TNT params
    auto_corr=True,
    use_bbox_gt=True,
    use_track_gt=True,
    use_ref_bbox_gt=True,
    use_ref_rois=False,
    cross_nms_thr=0.5,
    # -----------
    frozen_modules='detector',
    detector=dict(
        roi_head=dict(bbox_head=dict(num_classes=8)),
        test_cfg=dict(score_thr=0.001)),
    track_head=dict(
        type='TrackNoTrackHead',
        match_method='dense_aggregation',
        match_with_gumbel=False,
        embed_head=dict(
            loss_track_aux=dict(
                type='L2Loss',
                neg_pos_ub=3,
                pos_margin=0,
                neg_margin=0.3,
                hard_mining=True,
                loss_weight=1.0),
            # loss_track_aux=None,
        )),
    tracker=dict(init_score_thr=0.8, obj_score_thr=0.5))
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    # test=dict(ann_file='data/bdd100k/tracking/annotations/3vid.json'),
)
# optimizer
optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[8, 11])
# runtime settings
total_epochs = 12
evaluation = dict(metric=['track'], interval=1)
log_config = dict(interval=50)
search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDSw', 'MT', 'ML']
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
load_from = 'ckpts/bdd100k/auto-corr_bdd100k-det_50e-7566dc89_new.pth'
dist_params = dict(port='12345')
