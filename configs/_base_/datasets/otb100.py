# test pipeline
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadTrackAnnotations', with_instance_id=False),
    dict(type='PackTrackInputs', pack_single_img=True)
]

# dataloader
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='VideoSampler'),
    dataset=dict(
        type='OTB100Dataset',
        data_root='data/',
        ann_file='OTB100/annotations/otb100_infos.txt',
        data_prefix=dict(img_path='OTB100'),
        pipeline=test_pipeline,
        test_mode=True))
test_dataloader = val_dataloader

# evaluator
val_evaluator = dict(
    type='SOTMetric',
    metric='OPE',
    metric_options=dict(only_eval_visible=False))
test_evaluator = val_evaluator

# runner loop
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
