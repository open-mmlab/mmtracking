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
        type='GOT10kDataset',
        data_root='data/',
        ann_file='GOT10k/annotations/got10k_test_infos.txt',
        data_prefix=dict(img_path='GOT10k'),
        pipeline=test_pipeline,
        test_mode=True))
test_dataloader = val_dataloader

# evaluator
val_evaluator = dict(
    type='SOTMetric',
    format_only=True,
    metric_options=dict(dataset_type='got10k'))
test_evaluator = val_evaluator

# runner loop
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
