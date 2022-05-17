pipeline = [
    dict(
        type='mmtrack.TransformBroadcaster',
        share_random_params=True,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='mmtrack.LoadTrackAnnotations', with_instance_id=True),
            dict(type='Resize', scale=(1000, 600), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
        ]),
    dict(type='mmtrack.ConcatVideoReferences', ref_prefix='ref'),
    dict(type='mmtrack.PackTrackInputs', ref_prefix='ref')
]
