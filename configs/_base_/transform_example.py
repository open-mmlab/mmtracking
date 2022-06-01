# TODO: delete this file
pipeline_1 = [
    dict(
        type='TransformBroadcaster',
        share_random_params=True,
        transforms=[
            dict(
                type='mmdet.RandomCrop',
                crop_size=(5, 5),
                bbox_clip_border=True),
        ])
]

pipeline_2 = [
    dict(
        type='TransformBroadcaster',
        share_random_params=True,
        transforms=[
            dict(type='mmdet.PhotoMetricDistortion'),
        ])
]
