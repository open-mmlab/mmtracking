_base_ = ['./mixformer_got10k.py']

# model setting
model = dict(
    type='MixFormer',
    backbone=dict(
        type='ConvVisionTransformer',
        num_stages=3,
        patch_size=[7, 3, 3],
        patch_stride=[4, 2, 2],
        patch_padding=[2, 1, 1],
        dim_embed=[64, 192, 384],
        num_heads=[1, 3, 6],
        depth=[1, 4, 16],
        mlp_ratio=[4, 4, 4],
        attn_drop_rate=[0.0, 0.0, 0.0],
        drop_rate=[0.0, 0.0, 0.0],
        drop_path_rate=[0.0, 0.0, 0.1],
        qkv_bias=[True, True, True],
        cls_token=[False, False, False],
        pos_embed=[False, False, False],
        qkv_proj_method=['dw_bn', 'dw_bn', 'dw_bn'],
        kernel_qkv=[3, 3, 3],
        padding_kv=[1, 1, 1],
        stride_kv=[2, 2, 2],
        padding_q=[1, 1, 1],
        stride_q=[1, 1, 1],
        norm_cfg=dict(type='BN', requires_grad=False)),
    head=dict(
        type='MixFormerHead',
        bbox_head=dict(
            type='CornerPredictorHead',
            inplanes=384,
            channel=384,
            feat_size=20,
            stride=16),
        score_head=dict(
            type='MixFormerScoreDecoder',
            pool_size=4,
            feat_size=20,
            stride=16,
            num_heads=6,
            hidden_dim=384,
            num_layers=3)),
    test_cfg=dict(
        search_factor=4.5,
        search_size=320,
        template_factor=2.0,
        template_size=128,
        update_interval=[25],
        online_size=[2],
        max_score_decay=[1.0],
    ))

data_root = 'data/'
data = dict(
    test=dict(
        type='TrackingNetDataset',
        ann_file=data_root +
        'trackingnet/annotations/trackingnet_test_infos.txt',
        img_prefix=data_root + 'trackingnet'))
