_base_ = ['./mixformer_got10k.py']

# model setting
model = dict(
    type='MixFormer',
    backbone=dict(
        type='ConvolutionalVisionTransformer',
        spec=dict(
            NUM_STAGES=3,
            PATCH_SIZE=[7, 3, 3],
            PATCH_STRIDE=[4, 2, 2],
            PATCH_PADDING=[2, 1, 1],
            DIM_EMBED=[64, 192, 384],
            NUM_HEADS=[1, 3, 6],
            DEPTH=[1, 4, 16],
            MLP_RATIO=[4, 4, 4],
            ATTN_DROP_RATE=[0.0, 0.0, 0.0],
            DROP_RATE=[0.0, 0.0, 0.0],
            DROP_PATH_RATE=[0.0, 0.0, 0.1],
            QKV_BIAS=[True, True, True],
            CLS_TOKEN=[False, False, False],
            POS_EMBED=[False, False, False],
            QKV_PROJ_METHOD=['dw_bn', 'dw_bn', 'dw_bn'],
            KERNEL_QKV=[3, 3, 3],
            PADDING_KV=[1, 1, 1],
            STRIDE_KV=[2, 2, 2],
            PADDING_Q=[1, 1, 1],
            STRIDE_Q=[1, 1, 1],
            FREEZE_BN=True
        )
    ),
    head=dict(
        type='MixFormerHead',
        bbox_head=dict(
            type='MixformerCornerPredictorHead',
            inplanes=384,
            channel=384,
            feat_sz=20,
            stride=16,
            freeze_bn=False
        ),
        score_head=dict(
            type='ScoreDecoder',
            pool_size=4,
            feat_sz=20,
            stride=16,
            num_heads=6,
            hidden_dim=384,
            num_layers=3
        )
   ),
    test_cfg=dict(
        search_factor=4.55,
        search_size=320,
        template_factor=2.0,
        template_size=128,
        update_interval=[200],
        online_size=[2],
        max_score_decay=[1.0],
    )
)



data_root = 'data/'
data = dict(
    test=dict(
        type='LaSOTDataset',
        ann_file=data_root + 'lasot/annotations/lasot_test_infos.txt',
        img_prefix=data_root + 'lasot/LaSOTBenchmark'))
