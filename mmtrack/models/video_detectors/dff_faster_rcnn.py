from ..builder import MODELS
from .dff_two_stage import DffTwoStage


@MODELS.register_module()
class DffFasterRCNN(DffTwoStage):

    def __init__(self,
                 backbone,
                 motion,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 key_frame_interval=10):
        super(DffFasterRCNN, self).__init__(
            backbone=backbone,
            motion=motion,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            key_frame_interval=key_frame_interval)
