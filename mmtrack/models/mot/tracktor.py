# from mmtrack.core import track2result
# import cv2
# import numpy as np
from mmdet.core import bbox2result

from mmtrack.core import track2result
from ..builder import MODELS, build_detector, build_reid, build_tracker
from .base import BaseMultiObjectTracker


@MODELS.register_module()
class Tracktor(BaseMultiObjectTracker):

    def __init__(self,
                 detector=None,
                 reid=None,
                 tracker=None,
                 motion=None,
                 with_align=None,
                 with_public_bboxes=False,
                 pretrains=None):
        super().__init__()
        if detector is not None:
            self.detector = build_detector(detector)

        if reid is not None:
            self.reid = build_reid(reid)

        if tracker is not None:
            self.tracker = build_tracker(tracker)

        self.with_public_bboxes = with_public_bboxes
        self.init_weights(pretrains)

    def init_weights(self, pretrain):
        if pretrain is None:
            pretrain = dict()
        assert isinstance(pretrain, dict), '`pretrain` must be a dict.'
        if self.with_detector and pretrain.get('detector', False):
            self.init_module('detector', pretrain['detector'])
        if self.with_reid and pretrain.get('reid', False):
            self.init_module('reid', pretrain['reid'])

    def forward_train(self, *args, **kwargs):
        raise NotImplementedError(
            'Please train `detector` and `reid` models first and \
                inference with Tracktor.')

    def simple_test(self, img, img_metas, public_bboxes=None, rescale=False):
        frame_id = img_metas[0].get('frame_id', -1)
        if frame_id == 0:
            self.tracker.reset()

        x = self.detector.extract_feat(img)
        if hasattr(self.detector, 'roi_head'):
            proposals = self.detector.rpn_head.simple_test_rpn(x, img_metas)
            det_bboxes, det_labels = self.detector.roi_head.simple_test_bboxes(
                x,
                img_metas,
                proposals,
                self.detector.roi_head.test_cfg,
                rescale=rescale)
            # TODO: support batch inference
            det_bboxes = det_bboxes[0]
            det_labels = det_labels[0]
            num_classes = self.detector.roi_head.bbox_head.num_classes
        elif hasattr(self.detector, 'bbox_head'):
            num_classes = self.detector.bbox_head.num_classes
            raise NotImplementedError()
        else:
            raise TypeError('detector must has roi_head or bbox_head.')

        bboxes, labels, ids = self.tracker.track(
            img=img,
            model=self,
            bboxes=det_bboxes,
            labels=det_labels,
            frame_id=frame_id)

        track_result = track2result(bboxes, labels, ids, num_classes)
        bbox_result = bbox2result(det_bboxes, det_labels, num_classes)
        return dict(bbox_results=bbox_result, track_results=track_result)
