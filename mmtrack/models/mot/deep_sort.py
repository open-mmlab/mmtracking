# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmdet.models import build_detector

from mmtrack.core import outs2results
from ..builder import MODELS, build_motion, build_reid, build_tracker
from .base import BaseMultiObjectTracker


@MODELS.register_module()
class DeepSORT(BaseMultiObjectTracker):
    """Simple online and realtime tracking with a deep association metric.

    Details can be found at `DeepSORT<https://arxiv.org/abs/1703.07402>`_.
    """

    def __init__(self,
                 detector=None,
                 reid=None,
                 tracker=None,
                 motion=None,
                 pretrains=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if isinstance(pretrains, dict):
            warnings.warn('DeprecationWarning: pretrains is deprecated, '
                          'please use "init_cfg" instead')
            if detector:
                detector_pretrain = pretrains.get('detector', None)
                if detector_pretrain:
                    detector.init_cfg = dict(
                        type='Pretrained', checkpoint=detector_pretrain)
                else:
                    detector.init_cfg = None
            if reid:
                reid_pretrain = pretrains.get('reid', None)
                if reid_pretrain:
                    reid.init_cfg = dict(
                        type='Pretrained', checkpoint=reid_pretrain)
                else:
                    reid.init_cfg = None

        if detector is not None:
            self.detector = build_detector(detector)

        if reid is not None:
            self.reid = build_reid(reid)

        if motion is not None:
            self.motion = build_motion(motion)

        if tracker is not None:
            self.tracker = build_tracker(tracker)

    def forward_train(self, *args, **kwargs):
        """Forward function during training."""
        raise NotImplementedError(
            'Please train `detector` and `reid` models firstly, then \
                inference with SORT/DeepSORT.')

    def simple_test(self,
                    img,
                    img_metas,
                    rescale=False,
                    public_bboxes=None,
                    **kwargs):
        """Test without augmentations.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            rescale (bool, optional): If False, then returned bboxes and masks
                will fit the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to False.
            public_bboxes (list[Tensor], optional): Public bounding boxes from
                the benchmark. Defaults to None.

        Returns:
            dict[str : list(ndarray)]: The tracking results.
        """
        frame_id = img_metas[0].get('frame_id', -1)
        if frame_id == 0:
            self.tracker.reset()

        x = self.detector.extract_feat(img)
        if hasattr(self.detector, 'roi_head'):
            # TODO: check whether this is the case
            if public_bboxes is not None:
                public_bboxes = [_[0] for _ in public_bboxes]
                proposals = public_bboxes
            else:
                proposals = self.detector.rpn_head.simple_test_rpn(
                    x, img_metas)
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
            outs = self.detector.bbox_head(x)
            result_list = self.detector.bbox_head.get_bboxes(
                *outs, img_metas=img_metas, rescale=rescale)
            # TODO: support batch inference
            det_bboxes = result_list[0][0]
            det_labels = result_list[0][1]
            num_classes = self.detector.bbox_head.num_classes
        else:
            raise TypeError('detector must has roi_head or bbox_head.')

        track_bboxes, track_labels, track_ids = self.tracker.track(
            img=img,
            img_metas=img_metas,
            model=self,
            feats=x,
            bboxes=det_bboxes,
            labels=det_labels,
            frame_id=frame_id,
            rescale=rescale,
            **kwargs)

        track_results = outs2results(
            bboxes=track_bboxes,
            labels=track_labels,
            ids=track_ids,
            num_classes=num_classes)
        det_results = outs2results(
            bboxes=det_bboxes, labels=det_labels, num_classes=num_classes)

        return dict(
            det_bboxes=det_results['bbox_results'],
            track_bboxes=track_results['bbox_results'])
