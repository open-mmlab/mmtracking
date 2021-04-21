from mmdet.core import bbox2result

from mmtrack.core import track2result
from ..builder import (MODELS, build_detector, build_motion, build_reid,
                       build_tracker)
from ..motion import CameraMotionCompensation, LinearMotion
from .base import BaseMultiObjectTracker
import torch

@MODELS.register_module()
class Tracktor(BaseMultiObjectTracker):
    """Tracking without bells and whistles.

    Details can be found at `Tracktor<https://arxiv.org/abs/1903.05625>`_.
    """

    def __init__(self,
                 detector=None,
                 reid=None,
                 tracker=None,
                 motion=None,
                 pretrains=None):
        super().__init__()
        if detector is not None:
            self.detector = build_detector(detector)

        if reid is not None:
            self.reid = build_reid(reid)

        if motion is not None:
            self.motion = build_motion(motion)
            if not isinstance(self.motion, list):
                self.motion = [self.motion]
            for m in self.motion:
                if isinstance(m, CameraMotionCompensation):
                    self.cmc = m
                if isinstance(m, LinearMotion):
                    self.linear_motion = m

        if tracker is not None:
            self.tracker = build_tracker(tracker)

        self.init_weights(pretrains)

    def init_weights(self, pretrain):
        """Initialize the weights of the modules.

        Args:
            pretrained (dict): Path to pre-trained weights.
        """
        if pretrain is None:
            pretrain = dict()
        assert isinstance(pretrain, dict), '`pretrain` must be a dict.'
        if self.with_detector and pretrain.get('detector', False):
            self.init_module('detector', pretrain['detector'])
        if self.with_reid and pretrain.get('reid', False):
            self.init_module('reid', pretrain['reid'])

    @property
    def with_cmc(self):
        """bool: whether the framework has a camera model compensation
                model.
        """
        return hasattr(self, 'cmc') and self.cmc is not None

    @property
    def with_linear_motion(self):
        """bool: whether the framework has a linear motion model."""
        return hasattr(self,
                       'linear_motion') and self.linear_motion is not None

    def forward_train(self, *args, **kwargs):
        """Forward function during training."""
        raise NotImplementedError(
            'Please train `detector` and `reid` models first and \
                inference with Tracktor.')

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
            if (proposals[0].size(0) != 0):
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
            else:
                det_bboxes = torch.empty((0, 5))
                det_labels = torch.empty((0))
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

        bboxes, labels, ids = self.tracker.track(
            img=img,
            img_metas=img_metas,
            model=self,
            feats=x,
            bboxes=det_bboxes,
            labels=det_labels,
            frame_id=frame_id,
            rescale=rescale,
            **kwargs)

        track_result = track2result(bboxes, labels, ids, num_classes)
        bbox_result = bbox2result(det_bboxes, det_labels, num_classes)
        return dict(bbox_results=bbox_result, track_results=track_result)
