# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import Dict, List, Tuple, Union

import addict
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmengine.model import BaseModel
from mmengine.structures import InstanceData
from torch import Tensor

from mmtrack.evaluation import bbox2region
from mmtrack.structures.bbox import calculate_region_overlap, quad2bbox_cxcywh
from mmtrack.utils import (ForwardResults, InstanceList, OptConfigType,
                           OptMultiConfig, OptSampleList, SampleList)


class BaseSingleObjectTracker(BaseModel, metaclass=ABCMeta):
    """Base class for single object tracker.

    Args:
        data_preprocessor (dict or ConfigDict, optional): The pre-process
           config of :class:`TrackDataPreprocessor`.  it usually includes,
            ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or list[dict]): Initialization config dict.
    """

    def __init__(self,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

    def freeze_module(self, module: Union[List[str], Tuple[str], str]) -> None:
        """Freeze module during training."""
        if isinstance(module, str):
            modules = [module]
        else:
            if not (isinstance(module, list) or isinstance(module, tuple)):
                raise TypeError('module must be a str or a list.')
            else:
                modules = module
        for module in modules:
            m = getattr(self, module)
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    @property
    def with_backbone(self):
        """bool: whether the framework has a backbone"""
        return hasattr(self, 'backbone') and self.backbone is not None

    @property
    def with_neck(self):
        """bool: whether the framework has a neck"""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_head(self):
        """bool: whether the framework has a head"""
        return hasattr(self, 'head') and self.head is not None

    def forward(self,
                inputs: Dict[str, Tensor],
                data_samples: OptSampleList = None,
                mode: str = 'predict',
                **kwargs) -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`TrackDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (Dict[str, Tensor]): of shape (N, T, C, H, W)
                encoding input images. Typically these should be mean centered
                and std scaled. The N denotes batch size. The T denotes the
                number of key/reference frames.
                - img (Tensor) : The key images.
                - ref_img (Tensor): The reference images.
            data_samples (list[:obj:`TrackDataSample`], optional): The
                annotation data of every samples. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'predict'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`TrackDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples, **kwargs)
        elif mode == 'predict':
            return self.predict(inputs, data_samples, **kwargs)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    @abstractmethod
    def loss(self, inputs: Dict[str, Tensor], data_samples: SampleList,
             **kwargs) -> Union[dict, tuple]:
        """Calculate losses from a batch of inputs and data samples."""
        pass

    def predict(self, inputs: Dict[str, Tensor], data_samples: SampleList,
                **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (dict[Tensor]): of shape (N, T, C, H, W)
                encodingbinput images. Typically these should be mean centered
                and stdbscaled. The N denotes batch size. The T denotes the
                number of key/reference frames.
                - img (Tensor) : The key images.
                - ref_img (Tensor): The reference images.
                In test mode, T = 1 and there is only ``img`` and no
                ``ref_img``.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as ``gt_instances`` and ``metainfo``.

        Returns:
            list[obj:`TrackDataSample`]: Tracking results of the
                input images. Each TrackDataSample usually contains
                ``pred_det_instances`` or ``pred_track_instances``.
        """
        test_mode = self.test_cfg.get('test_mode', 'OPE')
        assert test_mode in ['OPE', 'VOT']
        if test_mode == 'VOT':
            pred_track_instances = self.predict_vot(inputs, data_samples)
        else:
            pred_track_instances = self.predict_ope(inputs, data_samples)
        track_data_samples = deepcopy(data_samples)
        for _data_sample, _pred_instances in zip(track_data_samples,
                                                 pred_track_instances):
            _data_sample.pred_track_instances = _pred_instances
        return track_data_samples

    def predict_ope(self, inputs: dict, data_samples: SampleList):

        img = inputs['img']
        assert img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert img.size(0) == 1, \
            'Only support 1 batch size per gpu in test mode'
        img = img[0]

        metainfo = data_samples[0].metainfo
        frame_id = metainfo.get('frame_id', -1)
        assert frame_id >= 0

        if frame_id == 0:
            # We store the current frame id, previous bbox and template
            # information in ``self.memo``.
            self.memo = addict.Dict()
            self.memo.frame_id = frame_id
            gt_bboxes = data_samples[0].gt_instances['bboxes']
            self.memo.bbox = quad2bbox_cxcywh(gt_bboxes)
            self.init(img)
            results = [InstanceData()]
            results[0].bboxes = bbox_cxcywh_to_xyxy(self.memo.bbox)[None]
            results[0].scores = gt_bboxes.new_tensor([-1.])
        else:
            self.memo.frame_id = frame_id
            results = self.track(img, data_samples)
            self.memo.bbox = bbox_xyxy_to_cxcywh(results[0].bboxes.squeeze())

        return results

    def predict_vot(self, inputs: dict,
                    data_samples: SampleList) -> List[InstanceData]:
        """Test using VOT test mode.

        Args:
            inputs (Dict[str, Tensor]): of shape (N, T, C, H, W)
                encoding input images. Typically these should be mean centered
                and std scaled. The N denotes batch size. The T denotes the
                number of key/reference frames.
                - img (Tensor) : The key images.
                - ref_img (Tensor): The reference images.
                In test mode, T = 1 and there is only ``img`` and no
                ``ref_img``.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as ``gt_instances`` and 'metainfo'.

        Returns:
            List[:obj:`InstanceData`]: Object tracking results of each image
            after the post process. Each item usually contains following keys.
                - scores (Tensor): Classification scores, has a shape (1, )
                - bboxes (Tensor): Has a shape (1, 4),
                  the last dimension 4 arrange as [x1, y1, x2, y2].
        """
        img = inputs['img']
        assert img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert img.size(0) == 1, \
            'Only support 1 batch size per gpu in test mode'
        img = img[0]

        gt_bboxes = data_samples[0].gt_instances['bboxes']
        metainfo = data_samples[0].metainfo
        frame_id = metainfo.get('frame_id', -1)
        assert frame_id >= 0

        if frame_id == 0:
            self.init_frame_id = 0
        if self.init_frame_id == frame_id:
            # initialization
            # We store the previous bbox、 current frame id and some template
            # information in ``self.memo``.
            self.memo = addict.Dict()
            self.memo.frame_id = frame_id
            self.memo.bbox = quad2bbox_cxcywh(gt_bboxes)
            self.init(img)
            # 1 denotes the initialization state
            results = [InstanceData()]
            results[0].bboxes = gt_bboxes.new_tensor([[1.]])
            results[0].scores = gt_bboxes.new_tensor([-1.])
        elif self.init_frame_id > frame_id:
            # 0 denotes unknown state, namely the skipping frame after failure
            self.memo.frame_id = frame_id
            results = [InstanceData()]
            results[0].bboxes = gt_bboxes.new_tensor([[0.]])
            results[0].scores = gt_bboxes.new_tensor([-1.])
        else:
            # normal tracking state
            self.memo.frame_id = frame_id
            results = self.track(img, data_samples)
            self.memo.bbox = bbox_xyxy_to_cxcywh(results[0].bboxes.squeeze())

            # convert bbox to region for overlap calculation
            track_bbox = results[0].bboxes[0].cpu().numpy()
            track_region = bbox2region(track_bbox)
            gt_region = bbox2region(gt_bboxes[0].cpu().numpy())

            if 'img_shape' in metainfo:
                image_shape = metainfo['img_shape']
                image_wh = (image_shape[1], image_shape[0])
            else:
                image_wh = None
                Warning('image shape are need when calculating bbox overlap')
            overlap = calculate_region_overlap(
                track_region, gt_region, bounds=image_wh)
            if overlap <= 0:
                # tracking failure
                self.init_frame_id = frame_id + 5
                # 2 denotes the failure state
                results[0].bboxes = img.new_tensor([[2.]])

        return results

    @abstractmethod
    def init(img: Tensor):
        """Initialize the single object tracker in the first frame.

        Args:
            img (Tensor): of shape (1, C, H, W) encoding original input
                image.
        """
        pass

    @abstractmethod
    def track(img: Tensor, data_samples: SampleList) -> InstanceList:
        """Track the box of previous frame to current frame `img`.

        Args:
            img (Tensor): of shape (1, C, H, W) encoding original input
                image.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as ``gt_instances`` and 'metainfo'.

        Returns:
            InstanceList: Tracking results of each image after the postprocess.
                - scores: a Tensor denoting the score of best_bbox.
                - bboxes: a Tensor of shape (4, ) in [x1, x2, y1, y2]
                format, and denotes the best tracked bbox in current frame.
        """
        pass

    def _forward(self,
                 inputs: Dict[str, Tensor],
                 data_samples: OptSampleList = None,
                 **kwargs):
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            inputs (Dict[str, Tensor]): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`TrackDataSample`], optional): The
                Data Samples. It usually includes information such as
                `gt_instance`.

        Returns:
            tuple[list]: A tuple of features from ``head`` forward.
        """
        raise NotImplementedError(
            "_forward function (namely 'tensor' mode) is not supported now")
